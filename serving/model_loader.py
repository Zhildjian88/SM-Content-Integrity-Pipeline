"""
serving/model_loader.py
Central registry for loading all models and data at startup
"""

import tensorflow as tf
import polars as pl
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class ModelRegistry:
    """
    Singleton registry that loads everything once at startup.
    Provides O(1) lookups for serving.
    """

    def __init__(self):
        self.retrieval_model = None
        self.video_embeddings = None
        self.video_ids = None
        self.video_meta_map = {}
        self.fraud_scores_map = {}
        self.policy = None
        self._initialized = False

    def initialize(self):
        """Load all models and data. Call once at startup."""
        if self._initialized:
            print("Registry already initialized")
            return

        print("=" * 70)
        print("INITIALIZING MODEL REGISTRY")
        print("=" * 70)

        self._load_retrieval_model()
        self._load_video_data()
        self._load_fraud_scores()
        self._initialize_policy()

        self._initialized = True
        print("\n✅ Registry initialized successfully")

    def _load_retrieval_model(self):
        """Load Two-Tower model from weights."""
        print("\n1. Loading retrieval model...")

        class TwoTowerModel(tf.keras.Model):
            def __init__(self, embedding_dim=128):
                super().__init__()
                self.embedding_dim = embedding_dim

                self.user_lookup = tf.keras.layers.StringLookup(mask_token=None, num_oov_indices=1)
                self.user_embedding = None
                self.user_dense1 = tf.keras.layers.Dense(256, activation='relu')
                self.user_dense2 = tf.keras.layers.Dense(embedding_dim)

                self.video_lookup = tf.keras.layers.StringLookup(mask_token=None, num_oov_indices=1)
                self.video_embedding = None
                self.video_dense1 = tf.keras.layers.Dense(256, activation='relu')
                self.video_dense2 = tf.keras.layers.Dense(embedding_dim)

                self._towers_built = False

            def adapt_vocabularies(self, user_vocab, video_vocab):
                self.user_lookup.adapt(user_vocab)
                self.video_lookup.adapt(video_vocab)

                self.user_embedding = tf.keras.layers.Embedding(
                    self.user_lookup.vocabulary_size(),
                    self.embedding_dim,
                    embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
                )
                self.video_embedding = tf.keras.layers.Embedding(
                    self.video_lookup.vocabulary_size(),
                    self.embedding_dim,
                    embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
                )

                self._towers_built = True

            def user_tower(self, user_ids):
                x = self.user_lookup(user_ids)
                x = self.user_embedding(x)
                x = self.user_dense1(x)
                x = self.user_dense2(x)
                return tf.nn.l2_normalize(x, axis=1)

            def video_tower(self, video_ids):
                x = self.video_lookup(video_ids)
                x = self.video_embedding(x)
                x = self.video_dense1(x)
                x = self.video_dense2(x)
                return tf.nn.l2_normalize(x, axis=1)

            def call(self, inputs):
                user_emb = self.user_tower(inputs['user_id'])
                video_emb = self.video_tower(inputs['video_id'])
                return user_emb, video_emb

        with open('models/model_config.json') as f:
            config = json.load(f)

        print(f"  Embedding dim: {config['embedding_dim']}")

        with open('models/user_vocab.json') as f:
            user_vocab = json.load(f)

        with open('models/video_vocab.json') as f:
            video_vocab = json.load(f)

        print(f"  User vocab: {len(user_vocab):,}")
        print(f"  Video vocab: {len(video_vocab):,}")

        self.retrieval_model = TwoTowerModel(embedding_dim=config['embedding_dim'])
        self.retrieval_model.adapt_vocabularies(tf.constant(user_vocab), tf.constant(video_vocab))

        dummy_batch = {'user_id': tf.constant(['user_0']), 'video_id': tf.constant(['video_0'])}
        _ = self.retrieval_model(dummy_batch)

        self.retrieval_model.load_weights('models/two_tower_weights.weights.h5')

        print("  ✅ Model loaded from weights")

    def _load_video_data(self):
        """Load video embeddings and metadata."""
        print("\n2. Loading video data...")

        self.video_embeddings = np.load('features/video_embeddings.npy')

        with open('features/video_ids.json') as f:
            self.video_ids = json.load(f)

        print(f"  ✅ Video embeddings: {self.video_embeddings.shape}")
        print(f"  ✅ Video IDs: {len(self.video_ids):,}")

        norms = np.linalg.norm(self.video_embeddings, axis=1)
        mean_norm = norms.mean()
        print(f"  Embedding L2 norms: mean={mean_norm:.4f}, min={norms.min():.4f}, max={norms.max():.4f}")

        if not np.allclose(mean_norm, 1.0, atol=0.01):
            print("  ⚠️  Embeddings not normalized, normalizing now...")
            self.video_embeddings = self.video_embeddings / norms[:, np.newaxis]
            print("  ✅ Embeddings normalized")

        assert len(self.video_ids) == self.video_embeddings.shape[0], "Video IDs and embeddings length mismatch!"

        videos = pl.read_parquet('data/videos.parquet')
        video_features = pl.read_parquet('data/video_features.parquet')

        video_meta = videos.join(video_features, on='video_id', how='left')

        video_meta = video_meta.with_columns([
            pl.col('nsfw_prob').fill_null(0.0),
            pl.col('violence_prob').fill_null(0.0),
            pl.col('hate_speech_prob').fill_null(0.0),
            pl.col('manipulation_score').fill_null(0.0),
        ])

        for row in video_meta.iter_rows(named=True):
            self.video_meta_map[row['video_id']] = {
                'creator_id': row['creator_id'],
                'duration_sec': row['duration_sec'],
                'nsfw_prob': row['nsfw_prob'],
                'violence_prob': row['violence_prob'],
                'hate_speech_prob': row['hate_speech_prob'],
                'manipulation_score': row['manipulation_score'],
            }

        print(f"  ✅ Video metadata: {len(self.video_meta_map):,} videos")

        sample_video = next(iter(self.video_meta_map.values()))
        assert 'manipulation_score' in sample_video, "manipulation_score missing from video metadata!"

    def _load_fraud_scores(self):
        """Load pre-computed user fraud scores."""
        print("\n3. Loading fraud scores...")

        fraud_scores = pl.read_parquet('features/fraud_scores_user.parquet')

        for row in fraud_scores.iter_rows(named=True):
            self.fraud_scores_map[row['user_id']] = row['fraud_prob']

        print(f"  ✅ Fraud scores: {len(self.fraud_scores_map):,} users")
        print(f"  Score range: [{min(self.fraud_scores_map.values()):.4f}, {max(self.fraud_scores_map.values()):.4f}]")

    def _initialize_policy(self):
        """Initialize integrity policy."""
        print("\n4. Initializing integrity policy...")

        from serving.integrity.policy import IntegrityPolicy

        self.policy = IntegrityPolicy(self.video_meta_map)

        print("  ✅ Policy initialized")

    def get_user_embedding(self, user_id: str) -> np.ndarray:
        """Compute user embedding on-demand."""
        user_emb = self.retrieval_model.user_tower(tf.constant([user_id]))
        user_emb = user_emb.numpy()[0]
        user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-12)
        return user_emb

    def retrieve_candidates(self, user_id: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve top-K candidate videos for user."""
        top_k = min(top_k, len(self.video_ids))
        user_emb = self.get_user_embedding(user_id)
        similarities = np.dot(self.video_embeddings, user_emb)

        top_k_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]

        candidates = [(self.video_ids[idx], float(similarities[idx])) for idx in top_k_indices]

        return candidates

    def get_user_fraud_score(self, user_id: str) -> float:
        """Get user fraud probability (O(1) lookup)."""
        return self.fraud_scores_map.get(user_id, 0.05)


registry = ModelRegistry()
