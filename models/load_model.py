# Day 6 Model Loading Script
import tensorflow as tf
import json

with open('models/model_config.json') as f:
    config = json.load(f)

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
            self.user_lookup.vocabulary_size(), self.embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.video_embedding = tf.keras.layers.Embedding(
            self.video_lookup.vocabulary_size(), self.embedding_dim,
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

with open('models/user_vocab.json') as f:
    user_vocab = json.load(f)

with open('models/video_vocab.json') as f:
    video_vocab = json.load(f)

model = TwoTowerModel(embedding_dim=config['embedding_dim'])
model.adapt_vocabularies(tf.constant(user_vocab), tf.constant(video_vocab))

_ = model.user_tower(tf.constant(['user_0']))
_ = model.video_tower(tf.constant(['video_0']))

model.load_weights('models/two_tower_weights.weights.h5')

print("✅ Model loaded and ready for inference")
