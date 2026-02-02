"""
serving/integrity/policy.py
Multi-layer content integrity filtering with adaptive thresholds
"""

from typing import List, Tuple, Dict, Optional

class IntegrityPolicy:
    """
    Deterministic multi-layer filtering for content integrity.
    Filters candidates based on content safety and manipulation risk.
    """

    NSFW_THRESHOLD = 0.5
    VIOLENCE_THRESHOLD = 0.7
    HATE_SPEECH_THRESHOLD = 0.6

    MANIPULATION_LOW_THRESHOLD = 0.7
    MANIPULATION_MED_THRESHOLD = 0.5
    MANIPULATION_HIGH_THRESHOLD = 0.3

    FRAUD_LOW_TIER = 0.3
    FRAUD_HIGH_TIER = 0.7

    def __init__(self, video_meta_map: Dict[str, Dict]):
        self.video_meta_map = video_meta_map

    def get_manipulation_threshold(self, fraud_score: float) -> float:
        if fraud_score >= self.FRAUD_HIGH_TIER:
            return self.MANIPULATION_HIGH_THRESHOLD
        elif fraud_score >= self.FRAUD_LOW_TIER:
            return self.MANIPULATION_MED_THRESHOLD
        else:
            return self.MANIPULATION_LOW_THRESHOLD

    def is_safe_content(self, video_id: str) -> bool:
        meta = self.video_meta_map.get(video_id)
        if meta is None:
            return False

        if meta['nsfw_prob'] >= self.NSFW_THRESHOLD:
            return False
        if meta['violence_prob'] >= self.VIOLENCE_THRESHOLD:
            return False
        if meta['hate_speech_prob'] >= self.HATE_SPEECH_THRESHOLD:
            return False

        return True

    def is_manipulation_safe(self, video_id: str, fraud_score: float) -> bool:
        meta = self.video_meta_map.get(video_id)
        if meta is None:
            return False

        threshold = self.get_manipulation_threshold(fraud_score)
        return meta['manipulation_score'] < threshold

    def filter_candidates(self, candidates: List[Tuple[str, float]], fraud_score: float) -> List[Tuple[str, float]]:
        filtered = []
        for video_id, score in candidates:
            if not self.is_safe_content(video_id):
                continue
            if not self.is_manipulation_safe(video_id, fraud_score):
                continue
            filtered.append((video_id, score))
        return filtered

    def filter_candidates_with_stats(self, candidates: List[Tuple[str, float]], fraud_score: float, num_requested: int) -> Tuple[List[Tuple[str, float]], Dict]:
        manipulation_threshold = self.get_manipulation_threshold(fraud_score)

        stats = {
            'retrieved': len(candidates),
            'after_safety': 0,
            'after_manipulation': 0,
            'final_returned': 0,
            'removed_by_top_n': 0,
            'blocked_safety': 0,
            'blocked_manipulation': 0,
            'blocked_unknown': 0,
            'thresholds': {
                'nsfw': self.NSFW_THRESHOLD,
                'violence': self.VIOLENCE_THRESHOLD,
                'hate_speech': self.HATE_SPEECH_THRESHOLD,
                'manipulation': manipulation_threshold,
            },
            'fraud_tier': self.get_fraud_tier(fraud_score),
        }

        after_safety = []
        for video_id, score in candidates:
            meta = self.video_meta_map.get(video_id)
            if meta is None:
                stats['blocked_unknown'] += 1
                continue

            if (meta['nsfw_prob'] >= self.NSFW_THRESHOLD or
                meta['violence_prob'] >= self.VIOLENCE_THRESHOLD or
                meta['hate_speech_prob'] >= self.HATE_SPEECH_THRESHOLD):
                stats['blocked_safety'] += 1
                continue

            after_safety.append((video_id, score))

        stats['after_safety'] = len(after_safety)

        filtered = []
        for video_id, score in after_safety:
            meta = self.video_meta_map[video_id]
            if meta['manipulation_score'] >= manipulation_threshold:
                stats['blocked_manipulation'] += 1
                continue
            filtered.append((video_id, score))

        stats['after_manipulation'] = len(filtered)

        final = filtered[:num_requested]
        stats['final_returned'] = len(final)
        stats['removed_by_top_n'] = len(filtered) - len(final)

        return final, stats

    def get_fraud_tier(self, fraud_score: float) -> str:
        if fraud_score < self.FRAUD_LOW_TIER:
            return 'low'
        elif fraud_score < self.FRAUD_HIGH_TIER:
            return 'medium'
        else:
            return 'high'
