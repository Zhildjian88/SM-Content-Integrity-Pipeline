"""
serving/api/app.py
FastAPI serving endpoints for content recommendation and fraud scoring
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager

from serving.model_loader import registry

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    print("\n🚀 Starting API server...")
    registry.initialize()
    print("✅ Server ready to accept requests\n")
    yield
    print("\n👋 Shutting down API server...")

app = FastAPI(
    title="iGaming Content Integrity API",
    description="Content recommendation with multi-layer integrity filtering",
    version="1.0.0",
    lifespan=lifespan
)

class FeedRequest(BaseModel):
    user_id: str
    num_videos: int = 20
    include_stats: bool = False
    fraud_score_override: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override fraud score for testing")

class VideoItem(BaseModel):
    video_id: str
    score: float
    rank: int

class FeedResponse(BaseModel):
    user_id: str
    fraud_score: float
    fraud_tier: str
    videos: List[VideoItem]
    stats: Optional[dict] = None

class RiskResponse(BaseModel):
    user_id: str
    fraud_score: float
    fraud_tier: str
    manipulation_threshold: float

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "iGaming Content Integrity API",
        "version": "1.0.0"
    }

@app.get("/risk/{user_id}", response_model=RiskResponse)
async def get_user_risk(user_id: str):
    """Get user fraud risk score and tier."""
    fraud_score = registry.get_user_fraud_score(user_id)
    fraud_tier = registry.policy.get_fraud_tier(fraud_score)
    manipulation_threshold = registry.policy.get_manipulation_threshold(fraud_score)

    return RiskResponse(
        user_id=user_id,
        fraud_score=fraud_score,
        fraud_tier=fraud_tier,
        manipulation_threshold=manipulation_threshold
    )

@app.post("/feed", response_model=FeedResponse)
async def get_personalized_feed(request: FeedRequest):
    """Get personalized video feed with integrity filtering."""
    user_id = request.user_id
    num_videos = request.num_videos
    include_stats = request.include_stats

    retrieval_k = num_videos * 5

    try:
        candidates = registry.retrieve_candidates(user_id, top_k=retrieval_k)

        if request.fraud_score_override is not None:
            fraud_score = max(0.0, min(1.0, request.fraud_score_override))
        else:
            fraud_score = registry.get_user_fraud_score(user_id)

        if include_stats:
            filtered_candidates, stats = registry.policy.filter_candidates_with_stats(
                candidates, fraud_score, num_requested=num_videos
            )
        else:
            filtered_candidates = registry.policy.filter_candidates(candidates, fraud_score)
            stats = None

        top_n = filtered_candidates[:num_videos]

        videos = [
            VideoItem(video_id=video_id, score=score, rank=i + 1)
            for i, (video_id, score) in enumerate(top_n)
        ]

        fraud_tier = registry.policy.get_fraud_tier(fraud_score)

        return FeedResponse(
            user_id=user_id,
            fraud_score=fraud_score,
            fraud_tier=fraud_tier,
            videos=videos,
            stats=stats
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating feed: {str(e)}")
