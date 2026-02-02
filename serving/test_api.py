"""
serving/test_api.py
Quick smoke tests for API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint."""
    print("\n" + "=" * 70)
    print("TEST 1: Health Check")
    print("=" * 70)

    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check passed")

def test_risk_endpoint():
    """Test risk endpoint with known organic user."""
    print("\n" + "=" * 70)
    print("TEST 2: Risk Endpoint (Organic User)")
    print("=" * 70)

    response = requests.get(f"{BASE_URL}/risk/user_100")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    data = response.json()
    assert "fraud_score" in data
    assert "fraud_tier" in data
    assert "manipulation_threshold" in data
    print("✅ Risk endpoint passed")

def test_feed_basic():
    """Test feed endpoint without stats."""
    print("\n" + "=" * 70)
    print("TEST 3: Feed Endpoint (Basic)")
    print("=" * 70)

    payload = {
        "user_id": "user_100",
        "num_videos": 10,
        "include_stats": False
    }

    response = requests.post(f"{BASE_URL}/feed", json=payload)
    print(f"Status: {response.status_code}")
    data = response.json()

    print(f"User: {data['user_id']}")
    print(f"Fraud Score: {data['fraud_score']}")
    print(f"Fraud Tier: {data['fraud_tier']}")
    print(f"Videos Returned: {len(data['videos'])}")

    if len(data['videos']) > 0:
        print(f"\nFirst 3 videos:")
        for video in data['videos'][:3]:
            print(f"  Rank {video['rank']}: {video['video_id']} (score: {video['score']:.4f})")

    assert response.status_code == 200
    assert len(data['videos']) <= 10
    print("✅ Feed endpoint (basic) passed")

def test_feed_with_stats():
    """Test feed endpoint with stats."""
    print("\n" + "=" * 70)
    print("TEST 4: Feed Endpoint (With Stats)")
    print("=" * 70)

    payload = {
        "user_id": "user_100",
        "num_videos": 20,
        "include_stats": True
    }

    response = requests.post(f"{BASE_URL}/feed", json=payload)
    print(f"Status: {response.status_code}")
    data = response.json()

    print(f"\nFunnel Statistics:")
    stats = data['stats']
    print(f"  Retrieved: {stats['retrieved']}")
    print(f"  After Safety: {stats['after_safety']}")
    print(f"  After Manipulation: {stats['after_manipulation']}")
    print(f"  Final Returned: {stats['final_returned']}")
    print(f"  Removed by Top-N: {stats['removed_by_top_n']}")

    print(f"\nBlocked Counts:")
    print(f"  Safety: {stats['blocked_safety']}")
    print(f"  Manipulation: {stats['blocked_manipulation']}")
    print(f"  Unknown: {stats['blocked_unknown']}")

    print(f"\nThresholds:")
    print(f"  NSFW: {stats['thresholds']['nsfw']}")
    print(f"  Violence: {stats['thresholds']['violence']}")
    print(f"  Hate Speech: {stats['thresholds']['hate_speech']}")
    print(f"  Manipulation: {stats['thresholds']['manipulation']}")

    print(f"\nFraud Tier: {stats['fraud_tier']}")

    assert response.status_code == 200
    assert stats is not None
    print("✅ Feed endpoint (with stats) passed")

def test_unknown_user():
    """Test feed endpoint with unknown user (should use default fraud score)."""
    print("\n" + "=" * 70)
    print("TEST 5: Unknown User (Default Fraud Score)")
    print("=" * 70)

    payload = {
        "user_id": "unknown_user_999999",
        "num_videos": 10,
        "include_stats": False
    }

    response = requests.post(f"{BASE_URL}/feed", json=payload)
    print(f"Status: {response.status_code}")
    data = response.json()

    print(f"Fraud Score: {data['fraud_score']}")
    print(f"Fraud Tier: {data['fraud_tier']}")
    print(f"Videos Returned: {len(data['videos'])}")

    assert response.status_code == 200
    assert data['fraud_score'] == 0.05  # Default for unknown users
    print("✅ Unknown user test passed")

def test_fraud_score_override():
    """Test fraud score override for high-risk user."""
    print("\n" + "=" * 70)
    print("TEST 6: Fraud Score Override (High Risk)")
    print("=" * 70)

    payload = {
        "user_id": "user_100",
        "num_videos": 20,
        "include_stats": True,
        "fraud_score_override": 0.9  # Force high-risk
    }

    response = requests.post(f"{BASE_URL}/feed", json=payload)
    print(f"Status: {response.status_code}")
    data = response.json()

    print(f"Fraud Score (override): {data['fraud_score']}")
    print(f"Fraud Tier: {data['fraud_tier']}")
    print(f"Manipulation Threshold: {data['stats']['thresholds']['manipulation']}")

    assert response.status_code == 200
    assert data['fraud_score'] == 0.9
    assert data['fraud_tier'] == 'high'
    assert data['stats']['thresholds']['manipulation'] == 0.3  # Strictest threshold
    print("✅ Fraud score override test passed")

def run_all_tests():
    """Run all smoke tests."""
    print("\n" + "=" * 70)
    print("RUNNING API SMOKE TESTS")
    print("=" * 70)

    try:
        test_health()
        test_risk_endpoint()
        test_feed_basic()
        test_feed_with_stats()
        test_unknown_user()
        test_fraud_score_override()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    run_all_tests()
