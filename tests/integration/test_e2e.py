"""
DAY 8 - INTEGRATION TESTS (E2E)
Test all API endpoints with edge cases
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"
FLOAT_TOLERANCE = 1e-6

class TestE2E:
    """End-to-end integration tests for API endpoints."""

    def test_health_check(self):
        """Test 1: Health check returns 200."""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        print("✅ Test 1: Health check passed")

    def test_risk_endpoint_known_user(self):
        """Test 2: /risk returns score + tier for known user."""
        response = requests.get(f"{BASE_URL}/risk/user_100")
        assert response.status_code == 200
        data = response.json()
        assert "fraud_score" in data
        assert "fraud_tier" in data
        assert "manipulation_threshold" in data
        assert data["fraud_tier"] in ["low", "medium", "high"]
        print(f"✅ Test 2: Risk endpoint - fraud_score={data['fraud_score']}, tier={data['fraud_tier']}")

    def test_feed_basic(self):
        """Test 3: /feed returns num_videos items (or fewer, no errors)."""
        payload = {"user_id": "user_100", "num_videos": 20, "include_stats": False}
        response = requests.post(f"{BASE_URL}/feed", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "videos" in data
        assert len(data["videos"]) <= 20
        assert "fraud_score" in data
        assert "fraud_tier" in data
        print(f"✅ Test 3: Feed basic - returned {len(data['videos'])} videos")

    def test_unknown_user_defaults(self):
        """Test 4: Unknown user defaults to fraud score 0.05."""
        payload = {"user_id": "unknown_user_999999", "num_videos": 10, "include_stats": False}
        response = requests.post(f"{BASE_URL}/feed", json=payload)
        assert response.status_code == 200
        data = response.json()
        # FIXED: Use tolerance for float comparison
        assert abs(data["fraud_score"] - 0.05) < FLOAT_TOLERANCE
        assert data["fraud_tier"] == "low"
        print(f"✅ Test 4: Unknown user - fraud_score={data['fraud_score']} (default)")

    def test_fraud_override_extreme_low(self):
        """Test 5a: Override fraud score at extreme low (0.0)."""
        payload = {"user_id": "user_100", "num_videos": 20, "fraud_score_override": 0.0, "include_stats": True}
        response = requests.post(f"{BASE_URL}/feed", json=payload)
        assert response.status_code == 200
        data = response.json()
        # FIXED: Use tolerance
        assert abs(data["fraud_score"] - 0.0) < FLOAT_TOLERANCE
        assert data["fraud_tier"] == "low"
        # FIXED: Check stats exist before accessing
        assert "stats" in data
        assert "thresholds" in data["stats"]
        assert abs(data["stats"]["thresholds"]["manipulation"] - 0.7) < FLOAT_TOLERANCE
        print(f"✅ Test 5a: Override 0.0 - threshold={data['stats']['thresholds']['manipulation']}")

    def test_fraud_override_extreme_high(self):
        """Test 5b: Override fraud score at extreme high (1.0)."""
        payload = {"user_id": "user_100", "num_videos": 20, "fraud_score_override": 1.0, "include_stats": True}
        response = requests.post(f"{BASE_URL}/feed", json=payload)
        assert response.status_code == 200
        data = response.json()
        # FIXED: Use tolerance
        assert abs(data["fraud_score"] - 1.0) < FLOAT_TOLERANCE
        assert data["fraud_tier"] == "high"
        assert "stats" in data
        assert "thresholds" in data["stats"]
        assert abs(data["stats"]["thresholds"]["manipulation"] - 0.3) < FLOAT_TOLERANCE
        print(f"✅ Test 5b: Override 1.0 - threshold={data['stats']['thresholds']['manipulation']}")

    def test_empty_result_resilience(self):
        """Test 6: Empty filtering result (no crash, returns empty list + stats)."""
        # Request very few videos with high-risk override - may result in empty feed
        payload = {"user_id": "user_100", "num_videos": 5, "fraud_score_override": 1.0, "include_stats": True}
        response = requests.post(f"{BASE_URL}/feed", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "videos" in data
        assert isinstance(data["videos"], list)
        assert "stats" in data
        # May be empty if filtering is aggressive
        print(f"✅ Test 6: Empty result resilience - returned {len(data['videos'])} videos (OK if 0)")

    def test_feed_with_stats(self):
        """Test 7: /feed with stats returns complete funnel."""
        payload = {"user_id": "user_100", "num_videos": 20, "include_stats": True}
        response = requests.post(f"{BASE_URL}/feed", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        stats = data["stats"]
        assert "retrieved" in stats
        assert "after_safety" in stats
        assert "after_manipulation" in stats
        assert "final_returned" in stats
        assert "thresholds" in stats
        print(f"✅ Test 7: Feed with stats - funnel complete")


def run_all_tests():
    """Run all integration tests and capture results."""

    print("=" * 70)
    print("DAY 8: INTEGRATION TESTS (E2E)")
    print("=" * 70)

    test_suite = TestE2E()
    results = []

    tests = [
        ("Health Check", test_suite.test_health_check),
        ("Risk Endpoint", test_suite.test_risk_endpoint_known_user),
        ("Feed Basic", test_suite.test_feed_basic),
        ("Unknown User Defaults", test_suite.test_unknown_user_defaults),
        ("Override Extreme Low (0.0)", test_suite.test_fraud_override_extreme_low),
        ("Override Extreme High (1.0)", test_suite.test_fraud_override_extreme_high),
        ("Empty Result Resilience", test_suite.test_empty_result_resilience),
        ("Feed with Stats", test_suite.test_feed_with_stats),
    ]

    for test_name, test_func in tests:
        try:
            test_func()
            results.append({"test": test_name, "status": "PASS"})
        except Exception as e:
            print(f"❌ Test failed: {test_name} - {str(e)}")
            results.append({"test": test_name, "status": "FAIL", "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")

    if failed == 0:
        print("\n✅ ALL TESTS PASSED")
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")

    # Save results
    Path("docs").mkdir(exist_ok=True)

    # Capture sample calls for documentation
    sample_calls = {}
    try:
        sample_calls["health"] = requests.get(f"{BASE_URL}/").json()
        sample_calls["risk_user_100"] = requests.get(f"{BASE_URL}/risk/user_100").json()
        sample_calls["feed_basic"] = requests.post(
            f"{BASE_URL}/feed",
            json={"user_id": "user_100", "num_videos": 20, "include_stats": True}
        ).json()
    except Exception as e:
        print(f"⚠️  Warning: Could not capture sample calls - {str(e)}")

    with open("docs/integration_test_results.json", "w") as f:
        json.dump({
            "test_results": results,
            "summary": {"total": len(results), "passed": passed, "failed": failed},
            "sample_calls": sample_calls
        }, f, indent=2)

    print("\n✅ Results saved: docs/integration_test_results.json")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
