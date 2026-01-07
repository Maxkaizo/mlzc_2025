"""
Test script for Mushroom Classifier API
Tests both local and deployed endpoints
"""

import requests
import json
from typing import Optional

# Example mushroom data
SAMPLE_MUSHROOM = {
    "cap-diameter": 8.5,
    "stem-height": 7.2,
    "stem-width": 6.5,
    "cap-shape": "x",
    "cap-surface": "s",
    "cap-color": "n",
    "does-bruise-or-bleed": "f",
    "gill-attachment": "f",
    "gill-spacing": "c",
    "gill-color": "k",
    "stem-surface": "s",
    "stem-color": "w",
    "has-ring": "t",
    "ring-type": "p",
    "habitat": "d",
    "season": "s",
}

POISONOUS_MUSHROOM = {
    "cap-diameter": 10.0,
    "stem-height": 8.5,
    "stem-width": 7.0,
    "cap-shape": "b",
    "cap-surface": "f",
    "cap-color": "r",
    "does-bruise-or-bleed": "t",
    "gill-attachment": "a",
    "gill-spacing": "w",
    "gill-color": "w",
    "stem-surface": "f",
    "stem-color": "p",
    "has-ring": "t",
    "ring-type": "e",
    "habitat": "g",
    "season": "u",
}


def test_health_check(base_url: str) -> bool:
    """Test health check endpoint"""
    print("\n🏥 Testing health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ Health check passed: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_prediction(base_url: str, mushroom_data: dict, expected_class: Optional[str] = None) -> bool:
    """Test prediction endpoint"""
    print(f"\n🍄 Testing prediction...")
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=mushroom_data,
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Probability: {result['probability']}")
            print(f"   Confidence: {result['confidence_percent']}")

            if expected_class and result["prediction"] != expected_class:
                print(f"⚠️  Expected {expected_class}, got {result['prediction']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False


def test_batch_prediction(base_url: str, mushroom_list: list) -> bool:
    """Test batch prediction endpoint"""
    print(f"\n📦 Testing batch prediction ({len(mushroom_list)} samples)...")
    try:
        response = requests.post(
            f"{base_url}/batch_predict",
            json=mushroom_list,
            timeout=15,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch prediction successful!")
            print(f"   Processed: {result['count']} samples")
            for i, pred in enumerate(result["predictions"], 1):
                print(f"   Sample {i}: {pred['prediction']} ({pred['confidence_percent']})")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False


def test_documentation(base_url: str) -> bool:
    """Test API documentation endpoints"""
    print(f"\n📚 Testing documentation endpoints...")
    try:
        # Test Swagger UI
        response_swagger = requests.get(f"{base_url}/docs", timeout=5)
        # Test ReDoc
        response_redoc = requests.get(f"{base_url}/redoc", timeout=5)

        if response_swagger.status_code == 200 and response_redoc.status_code == 200:
            print("✅ Documentation endpoints available")
            print(f"   📖 Swagger UI: {base_url}/docs")
            print(f"   📖 ReDoc: {base_url}/redoc")
            return True
        else:
            print(f"❌ Documentation endpoints failed")
            return False
    except Exception as e:
        print(f"❌ Documentation error: {e}")
        return False


def run_full_test_suite(base_url: str = "http://localhost:8000"):
    """Run complete test suite"""
    print("=" * 70)
    print("🧪 MUSHROOM CLASSIFIER API - TEST SUITE")
    print("=" * 70)
    print(f"Testing endpoint: {base_url}")

    results = []

    # Test 1: Health check
    results.append(("Health Check", test_health_check(base_url)))

    # Test 2: Single prediction (edible)
    results.append(("Single Prediction", test_prediction(base_url, SAMPLE_MUSHROOM)))

    # Test 3: Single prediction (poisonous)
    results.append(("Poisonous Prediction", test_prediction(base_url, POISONOUS_MUSHROOM)))

    # Test 4: Batch prediction
    results.append(("Batch Prediction", test_batch_prediction(base_url, [SAMPLE_MUSHROOM, POISONOUS_MUSHROOM])))

    # Test 5: Documentation
    results.append(("Documentation", test_documentation(base_url)))

    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<50} {status}")

    print("=" * 70)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    import sys

    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    # Run tests
    success = run_full_test_suite(base_url)

    # Exit with appropriate code
    sys.exit(0 if success else 1)
