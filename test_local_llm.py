#!/usr/bin/env python3
"""
Test script for local LLM integration with Ollama
Run this to verify your local setup is working correctly
"""

import sys
import requests
import time
from typing import Dict, Any


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_ollama_running() -> bool:
    """Test if Ollama service is running"""
    print_section("Testing Ollama Service")

    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            print(f"   Version: {response.json()}")
            return True
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama on http://localhost:11434")
        print("\n💡 Solution:")
        print("   1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   2. Start Ollama: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_models_available() -> bool:
    """Test if required models are available"""
    print_section("Checking Ollama Models")

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]

        print(f"Found {len(models)} model(s):")
        for model in models:
            print(f"  • {model['name']} ({model.get('size', 'unknown')} bytes)")

        # Check for required models
        required = ["qwen2.5:3b", "nomic-embed-text"]
        missing = []

        for req in required:
            # Check for exact match or base name match
            found = any(req in name for name in model_names)
            if found:
                print(f"✅ Found {req}")
            else:
                print(f"❌ Missing {req}")
                missing.append(req)

        if missing:
            print("\n💡 Solution: Pull missing models:")
            for m in missing:
                print(f"   ollama pull {m}")
            return False

        return True
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False


def test_kgirl_api() -> bool:
    """Test if kgirl API is running"""
    print_section("Testing kgirl API")

    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ kgirl API is running")
            print(f"   Models: {data.get('models', [])}")
            print(f"   CTH available: {data.get('cth', False)}")
            return True
        else:
            print(f"❌ kgirl API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to kgirl API on http://localhost:8000")
        print("\n💡 Solution: Start kgirl API:")
        print("   python main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_ask_endpoint() -> bool:
    """Test the /ask endpoint with local LLM"""
    print_section("Testing /ask Endpoint (Local LLM)")

    try:
        print("Sending query: 'What is 2+2?'")
        start_time = time.time()

        response = requests.post(
            "http://localhost:8000/ask",
            json={
                "prompt": "What is 2+2? Answer in one short sentence.",
                "min_coherence": 0.5,
                "max_energy": 0.5,
                "return_all": True
            },
            timeout=30
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query successful ({elapsed:.2f}s)")
            print(f"\n📝 Answer: {data.get('answer', 'N/A')}")
            print(f"   Decision: {data.get('decision')}")
            print(f"   Coherence: {data.get('coherence', 0):.3f}")
            print(f"   Energy: {data.get('energy', 0):.3f}")
            print(f"   Model: {data.get('model_names', [])}")

            if data.get('all_outputs'):
                print(f"\n   All outputs:")
                for i, output in enumerate(data['all_outputs'], 1):
                    print(f"     {i}. {output[:100]}...")

            return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_rerank_endpoint() -> bool:
    """Test the /rerank endpoint"""
    print_section("Testing /rerank Endpoint")

    try:
        print("Testing document reranking...")

        response = requests.post(
            "http://localhost:8000/rerank",
            json={
                "query": "quantum computing",
                "docs": [
                    {"id": "doc1", "text": "Quantum computers use qubits for computation"},
                    {"id": "doc2", "text": "Classical computers use bits"},
                    {"id": "doc3", "text": "Python is a programming language"}
                ]
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            print("✅ Reranking successful")
            print(f"   Ranked order: {data.get('ranked_ids', [])}")
            print(f"   Scores: {[f'{s:.3f}' for s in data.get('scores', [])]}")
            return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  🧪 kgirl Local LLM Test Suite")
    print("="*60)

    tests = [
        ("Ollama Service", test_ollama_running),
        ("Ollama Models", test_models_available),
        ("kgirl API", test_kgirl_api),
        ("Ask Endpoint", test_ask_endpoint),
        ("Rerank Endpoint", test_rerank_endpoint),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print("\n\n⚠ Tests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results[name] = False

    # Summary
    print_section("Test Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed")
    print('='*60)

    if passed == total:
        print("\n🎉 All tests passed! Your local LLM setup is working perfectly!")
        print("\nNext steps:")
        print("  • Try the full system: ./START_NOW.sh")
        print("  • Explore the API: http://localhost:8000/docs")
        print("  • Read the guide: cat LOCAL_LLM_SETUP.md")
        return 0
    else:
        print("\n⚠ Some tests failed. Please fix the issues above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
