#!/usr/bin/env python3
"""
Quick Demo - Test kgirl's topological consensus instantly!
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo_consensus import test_consensus, get_demo_model_pool, get_divergent_pool

print("\n" + "=" * 70)
print("🚀 KGIRL QUICK DEMO - Topological Consensus Without API Keys")
print("=" * 70)

print("\n\n" + "🧪 TEST 1: High Coherence (3 similar models)")
print("Should get AUTO response with high agreement\n")
test_consensus(
    prompt="Explain quantum entanglement for a software engineer",
    model_pool=get_demo_model_pool(),
    min_coherence=0.80,
)

input("\n\nPress Enter for Test 2...")

print("\n\n" + "🧪 TEST 2: Low Coherence (divergent model)")
print("Should ESCALATE due to model disagreement\n")
test_consensus(
    prompt="What are the implications of artificial intelligence?",
    model_pool=get_divergent_pool(),
    min_coherence=0.80,
)

print("\n\n" + "=" * 70)
print("✅ DEMO COMPLETE!")
print("=" * 70)
print("\nYou just saw kgirl's topological consensus in action!")
print("\nKey features demonstrated:")
print("  • Phase coherence calculation (spectral weights on semantic space)")
print("  • Cardy energy for hallucination detection")
print("  • Decision logic (auto/needs_citations/escalate)")
print("\nWith real API keys, you'd get actual LLM responses!")
print("\nTry: python demo_consensus.py for interactive mode")
print("=" * 70 + "\n")
