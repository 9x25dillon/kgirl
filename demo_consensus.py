#!/usr/bin/env python3
"""
Interactive Demo of kgirl's Topological Consensus
Tests coherence and energy metrics without requiring API keys.
"""
import os
import sys
from typing import List, Optional

import numpy as np

# Import from main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_adapter import get_demo_model_pool, get_divergent_pool, DemoAdapter

# Import consensus logic from main.py
from main import PhaseCoherence, CardyEnergy, spectral_weights


def print_separator(char="=", length=70):
    print(char * length)


def test_consensus(
    prompt: str,
    model_pool: List[DemoAdapter],
    min_coherence: float = 0.80,
    max_energy: float = 0.30,
) -> dict:
    """Run topological consensus on a prompt."""
    print_separator()
    print(f"PROMPT: {prompt}")
    print_separator()

    # 1. Generate outputs from all models
    print("\n[1/5] Generating responses from models...")
    outputs = []
    for model in model_pool:
        output = model.generate(prompt)
        outputs.append(output)
        print(f"  • {model.name}: {output[:80]}...")

    # 2. Embed outputs
    print("\n[2/5] Embedding responses...")
    embs = np.stack([model.embed(output) for model, output in zip(model_pool, outputs)], axis=0)
    print(f"  • Embedding shape: {embs.shape}")

    # 3. Calculate phase coherence
    print("\n[3/5] Calculating topological coherence...")
    pc = PhaseCoherence(n_anyons=max(len(model_pool), 5), central_charge=627)
    weights = pc.weights(embs)
    coherence = pc.scalar(embs)

    print(f"  • Model weights: {[f'{w:.3f}' for w in weights]}")
    print(f"  • Phase coherence: {coherence:.4f} (0=disagree, 1=perfect agreement)")

    # 4. Calculate Cardy energy (hallucination risk)
    print("\n[4/5] Calculating Cardy boundary energy (hallucination detection)...")
    ce = CardyEnergy(n_anyons=27, central_charge=627)
    energy = ce.boundary(model_pool[0].get_hidden_states(prompt))
    print(f"  • Energy: {energy:.4f} (0=confident, 1=uncertain)")

    # 5. Decision logic
    print("\n[5/5] Applying decision logic...")
    print(f"  • Coherence threshold: {min_coherence:.2f}")
    print(f"  • Energy threshold: {max_energy:.2f}")

    best_idx = int(np.argmax(weights))
    answer = None
    decision = None

    if coherence >= min_coherence and energy <= max_energy:
        decision = "auto"
        answer = outputs[best_idx]
        print(f"  ✓ AUTO-RESPOND: High coherence + Low energy")
    elif coherence >= 0.5 and energy <= 0.5:
        decision = "needs_citations"
        answer = outputs[best_idx]
        print(f"  ⚠ NEEDS CITATIONS: Medium coherence/energy")
    else:
        decision = "escalate"
        answer = None
        print(f"  ✗ ESCALATE: Low coherence OR high energy - human review needed")

    print_separator()
    print("CONSENSUS RESULT")
    print_separator()
    print(f"Decision: {decision.upper()}")
    print(f"Coherence: {coherence:.2%}")
    print(f"Energy: {energy:.2%}")
    print(f"Best model: {model_pool[best_idx].name} (weight: {weights[best_idx]:.3f})")

    if answer:
        print("\n" + "-" * 70)
        print("ANSWER:")
        print("-" * 70)
        print(answer)
        print("-" * 70)
    else:
        print("\n⚠️  No answer - requires human review")

    return {
        "decision": decision,
        "coherence": coherence,
        "energy": energy,
        "weights": weights.tolist(),
        "answer": answer,
        "model_names": [m.name for m in model_pool],
    }


def interactive_demo():
    """Run interactive demo."""
    print("\n" + "=" * 70)
    print("KGIRL TOPOLOGICAL CONSENSUS - INTERACTIVE DEMO")
    print("=" * 70)
    print("\nThis demo uses mock models to demonstrate coherence calculation")
    print("without requiring API keys.\n")

    while True:
        print("\n" + "=" * 70)
        print("Choose a test scenario:")
        print("=" * 70)
        print("1. High coherence test (similar models)")
        print("2. Low coherence test (divergent model)")
        print("3. Custom prompt with high coherence models")
        print("4. Custom prompt with divergent models")
        print("5. Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            print("\n🧪 Testing with COHERENT models (should get AUTO decision)...\n")
            test_consensus(
                prompt="Explain quantum entanglement in simple terms",
                model_pool=get_demo_model_pool(),
                min_coherence=0.80,
                max_energy=0.30,
            )

        elif choice == "2":
            print("\n🧪 Testing with DIVERGENT models (should get ESCALATE decision)...\n")
            test_consensus(
                prompt="What are the implications of AI?",
                model_pool=get_divergent_pool(),
                min_coherence=0.80,
                max_energy=0.30,
            )

        elif choice == "3":
            prompt = input("\nEnter your question: ").strip()
            if prompt:
                print(f"\n🧪 Testing with COHERENT models...\n")
                test_consensus(
                    prompt=prompt,
                    model_pool=get_demo_model_pool(),
                )

        elif choice == "4":
            prompt = input("\nEnter your question: ").strip()
            if prompt:
                print(f"\n🧪 Testing with DIVERGENT models...\n")
                test_consensus(
                    prompt=prompt,
                    model_pool=get_divergent_pool(),
                )

        elif choice == "5":
            print("\nThanks for testing kgirl! 🚀")
            break

        else:
            print("\n❌ Invalid choice. Please enter 1-5.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        interactive_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!")
        sys.exit(0)
