#!/usr/bin/env python3
"""
Topological Consensus API Client - /ask Endpoint

This script demonstrates how to use the /ask endpoint to query multiple LLMs
with topological consensus and hallucination detection.

Usage:
    python ask_client.py "Your question here"
    python ask_client.py --help
"""

import argparse
import os
import sys
from typing import Optional

import requests


def ask_consensus(
    prompt: str,
    min_coherence: float = 0.80,
    max_energy: float = 0.30,
    return_all: bool = False,
    base_url: Optional[str] = None,
) -> dict:
    """
    Query the Topological Consensus API with a prompt.

    Args:
        prompt: The question or prompt to send
        min_coherence: Minimum coherence threshold (0-1)
        max_energy: Maximum energy threshold for hallucination detection (0-1)
        return_all: Whether to return all model outputs
        base_url: Base URL of the API (defaults to http://localhost:8000)

    Returns:
        Response dictionary with answer, decision, metrics, etc.
    """
    if base_url is None:
        port = os.getenv("MAIN_API_PORT", "8000")
        base_url = f"http://localhost:{port}"

    endpoint = f"{base_url}/ask"

    payload = {
        "prompt": prompt,
        "min_coherence": min_coherence,
        "max_energy": max_energy,
        "return_all": return_all,
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}", file=sys.stderr)
        sys.exit(1)


def print_result(result: dict, verbose: bool = False) -> None:
    """Pretty print the API response."""
    print("\n" + "=" * 70)
    print("TOPOLOGICAL CONSENSUS RESULT")
    print("=" * 70)

    print(f"\nDecision: {result['decision'].upper()}")
    print(f"Coherence: {result['coherence']:.2%} (model agreement)")
    print(f"Energy: {result['energy']:.2%} (hallucination risk)")

    print(f"\nModel Pool: {', '.join(result['model_names'])}")
    print(f"Weights: {[f'{w:.2f}' for w in result['weights']]}")

    if result["answer"]:
        print("\n" + "-" * 70)
        print("ANSWER:")
        print("-" * 70)
        print(result["answer"])
        print("-" * 70)
    else:
        print("\n⚠️  No answer provided - human review required")

    if verbose and result.get("all_outputs"):
        print("\n" + "=" * 70)
        print("ALL MODEL OUTPUTS:")
        print("=" * 70)
        for i, (name, output) in enumerate(
            zip(result["model_names"], result["all_outputs"]), 1
        ):
            print(f"\n[{i}] {name}:")
            print("-" * 70)
            print(output)
            print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Query the Topological Consensus API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query
  python ask_client.py "What is quantum entanglement?"

  # Strict thresholds
  python ask_client.py "Explain black holes" --min-coherence 0.9 --max-energy 0.2

  # Show all model outputs
  python ask_client.py "How does photosynthesis work?" --verbose

  # Custom API URL
  python ask_client.py "What is DNA?" --url http://api.example.com:8000
        """,
    )

    parser.add_argument("prompt", help="The question or prompt to send")
    parser.add_argument(
        "--min-coherence",
        type=float,
        default=0.80,
        help="Minimum coherence threshold (default: 0.80)",
    )
    parser.add_argument(
        "--max-energy",
        type=float,
        default=0.30,
        help="Maximum energy threshold (default: 0.30)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show all model outputs",
    )
    parser.add_argument(
        "--url",
        help="Base URL of the API (default: http://localhost:$MAIN_API_PORT)",
    )

    args = parser.parse_args()

    # Make the request
    result = ask_consensus(
        prompt=args.prompt,
        min_coherence=args.min_coherence,
        max_energy=args.max_energy,
        return_all=args.verbose,
        base_url=args.url,
    )

    # Print results
    print_result(result, verbose=args.verbose)

    # Exit with appropriate code based on decision
    decision_codes = {
        "auto": 0,
        "needs_citations": 1,
        "escalate": 2,
    }
    sys.exit(decision_codes.get(result["decision"], 0))


if __name__ == "__main__":
    main()
