#!/usr/bin/env python3
"""
Topological Consensus API Client - /rerank Endpoint

This script demonstrates how to use the /rerank endpoint to rerank documents
using spectral coherence weights and query similarity.

Usage:
    python rerank_client.py "query" doc1.txt doc2.txt doc3.txt
    python rerank_client.py --help
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import requests


def rerank_documents(
    query: str,
    docs: List[dict],
    trinary_threshold: float = 0.25,
    alpha: float = 0.7,
    beta: float = 0.3,
    base_url: Optional[str] = None,
) -> dict:
    """
    Rerank documents using the Topological Consensus API.

    Args:
        query: The search query
        docs: List of document dictionaries with 'id' and 'text' fields
        trinary_threshold: Threshold for trinary quantization
        alpha: Weight for query-document similarity
        beta: Weight for document coherence
        base_url: Base URL of the API

    Returns:
        Response dictionary with ranked_ids and scores
    """
    if base_url is None:
        port = os.getenv("MAIN_API_PORT", "8000")
        base_url = f"http://localhost:{port}"

    endpoint = f"{base_url}/rerank"

    payload = {
        "query": query,
        "docs": docs,
        "trinary_threshold": trinary_threshold,
        "alpha": alpha,
        "beta": beta,
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}", file=sys.stderr)
        sys.exit(1)


def load_documents_from_files(file_paths: List[str]) -> List[dict]:
    """Load documents from file paths."""
    docs = []
    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: File not found: {path}", file=sys.stderr)
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append({"id": path.name, "text": text})
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}", file=sys.stderr)

    return docs


def create_sample_documents() -> List[dict]:
    """Create sample documents for demonstration."""
    return [
        {
            "id": "doc1",
            "text": "Quantum computing uses qubits that can exist in superposition, "
            "allowing them to represent both 0 and 1 simultaneously. This enables "
            "quantum computers to perform certain calculations exponentially faster "
            "than classical computers.",
        },
        {
            "id": "doc2",
            "text": "Classical computing relies on bits that are either 0 or 1. "
            "These bits are the fundamental unit of information in traditional "
            "computers, and they process data sequentially using Boolean logic.",
        },
        {
            "id": "doc3",
            "text": "Quantum entanglement is a phenomenon where two or more particles "
            "become correlated in such a way that the state of one particle "
            "instantaneously affects the state of the other, regardless of distance.",
        },
        {
            "id": "doc4",
            "text": "Machine learning algorithms can be categorized into supervised, "
            "unsupervised, and reinforcement learning. Each approach has different "
            "applications and requires different types of training data.",
        },
        {
            "id": "doc5",
            "text": "The uncertainty principle in quantum mechanics states that you "
            "cannot simultaneously know both the position and momentum of a particle "
            "with arbitrary precision. This is a fundamental limit of measurement.",
        },
    ]


def print_result(result: dict, docs: List[dict], query: str) -> None:
    """Pretty print the reranking results."""
    print("\n" + "=" * 70)
    print("DOCUMENT RERANKING RESULT")
    print("=" * 70)

    print(f"\nQuery: {query}")
    print(f"Total documents: {len(docs)}")

    # Create a mapping of doc_id to text
    doc_map = {d["id"]: d["text"] for d in docs}

    print("\n" + "-" * 70)
    print("RANKED RESULTS:")
    print("-" * 70)

    for rank, (doc_id, score) in enumerate(
        zip(result["ranked_ids"], result["scores"]), 1
    ):
        print(f"\n[{rank}] {doc_id} (score: {score:.4f})")
        text = doc_map.get(doc_id, "Text not found")
        # Print first 150 chars
        preview = text[:150] + "..." if len(text) > 150 else text
        print(f"    {preview}")


def main():
    parser = argparse.ArgumentParser(
        description="Rerank documents using the Topological Consensus API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode with sample documents
  python rerank_client.py "quantum computing basics" --demo

  # Rerank text files
  python rerank_client.py "machine learning" doc1.txt doc2.txt doc3.txt

  # Adjust similarity vs coherence weights
  python rerank_client.py "AI research" *.txt --alpha 0.8 --beta 0.2

  # Custom API URL
  python rerank_client.py "query" file.txt --url http://api.example.com:8000
        """,
    )

    parser.add_argument("query", help="The search query")
    parser.add_argument(
        "files",
        nargs="*",
        help="Document files to rerank (use --demo for sample docs)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use sample documents instead of files",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for query-document similarity (default: 0.7)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Weight for document coherence (default: 0.3)",
    )
    parser.add_argument(
        "--trinary-threshold",
        type=float,
        default=0.25,
        help="Trinary quantization threshold (default: 0.25)",
    )
    parser.add_argument(
        "--url",
        help="Base URL of the API (default: http://localhost:$MAIN_API_PORT)",
    )

    args = parser.parse_args()

    # Load or create documents
    if args.demo:
        docs = create_sample_documents()
        print("Using sample documents for demonstration")
    elif args.files:
        docs = load_documents_from_files(args.files)
        if not docs:
            print("Error: No documents loaded", file=sys.stderr)
            sys.exit(1)
    else:
        print(
            "Error: Provide document files or use --demo flag", file=sys.stderr
        )
        parser.print_help()
        sys.exit(1)

    # Make the request
    result = rerank_documents(
        query=args.query,
        docs=docs,
        trinary_threshold=args.trinary_threshold,
        alpha=args.alpha,
        beta=args.beta,
        base_url=args.url,
    )

    # Print results
    print_result(result, docs, args.query)


if __name__ == "__main__":
    main()
