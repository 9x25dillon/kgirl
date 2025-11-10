#!/usr/bin/env python3
"""
ChaosRAG API Client

This script demonstrates how to use the ChaosRAG Julia service endpoints
for indexing documents, pushing telemetry, and querying with chaos-aware routing.

Usage:
    python chaos_rag_client.py index document.txt
    python chaos_rag_client.py query "What is quantum computing?"
    python chaos_rag_client.py telemetry --asset BTC --vol 0.45 --entropy 0.62
    python chaos_rag_client.py --help
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import requests


class ChaosRAGClient:
    """Client for the ChaosRAG Julia API."""

    def __init__(self, base_url: Optional[str] = None):
        if base_url is None:
            port = os.getenv("CHAOS_RAG_PORT", "8001")
            self.base_url = f"http://localhost:{port}"
        else:
            self.base_url = base_url.rstrip("/")

    def index_documents(self, docs: list) -> dict:
        """Index documents into the vector database."""
        endpoint = f"{self.base_url}/chaos/rag/index"
        payload = {"docs": docs}

        try:
            response = requests.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error indexing documents: {e}", file=sys.stderr)
            sys.exit(1)

    def push_telemetry(
        self,
        asset: str,
        realized_vol: float,
        entropy: float,
        mod_intensity_grad: float = 0.0,
        router_noise: float = 0.0,
    ) -> dict:
        """Push asset telemetry for chaos routing."""
        endpoint = f"{self.base_url}/chaos/telemetry"
        payload = {
            "asset": asset,
            "realized_vol": realized_vol,
            "entropy": entropy,
            "mod_intensity_grad": mod_intensity_grad,
            "router_noise": router_noise,
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error pushing telemetry: {e}", file=sys.stderr)
            sys.exit(1)

    def query(self, query_text: str, k: int = 12) -> dict:
        """Query the knowledge base with chaos-aware routing."""
        endpoint = f"{self.base_url}/chaos/rag/query"
        payload = {"q": query_text, "k": k}

        try:
            response = requests.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying: {e}", file=sys.stderr)
            sys.exit(1)


def load_document_from_file(file_path: str) -> dict:
    """Load a document from a file."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "source": str(path),
            "kind": path.suffix.lstrip(".") or "text",
            "content": content,
            "meta": {"filename": path.name, "size": len(content)},
        }
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)


def print_index_result(result: dict) -> None:
    """Print indexing result."""
    print("\n" + "=" * 70)
    print("INDEXING RESULT")
    print("=" * 70)
    print(f"\nDocuments indexed: {result['inserted']}")
    print("\n✓ Success")


def print_telemetry_result(result: dict, asset: str) -> None:
    """Print telemetry push result."""
    print("\n" + "=" * 70)
    print("TELEMETRY UPDATE")
    print("=" * 70)
    print(f"\nAsset: {asset}")
    print(f"Status: {'OK' if result.get('ok') else 'Failed'}")
    print("\n✓ Telemetry pushed successfully")


def print_query_result(result: dict, query: str) -> None:
    """Print query result."""
    print("\n" + "=" * 70)
    print("CHAOS-AWARE RAG QUERY RESULT")
    print("=" * 70)

    print(f"\nQuery: {query}")

    # Router information
    router = result.get("router", {})
    print("\nRouter Decision:")
    print(f"  Stress Level: {router.get('stress', 0):.2%}")
    print(f"  Top-K: {router.get('top_k', 0)}")

    mix = router.get("mix", {})
    print("\nRetrieval Mix:")
    print(f"  Vector Search:  {mix.get('vector', 0):.1%}")
    print(f"  Graph Traversal: {mix.get('graph', 0):.1%}")
    print(f"  HHT Analysis:    {mix.get('hht', 0):.1%}")

    # Answer
    print("\n" + "-" * 70)
    print("ANSWER:")
    print("-" * 70)
    print(result.get("answer", "(No answer)"))
    print("-" * 70)

    # Hits (context)
    hits = result.get("hits", [])
    if hits:
        print(f"\nContext Hits: {len(hits)}")
        print("\nTop 3 Retrieved Contexts:")
        for i, hit in enumerate(hits[:3], 1):
            preview = hit[:100] + "..." if len(hit) > 100 else hit
            print(f"  [{i}] {preview}")


def main():
    parser = argparse.ArgumentParser(
        description="ChaosRAG API Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  index       Index documents into the vector database
  query       Query the knowledge base with chaos routing
  telemetry   Push asset telemetry data

Examples:
  # Index a document
  python chaos_rag_client.py index document.txt

  # Index with custom metadata
  python chaos_rag_client.py index paper.pdf --source "research paper" --kind research

  # Query the knowledge base
  python chaos_rag_client.py query "What is quantum entanglement?" --k 8

  # Push telemetry (affects routing decisions)
  python chaos_rag_client.py telemetry --asset BTC --vol 0.45 --entropy 0.62

  # Custom API URL
  python chaos_rag_client.py query "AI" --url http://api.example.com:8001
        """,
    )

    parser.add_argument(
        "--url",
        help="Base URL of the ChaosRAG API (default: http://localhost:$CHAOS_RAG_PORT)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("file", help="Document file to index")
    index_parser.add_argument("--source", help="Source identifier (default: filename)")
    index_parser.add_argument("--kind", help="Document kind (default: file extension)")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("query", help="Query string")
    query_parser.add_argument(
        "--k", type=int, default=12, help="Number of results (default: 12)"
    )

    # Telemetry command
    telemetry_parser = subparsers.add_parser("telemetry", help="Push telemetry data")
    telemetry_parser.add_argument(
        "--asset", required=True, help="Asset identifier (e.g., BTC, ETH)"
    )
    telemetry_parser.add_argument(
        "--vol", type=float, required=True, help="Realized volatility"
    )
    telemetry_parser.add_argument(
        "--entropy", type=float, required=True, help="System entropy"
    )
    telemetry_parser.add_argument(
        "--grad", type=float, default=0.0, help="Modular intensity gradient"
    )
    telemetry_parser.add_argument(
        "--noise", type=float, default=0.0, help="Router noise"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Create client
    client = ChaosRAGClient(base_url=args.url)

    # Execute command
    if args.command == "index":
        doc = load_document_from_file(args.file)

        # Override source/kind if provided
        if args.source:
            doc["source"] = args.source
        if args.kind:
            doc["kind"] = args.kind

        result = client.index_documents([doc])
        print_index_result(result)

    elif args.command == "query":
        result = client.query(args.query, k=args.k)
        print_query_result(result, args.query)

    elif args.command == "telemetry":
        result = client.push_telemetry(
            asset=args.asset,
            realized_vol=args.vol,
            entropy=args.entropy,
            mod_intensity_grad=args.grad,
            router_noise=args.noise,
        )
        print_telemetry_result(result, args.asset)


if __name__ == "__main__":
    main()
