#!/usr/bin/env python3
"""Minimal stub implementations for external dependencies used by demo.

These provide async-compatible interfaces used by the RecursiveCognitiveKnowledge
demo so it can run inside this environment without installing external packages.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any, Dict, List, Tuple


class HybridConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class HybridEmbeddingPipeline:
    def __init__(self, config: HybridConfig = None):
        self.config = config or HybridConfig()

    async def embed(self, text: str) -> Dict[str, Any]:
        # return a deterministic pseudo-embedding for testing
        vec = [float((ord(c) % 32) / 32.0) for c in text[:64]]
        return {"embedding": vec, "metadata": {"components_used": ["fractal"] , "embedding_dim": len(vec)}}

    async def close(self):
        return None


class IndexEntry:
    def __init__(self, id: str, text: str, metadata: Dict = None):
        self.id = id
        self.text = text
        self.metadata = metadata or {}


class EnhancedVectorIndex:
    def __init__(self, use_numbskull: bool = False):
        self.store: List[Tuple[IndexEntry, float]] = []

    async def add_entry(self, id: str, text: str, metadata: Dict = None):
        entry = IndexEntry(id, text, metadata)
        # small random score placeholder
        self.store.append((entry, random.random()))

    async def search(self, query: str, top_k: int = 3) -> List[Tuple[IndexEntry, float]]:
        # naive textual similarity: count common words
        qwords = set(query.lower().split())
        scored = []
        for entry, _ in self.store:
            ew = set(entry.text.lower().split())
            sim = len(qwords & ew) / max(1, len(qwords | ew))
            scored.append((entry, float(sim)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def close(self):
        return None

    def get_stats(self) -> Dict[str, Any]:
        return {"entries": len(self.store)}


class EnhancedGraphStore:
    def __init__(self, use_numbskull: bool = False):
        self.nodes = {}

    async def add_node(self, node_id: str, node_type: str, payload: Dict = None):
        self.nodes[node_id] = {"type": node_type, "payload": payload or {}}

    async def close(self):
        return None

    def get_stats(self) -> Dict[str, Any]:
        return {"nodes": len(self.nodes)}

