from __future__ import annotations
from typing import Dict, List

# Extremely simple in-memory store keyed by namespace
_STORE: Dict[str, List[str]] = {}

async def ingest_texts(docs: List[str], namespace: str = "default") -> int:
    bucket = _STORE.setdefault(namespace, [])
    bucket.extend(docs or [])
    return len(bucket)

async def search(query: str, namespace: str = "default", top_k: int = 5) -> List[str]:
    # Naive substring rank by count occurrences
    corpus = _STORE.get(namespace, [])
    scored = [(doc, doc.lower().count((query or "").lower())) for doc in corpus]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, s in scored[:top_k] if s > 0]
