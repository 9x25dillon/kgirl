# Minimal stub; later wire FAISS and embeddings
from typing import List

def semantic_scores(query: str, texts: List[str]) -> list[float]:
    # TODO: real embeddings; return dummy equal scores for now
    return [1.0 for _ in texts]