"""
Demo adapter for testing kgirl without API keys.
Provides deterministic mock responses to test topological consensus.
"""
import hashlib

import numpy as np
from typing import List


class DemoAdapter:
    """Mock adapter that generates deterministic responses for testing."""

    def __init__(self, name: str, response_style: str = "factual"):
        self.name = f"demo:{name}"
        self.response_style = response_style

    def generate(self, prompt: str) -> str:
        """Generate a deterministic mock response."""
        # Create different response styles to test coherence
        if self.response_style == "factual":
            return (
                f"Based on the query '{prompt[:50]}...', here's a factual response: "
                f"This is a comprehensive answer that addresses the key concepts with "
                f"scientific accuracy and proper context. The fundamental principles "
                f"involve systematic analysis and evidence-based reasoning."
            )
        elif self.response_style == "concise":
            return (
                f"Regarding '{prompt[:30]}...': This involves fundamental principles "
                f"that can be understood through systematic analysis."
            )
        elif self.response_style == "detailed":
            return (
                f"In response to '{prompt[:40]}...', let me provide a detailed explanation. "
                f"The core concepts require comprehensive understanding of the underlying "
                f"principles. Through systematic analysis and evidence-based reasoning, "
                f"we can derive accurate conclusions about this topic."
            )
        else:  # divergent
            return (
                f"Thinking about '{prompt[:30]}...': This is actually quite different "
                f"from conventional understanding. Alternative perspectives suggest "
                f"that we should reconsider the traditional approach entirely."
            )

    def embed(self, text: str, dim: int = 1536) -> np.ndarray:
        """Content embedding: signed feature hashing of character trigrams and words, L2-normalised.

        Stable across processes (blake2b, not the salted built-in hash()), and similar texts get
        similar vectors, so the consensus demo measures real content overlap between responses.
        """
        t = " ".join(text.lower().split())
        feats = [t[i:i + 3] for i in range(max(0, len(t) - 2))] + [f"w:{w}" for w in t.split()]
        emb = np.zeros(dim)
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            emb[h % dim] += 1.0 if (h >> 63) & 1 else -1.0
        return emb / (np.linalg.norm(emb) + 1e-12)

    def get_hidden_states(self, prompt: str) -> np.ndarray:
        """Return embedding as proxy for hidden states."""
        return self.embed(prompt)


def get_demo_model_pool() -> List[DemoAdapter]:
    """Create a pool of demo models with different response styles."""
    return [
        DemoAdapter("factual-model", "factual"),
        DemoAdapter("concise-model", "concise"),
        DemoAdapter("detailed-model", "detailed"),
    ]


def get_divergent_pool() -> List[DemoAdapter]:
    """Create a pool with one divergent model to test low coherence."""
    return [
        DemoAdapter("factual-model", "factual"),
        DemoAdapter("divergent-model", "divergent"),
    ]
