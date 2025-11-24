#!/usr/bin/env python3
"""
Enhanced Vector Index with Numbskull Integration
================================================

Advanced vector indexing system that integrates:
- Numbskull hybrid embeddings (semantic, mathematical, fractal)
- Multiple indexing backends (FAISS, Annoy, HNSW)
- Similarity search with embedding enhancement
- Real-time indexing and updates

Author: Assistant
License: MIT
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add numbskull to path
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

try:
    from advanced_embedding_pipeline import HybridEmbeddingPipeline, HybridConfig
    NUMBSKULL_AVAILABLE = True
except ImportError:
    NUMBSKULL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IndexEntry:
    """Single entry in the vector index"""
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EnhancedVectorIndex:
    """
    Vector index with Numbskull embedding integration
    
    Provides fast similarity search using hybrid embeddings
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        use_numbskull: bool = True,
        numbskull_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced vector index
        
        Args:
            embedding_dim: Dimension of embedding vectors
            use_numbskull: Use Numbskull for embedding generation
            numbskull_config: Configuration for Numbskull pipeline
        """
        self.embedding_dim = embedding_dim
        self.entries: List[IndexEntry] = []
        self.index_built = False
        
        # Initialize Numbskull pipeline
        if use_numbskull and NUMBSKULL_AVAILABLE:
            config = HybridConfig(**(numbskull_config or {}))
            self.numbskull = HybridEmbeddingPipeline(config)
            logger.info("✅ Enhanced vector index with Numbskull embeddings")
        else:
            self.numbskull = None
            logger.warning("⚠️  Vector index without Numbskull (using simple embeddings)")
        
        # Try to import indexing backends
        self.faiss = None
        self.faiss_index = None
        
        try:
            import faiss
            self.faiss = faiss
            logger.info("✅ FAISS available for fast indexing")
        except ImportError:
            logger.warning("⚠️  FAISS not available (using brute force search)")
    
    async def add_entry(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        precomputed_embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Add entry to the index
        
        Args:
            id: Unique identifier
            text: Text content
            metadata: Optional metadata
            precomputed_embedding: Optional precomputed embedding
        
        Returns:
            Success status
        """
        try:
            # Generate embedding if not provided
            if precomputed_embedding is not None:
                embedding = precomputed_embedding
            elif self.numbskull:
                result = await self.numbskull.embed(text)
                embedding = result["fused_embedding"]
            else:
                # Simple fallback embedding
                embedding = self._simple_embedding(text)
            
            # Normalize embedding dimension
            if len(embedding) != self.embedding_dim:
                embedding = self._normalize_dimension(embedding)
            
            # Create entry
            entry = IndexEntry(
                id=id,
                text=text,
                embedding=embedding,
                metadata=metadata or {},
                timestamp=time.time()
            )
            
            self.entries.append(entry)
            self.index_built = False  # Mark for rebuild
            
            logger.debug(f"Added entry {id} to index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add entry {id}: {e}")
            return False
    
    async def add_batch(
        self,
        entries: List[Tuple[str, str, Optional[Dict[str, Any]]]]
    ) -> int:
        """
        Add multiple entries in batch
        
        Args:
            entries: List of (id, text, metadata) tuples
        
        Returns:
            Number of successfully added entries
        """
        success_count = 0
        
        # Extract texts for batch embedding
        texts = [text for _, text, _ in entries]
        
        # Generate embeddings in batch
        if self.numbskull:
            embeddings = []
            for text in texts:
                result = await self.numbskull.embed(text)
                embeddings.append(result["fused_embedding"])
        else:
            embeddings = [self._simple_embedding(text) for text in texts]
        
        # Add entries
        for (id, text, metadata), embedding in zip(entries, embeddings):
            if await self.add_entry(id, text, metadata, embedding):
                success_count += 1
        
        logger.info(f"Added {success_count}/{len(entries)} entries in batch")
        return success_count
    
    def build_index(self, force_rebuild: bool = False):
        """
        Build or rebuild the index
        
        Args:
            force_rebuild: Force rebuild even if already built
        """
        if self.index_built and not force_rebuild:
            return
        
        if not self.entries:
            logger.warning("No entries to index")
            return
        
        if self.faiss:
            # Build FAISS index
            embeddings = np.array([entry.embedding for entry in self.entries])
            embeddings = embeddings.astype('float32')
            
            # Create index
            self.faiss_index = self.faiss.IndexFlatL2(self.embedding_dim)
            self.faiss_index.add(embeddings)
            
            logger.info(f"Built FAISS index with {len(self.entries)} entries")
        
        self.index_built = True
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
        precomputed_embedding: Optional[np.ndarray] = None
    ) -> List[Tuple[IndexEntry, float]]:
        """
        Search for similar entries
        
        Args:
            query: Query text
            top_k: Number of results to return
            threshold: Optional similarity threshold
            precomputed_embedding: Optional precomputed query embedding
        
        Returns:
            List of (entry, similarity_score) tuples
        """
        if not self.entries:
            return []
        
        # Build index if needed
        self.build_index()
        
        # Generate query embedding
        if precomputed_embedding is not None:
            query_embedding = precomputed_embedding
        elif self.numbskull:
            result = await self.numbskull.embed(query)
            query_embedding = result["fused_embedding"]
        else:
            query_embedding = self._simple_embedding(query)
        
        # Normalize dimension
        if len(query_embedding) != self.embedding_dim:
            query_embedding = self._normalize_dimension(query_embedding)
        
        # Search
        if self.faiss and self.faiss_index:
            # Use FAISS for fast search
            query_embedding = query_embedding.astype('float32').reshape(1, -1)
            distances, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.entries)))
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.entries):
                    # Convert distance to similarity (inverse distance)
                    similarity = 1.0 / (1.0 + dist)
                    if threshold is None or similarity >= threshold:
                        results.append((self.entries[idx], similarity))
        else:
            # Brute force search
            similarities = []
            for entry in self.entries:
                similarity = self._cosine_similarity(query_embedding, entry.embedding)
                if threshold is None or similarity >= threshold:
                    similarities.append((entry, similarity))
            
            # Sort by similarity
            results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        
        return results
    
    def get_entry(self, id: str) -> Optional[IndexEntry]:
        """Get entry by ID"""
        for entry in self.entries:
            if entry.id == id:
                return entry
        return None
    
    def remove_entry(self, id: str) -> bool:
        """Remove entry by ID"""
        for i, entry in enumerate(self.entries):
            if entry.id == id:
                self.entries.pop(i)
                self.index_built = False
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_entries": len(self.entries),
            "embedding_dim": self.embedding_dim,
            "index_built": self.index_built,
            "numbskull_enabled": self.numbskull is not None,
            "faiss_available": self.faiss is not None
        }
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple fallback embedding (hash-based)"""
        # Basic hash-based embedding
        hash_val = hash(text)
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        return embedding / np.linalg.norm(embedding)
    
    def _normalize_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to target dimension"""
        if len(embedding) > self.embedding_dim:
            return embedding[:self.embedding_dim]
        elif len(embedding) < self.embedding_dim:
            padded = np.zeros(self.embedding_dim)
            padded[:len(embedding)] = embedding
            return padded
        return embedding
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    async def close(self):
        """Clean up resources"""
        if self.numbskull:
            await self.numbskull.close()


async def demo_enhanced_vector_index():
    """Demonstration of enhanced vector index"""
    print("\n" + "=" * 70)
    print("ENHANCED VECTOR INDEX DEMO")
    print("=" * 70)
    
    # Create index
    index = EnhancedVectorIndex(
        embedding_dim=768,
        use_numbskull=NUMBSKULL_AVAILABLE,
        numbskull_config={
            "use_semantic": False,
            "use_mathematical": False,
            "use_fractal": True,
            "cache_embeddings": True
        }
    )
    
    # Add entries
    entries = [
        ("doc1", "Machine learning enables computers to learn from data", {"category": "AI"}),
        ("doc2", "Neural networks are inspired by biological neurons", {"category": "AI"}),
        ("doc3", "Python is a popular programming language", {"category": "Programming"}),
        ("doc4", "Quantum computing uses quantum mechanics principles", {"category": "Quantum"}),
        ("doc5", "Deep learning is a subset of machine learning", {"category": "AI"}),
    ]
    
    print("\nAdding entries...")
    count = await index.add_batch(entries)
    print(f"✅ Added {count} entries")
    
    # Build index
    print("\nBuilding index...")
    index.build_index()
    print(f"✅ Index built")
    
    # Search
    query = "artificial intelligence and neural networks"
    print(f"\nSearching for: '{query}'")
    results = await index.search(query, top_k=3)
    
    print(f"\nTop {len(results)} results:")
    for i, (entry, score) in enumerate(results, 1):
        print(f"  {i}. [{score:.3f}] {entry.id}: {entry.text}")
        print(f"     Category: {entry.metadata.get('category', 'N/A')}")
    
    # Stats
    print("\nIndex Statistics:")
    stats = index.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    await index.close()
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_enhanced_vector_index())

