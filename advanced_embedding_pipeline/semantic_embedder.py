#!/usr/bin/env python3
"""
Semantic Embedder - Advanced semantic vectorization
Integrates Eopiez vectorization with enhanced semantic processing
"""

import asyncio
import logging
import numpy as np
import httpx
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SemanticConfig:
    """Configuration for semantic embedding"""
    eopiez_url: str = "http://localhost:8001"
    embedding_dim: int = 768
    batch_size: int = 32
    max_retries: int = 3
    timeout: float = 30.0
    use_cache: bool = True
    cache_dir: str = "./cache/embeddings"


class SemanticEmbedder:
    """
    Advanced semantic embedder that integrates with Eopiez and provides
    enhanced semantic processing capabilities.
    """
    
    def __init__(self, config: Optional[SemanticConfig] = None):
        self.config = config or SemanticConfig()
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
        self.cache = {}
        self._setup_cache()
        
    def _setup_cache(self):
        """Setup embedding cache directory"""
        if self.config.use_cache:
            cache_path = Path(self.config.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            self.cache_dir = cache_path
            
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from cache"""
        if not self.config.use_cache:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to cache"""
        if not self.config.use_cache:
            return
            
        cache_file = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text using Eopiez vectorization
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        cache_key = self._get_cache_key(text)
        
        # Check cache first
        cached_embedding = self._load_from_cache(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate embedding via Eopiez
        try:
            response = await self.client.post(
                f"{self.config.eopiez_url}/qvnm/upload_vectors",
                json={"text": text}
            )
            response.raise_for_status()
            
            result = response.json()
            if "embedding" in result:
                embedding = np.array(result["embedding"], dtype=np.float32)
            else:
                # Fallback: generate simple embedding
                embedding = self._generate_fallback_embedding(text)
            
            # Save to cache
            self._save_to_cache(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Eopiez embedding failed for text: {e}")
            # Fallback to local embedding
            embedding = self._generate_fallback_embedding(text)
            return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed a batch of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            # Check cache for batch
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch):
                cache_key = self._get_cache_key(text)
                cached_embedding = self._load_from_cache(cache_key)
                
                if cached_embedding is not None:
                    batch_embeddings.append(cached_embedding)
                else:
                    batch_embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(j)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    # Try batch processing with Eopiez
                    response = await self.client.post(
                        f"{self.config.eopiez_url}/qvnm/build_codes",
                        json={"texts": uncached_texts}
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    if "embeddings" in result:
                        new_embeddings = result["embeddings"]
                        for idx, embedding_data in zip(uncached_indices, new_embeddings):
                            embedding = np.array(embedding_data, dtype=np.float32)
                            batch_embeddings[idx] = embedding
                            # Cache the embedding
                            cache_key = self._get_cache_key(batch[idx])
                            self._save_to_cache(cache_key, embedding)
                    else:
                        # Fallback for each uncached text
                        for idx in uncached_indices:
                            embedding = self._generate_fallback_embedding(batch[idx])
                            batch_embeddings[idx] = embedding
                            
                except Exception as e:
                    logger.warning(f"Batch embedding failed: {e}")
                    # Fallback for each uncached text
                    for idx in uncached_indices:
                        embedding = self._generate_fallback_embedding(batch[idx])
                        batch_embeddings[idx] = embedding
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """
        Generate fallback embedding when Eopiez is unavailable
        
        Args:
            text: Input text
            
        Returns:
            Fallback embedding vector
        """
        # Simple hash-based embedding
        text_bytes = text.encode('utf-8')
        hash_val = hashlib.sha256(text_bytes).digest()
        
        # Convert hash to embedding vector
        embedding = np.zeros(self.config.embedding_dim, dtype=np.float32)
        
        for i in range(0, min(len(hash_val), self.config.embedding_dim // 4)):
            # Use hash bytes to seed random-like values
            seed = int.from_bytes(hash_val[i:i+4], 'big')
            np.random.seed(seed)
            
            # Generate 4 float values from this seed
            for j in range(4):
                if i * 4 + j < self.config.embedding_dim:
                    embedding[i * 4 + j] = np.random.normal(0, 0.1)
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    async def query_similar(self, query_text: str, candidate_embeddings: List[np.ndarray], 
                          top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to query
        
        Args:
            query_text: Query text
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if not candidate_embeddings:
            return []
        
        # Generate query embedding
        query_embedding = await self.embed_text(query_text)
        
        # Calculate similarities
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            if candidate is not None:
                similarity = np.dot(query_embedding, candidate)
                similarities.append((i, float(similarity)))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    async def extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """
        Extract semantic features from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of semantic features
        """
        features = {
            "length": len(text),
            "word_count": len(text.split()),
            "char_count": len(text),
            "unique_chars": len(set(text)),
            "embedding": await self.embed_text(text)
        }
        
        # Extract additional features
        features["has_numbers"] = any(c.isdigit() for c in text)
        features["has_special"] = any(not c.isalnum() and not c.isspace() for c in text)
        features["avg_word_length"] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        return features
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'client'):
            asyncio.create_task(self.close())
