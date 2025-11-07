#!/usr/bin/env python3
"""
Hybrid Embedding Pipeline - Unified orchestration of all embedding methods
Combines semantic, mathematical, and fractal embedding approaches
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .semantic_embedder import SemanticEmbedder, SemanticConfig
from .mathematical_embedder import MathematicalEmbedder, MathematicalConfig
from .fractal_cascade_embedder import FractalCascadeEmbedder, FractalConfig

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Configuration for hybrid embedding pipeline"""
    # Component configurations
    semantic_config: Optional[SemanticConfig] = None
    mathematical_config: Optional[MathematicalConfig] = None
    fractal_config: Optional[FractalConfig] = None
    
    # Pipeline settings
    use_semantic: bool = True
    use_mathematical: bool = True
    use_fractal: bool = True
    
    # Fusion settings
    fusion_method: str = "weighted_average"  # "weighted_average", "concatenation", "attention"
    semantic_weight: float = 0.4
    mathematical_weight: float = 0.3
    fractal_weight: float = 0.3
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    cache_embeddings: bool = True
    timeout: float = 60.0


class HybridEmbeddingPipeline:
    """
    Hybrid embedding pipeline that orchestrates multiple embedding methods
    and combines them into unified representations.
    """
    
    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()
        
        # Initialize component embedders
        self.semantic_embedder = None
        self.mathematical_embedder = None
        self.fractal_embedder = None
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.cache_lock = threading.Lock()
        
        # Performance metrics
        self.metrics = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "semantic_embeddings": 0,
            "mathematical_embeddings": 0,
            "fractal_embeddings": 0,
            "fusion_operations": 0,
            "average_time": 0.0
        }
        
        self._initialize_embedders()
    
    def _initialize_embedders(self):
        """Initialize component embedders based on configuration"""
        try:
            if self.config.use_semantic:
                self.semantic_embedder = SemanticEmbedder(self.config.semantic_config)
                logger.info("✅ Semantic embedder initialized")
            
            if self.config.use_mathematical:
                self.mathematical_embedder = MathematicalEmbedder(self.config.mathematical_config)
                logger.info("✅ Mathematical embedder initialized")
            
            if self.config.use_fractal:
                self.fractal_embedder = FractalCascadeEmbedder(self.config.fractal_config)
                logger.info("✅ Fractal embedder initialized")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedders: {e}")
    
    def _get_cache_key(self, text: str, config_hash: str = "") -> str:
        """Generate cache key for text and configuration"""
        import hashlib
        key_data = f"{text}_{config_hash}_{self.config.fusion_method}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_config_hash(self) -> str:
        """Generate hash of current configuration"""
        import hashlib
        config_data = {
            "semantic_weight": self.config.semantic_weight,
            "mathematical_weight": self.config.mathematical_weight,
            "fractal_weight": self.config.fractal_weight,
            "fusion_method": self.config.fusion_method
        }
        return hashlib.md5(json.dumps(config_data, sort_keys=True).encode()).hexdigest()
    
    async def embed(self, text: str) -> Dict[str, Any]:
        """
        Generate hybrid embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing embedding components and final result
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(text, self._get_config_hash())
            
            with self.cache_lock:
                if self.config.cache_embeddings and cache_key in self.embedding_cache:
                    self.metrics["cache_hits"] += 1
                    cached_result = self.embedding_cache[cache_key]
                    cached_result["cached"] = True
                    return cached_result
            
            # Generate embeddings from all components
            embeddings = {}
            
            if self.config.parallel_processing:
                # Parallel processing
                tasks = []
                
                if self.semantic_embedder and self.config.use_semantic:
                    tasks.append(("semantic", self._embed_semantic(text)))
                
                if self.mathematical_embedder and self.config.use_mathematical:
                    tasks.append(("mathematical", self._embed_mathematical(text)))
                
                if self.fractal_embedder and self.config.use_fractal:
                    tasks.append(("fractal", self._embed_fractal(text)))
                
                # Execute tasks in parallel
                results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
                
                for (component_name, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        logger.warning(f"Embedding failed for {component_name}: {result}")
                        embeddings[component_name] = None
                    else:
                        embeddings[component_name] = result
                        
            else:
                # Sequential processing
                if self.semantic_embedder and self.config.use_semantic:
                    embeddings["semantic"] = await self._embed_semantic(text)
                
                if self.mathematical_embedder and self.config.use_mathematical:
                    embeddings["mathematical"] = await self._embed_mathematical(text)
                
                if self.fractal_embedder and self.config.use_fractal:
                    embeddings["fractal"] = await self._embed_fractal(text)
            
            # Fuse embeddings
            fused_embedding = self._fuse_embeddings(embeddings)
            
            # Prepare result
            result = {
                "text": text,
                "embeddings": embeddings,
                "fused_embedding": fused_embedding,
                "metadata": {
                    "processing_time": time.time() - start_time,
                    "components_used": list(embeddings.keys()),
                    "fusion_method": self.config.fusion_method,
                    "embedding_dim": len(fused_embedding)
                },
                "cached": False
            }
            
            # Cache result
            with self.cache_lock:
                if self.config.cache_embeddings:
                    self.embedding_cache[cache_key] = result.copy()
            
            # Update metrics
            self._update_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Hybrid embedding failed: {e}")
            return {
                "text": text,
                "error": str(e),
                "embeddings": {},
                "fused_embedding": np.zeros(768, dtype=np.float32),
                "metadata": {"processing_time": time.time() - start_time}
            }
    
    async def _embed_semantic(self, text: str) -> np.ndarray:
        """Generate semantic embedding"""
        try:
            if self.semantic_embedder:
                embedding = await self.semantic_embedder.embed_text(text)
                self.metrics["semantic_embeddings"] += 1
                return embedding
            else:
                return np.zeros(768, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Semantic embedding failed: {e}")
            return np.zeros(768, dtype=np.float32)
    
    async def _embed_mathematical(self, text: str) -> np.ndarray:
        """Generate mathematical embedding"""
        try:
            if self.mathematical_embedder:
                embedding = await self.mathematical_embedder.embed_mathematical_expression(text)
                self.metrics["mathematical_embeddings"] += 1
                return embedding
            else:
                return np.zeros(1024, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Mathematical embedding failed: {e}")
            return np.zeros(1024, dtype=np.float32)
    
    async def _embed_fractal(self, text: str) -> np.ndarray:
        """Generate fractal embedding"""
        try:
            if self.fractal_embedder:
                embedding = self.fractal_embedder.embed_text_with_fractal(text)
                self.metrics["fractal_embeddings"] += 1
                return embedding
            else:
                return np.zeros(1024, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Fractal embedding failed: {e}")
            return np.zeros(1024, dtype=np.float32)
    
    def _fuse_embeddings(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse multiple embeddings into single representation"""
        try:
            # Filter out None embeddings
            valid_embeddings = {k: v for k, v in embeddings.items() if v is not None}
            
            if not valid_embeddings:
                return np.zeros(768, dtype=np.float32)
            
            if len(valid_embeddings) == 1:
                return list(valid_embeddings.values())[0]
            
            # Apply fusion method
            if self.config.fusion_method == "weighted_average":
                return self._weighted_average_fusion(valid_embeddings)
            elif self.config.fusion_method == "concatenation":
                return self._concatenation_fusion(valid_embeddings)
            elif self.config.fusion_method == "attention":
                return self._attention_fusion(valid_embeddings)
            else:
                return self._weighted_average_fusion(valid_embeddings)
                
        except Exception as e:
            logger.warning(f"Embedding fusion failed: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def _weighted_average_fusion(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average fusion of embeddings"""
        try:
            # Normalize embeddings to same dimension
            normalized_embeddings = {}
            target_dim = 768
            
            for name, embedding in embeddings.items():
                if len(embedding) != target_dim:
                    # Resize embedding
                    if len(embedding) > target_dim:
                        # Truncate
                        normalized_embeddings[name] = embedding[:target_dim]
                    else:
                        # Pad with zeros
                        padded = np.zeros(target_dim, dtype=np.float32)
                        padded[:len(embedding)] = embedding
                        normalized_embeddings[name] = padded
                else:
                    normalized_embeddings[name] = embedding
            
            # Calculate weights
            weights = {}
            total_weight = 0.0
            
            if "semantic" in normalized_embeddings:
                weights["semantic"] = self.config.semantic_weight
                total_weight += self.config.semantic_weight
            
            if "mathematical" in normalized_embeddings:
                weights["mathematical"] = self.config.mathematical_weight
                total_weight += self.config.mathematical_weight
            
            if "fractal" in normalized_embeddings:
                weights["fractal"] = self.config.fractal_weight
                total_weight += self.config.fractal_weight
            
            # Normalize weights
            if total_weight > 0:
                for name in weights:
                    weights[name] /= total_weight
            
            # Calculate weighted average
            fused = np.zeros(target_dim, dtype=np.float32)
            for name, embedding in normalized_embeddings.items():
                weight = weights.get(name, 0.0)
                fused += weight * embedding
            
            # Normalize result
            norm = np.linalg.norm(fused)
            if norm > 0:
                fused = fused / norm
            
            self.metrics["fusion_operations"] += 1
            return fused
            
        except Exception as e:
            logger.warning(f"Weighted average fusion failed: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def _concatenation_fusion(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenation fusion of embeddings"""
        try:
            # Concatenate all embeddings
            concatenated = np.concatenate(list(embeddings.values()))
            
            # Normalize
            norm = np.linalg.norm(concatenated)
            if norm > 0:
                concatenated = concatenated / norm
            
            self.metrics["fusion_operations"] += 1
            return concatenated
            
        except Exception as e:
            logger.warning(f"Concatenation fusion failed: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def _attention_fusion(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Attention-based fusion of embeddings"""
        try:
            # Simple attention mechanism
            # Calculate attention scores based on embedding magnitudes
            attention_scores = {}
            for name, embedding in embeddings.items():
                attention_scores[name] = np.linalg.norm(embedding)
            
            # Normalize attention scores
            total_score = sum(attention_scores.values())
            if total_score > 0:
                for name in attention_scores:
                    attention_scores[name] /= total_score
            
            # Apply attention weights
            target_dim = 768
            fused = np.zeros(target_dim, dtype=np.float32)
            
            for name, embedding in embeddings.items():
                # Normalize embedding dimension
                if len(embedding) != target_dim:
                    if len(embedding) > target_dim:
                        embedding = embedding[:target_dim]
                    else:
                        padded = np.zeros(target_dim, dtype=np.float32)
                        padded[:len(embedding)] = embedding
                        embedding = padded
                
                weight = attention_scores[name]
                fused += weight * embedding
            
            # Normalize result
            norm = np.linalg.norm(fused)
            if norm > 0:
                fused = fused / norm
            
            self.metrics["fusion_operations"] += 1
            return fused
            
        except Exception as e:
            logger.warning(f"Attention fusion failed: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def _update_metrics(self, result: Dict[str, Any]):
        """Update performance metrics"""
        self.metrics["total_embeddings"] += 1
        
        processing_time = result["metadata"]["processing_time"]
        if self.metrics["total_embeddings"] == 1:
            self.metrics["average_time"] = processing_time
        else:
            # Running average
            n = self.metrics["total_embeddings"]
            self.metrics["average_time"] = ((n - 1) * self.metrics["average_time"] + processing_time) / n
    
    async def embed_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Embed a batch of texts"""
        if self.config.parallel_processing:
            # Parallel batch processing
            tasks = [self.embed(text) for text in texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Batch embedding failed for text {i}: {result}")
                    processed_results.append({
                        "text": texts[i],
                        "error": str(result),
                        "embeddings": {},
                        "fused_embedding": np.zeros(768, dtype=np.float32),
                        "metadata": {"processing_time": 0.0}
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
        else:
            # Sequential batch processing
            results = []
            for text in texts:
                result = await self.embed(text)
                results.append(result)
            return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def clear_cache(self):
        """Clear embedding cache"""
        with self.cache_lock:
            self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self.cache_lock:
            return {
                "cache_size": len(self.embedding_cache),
                "cache_hits": self.metrics["cache_hits"]
            }
    
    async def close(self):
        """Close all embedders"""
        try:
            if self.semantic_embedder:
                await self.semantic_embedder.close()
            
            if self.mathematical_embedder:
                await self.mathematical_embedder.close()
                
            logger.info("All embedders closed")
        except Exception as e:
            logger.warning(f"Error closing embedders: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'semantic_embedder') and self.semantic_embedder:
            asyncio.create_task(self.close())
