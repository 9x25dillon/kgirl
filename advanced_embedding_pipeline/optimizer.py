#!/usr/bin/env python3
"""
Embedding Optimizer - Performance optimization and caching
Advanced optimization strategies for the embedding pipeline
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import hashlib
from pathlib import Path
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for embedding optimization"""
    # Caching settings
    use_disk_cache: bool = True
    cache_directory: str = "./cache/optimized_embeddings"
    max_cache_size_mb: int = 1000
    cache_compression: bool = True
    
    # Performance settings
    use_gpu: bool = False
    batch_processing: bool = True
    max_batch_size: int = 64
    prefetch_embeddings: bool = True
    
    # Memory optimization
    use_memory_mapping: bool = False
    max_memory_usage_mb: int = 2048
    garbage_collection_frequency: int = 100
    
    # Query optimization
    use_indexing: bool = True
    index_type: str = "faiss"  # "faiss", "annoy", "hnswlib"
    index_dimensions: int = 768
    
    # Adaptive optimization
    adaptive_batching: bool = True
    performance_monitoring: bool = True
    auto_tuning: bool = True


class EmbeddingCache:
    """Advanced embedding cache with disk persistence and compression"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "disk_hits": 0,
            "disk_misses": 0
        }
        self.cache_lock = threading.RLock()
        
        # Setup cache directory
        if self.config.use_disk_cache:
            self.cache_dir = Path(self.config.cache_directory)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._setup_disk_cache()
    
    def _setup_disk_cache(self):
        """Setup disk-based cache"""
        try:
            # SQLite database for cache metadata
            self.db_path = self.cache_dir / "cache.db"
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            
            # Create cache table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL NOT NULL,
                    size_bytes INTEGER NOT NULL
                )
            """)
            
            self.conn.commit()
            logger.info("✅ Disk cache initialized")
            
        except Exception as e:
            logger.warning(f"Disk cache setup failed: {e}")
            self.config.use_disk_cache = False
    
    def _get_cache_key(self, text: str, config_hash: str = "") -> str:
        """Generate cache key"""
        key_data = f"{text}_{config_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, text: str, config_hash: str = "") -> Optional[Dict[str, Any]]:
        """Get cached embedding"""
        cache_key = self._get_cache_key(text, config_hash)
        
        with self.cache_lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                self.cache_stats["hits"] += 1
                return self.memory_cache[cache_key]
            
            # Check disk cache
            if self.config.use_disk_cache:
                disk_result = self._get_from_disk(cache_key)
                if disk_result:
                    self.cache_stats["disk_hits"] += 1
                    # Load into memory cache
                    self.memory_cache[cache_key] = disk_result
                    return disk_result
                else:
                    self.cache_stats["disk_misses"] += 1
            
            self.cache_stats["misses"] += 1
            return None
    
    def _get_from_disk(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get embedding from disk cache"""
        try:
            cursor = self.conn.execute(
                "SELECT file_path FROM cache_metadata WHERE key = ?",
                (cache_key,)
            )
            result = cursor.fetchone()
            
            if result:
                file_path = Path(result[0])
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        if self.config.cache_compression:
                            import gzip
                            data = pickle.loads(gzip.decompress(f.read()))
                        else:
                            data = pickle.load(f)
                    
                    # Update access statistics
                    self.conn.execute(
                        "UPDATE cache_metadata SET access_count = access_count + 1, last_accessed = ? WHERE key = ?",
                        (time.time(), cache_key)
                    )
                    self.conn.commit()
                    
                    return data
            
            return None
            
        except Exception as e:
            logger.warning(f"Disk cache retrieval failed: {e}")
            return None
    
    def put(self, text: str, config_hash: str, embedding_data: Dict[str, Any]):
        """Store embedding in cache"""
        cache_key = self._get_cache_key(text, config_hash)
        
        with self.cache_lock:
            # Store in memory cache
            self.memory_cache[cache_key] = embedding_data
            
            # Store in disk cache
            if self.config.use_disk_cache:
                self._put_to_disk(cache_key, embedding_data)
            
            # Check memory usage
            self._check_memory_usage()
    
    def _put_to_disk(self, cache_key: str, embedding_data: Dict[str, Any]):
        """Store embedding to disk cache"""
        try:
            # Generate file path
            file_path = self.cache_dir / f"{cache_key}.pkl"
            
            # Serialize data
            serialized_data = pickle.dumps(embedding_data)
            
            # Compress if enabled
            if self.config.cache_compression:
                import gzip
                serialized_data = gzip.compress(serialized_data)
            
            # Write to disk
            with open(file_path, 'wb') as f:
                f.write(serialized_data)
            
            # Update metadata
            self.conn.execute(
                "INSERT OR REPLACE INTO cache_metadata (key, file_path, created_at, last_accessed, size_bytes) VALUES (?, ?, ?, ?, ?)",
                (cache_key, str(file_path), time.time(), time.time(), len(serialized_data))
            )
            self.conn.commit()
            
        except Exception as e:
            logger.warning(f"Disk cache storage failed: {e}")
    
    def _check_memory_usage(self):
        """Check and manage memory usage"""
        if len(self.memory_cache) > 1000:  # Arbitrary limit
            # Remove oldest entries
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1].get("metadata", {}).get("created_at", 0)
            )
            
            # Remove 20% of oldest entries
            remove_count = len(sorted_items) // 5
            for key, _ in sorted_items[:remove_count]:
                del self.memory_cache[key]
    
    def clear(self):
        """Clear all caches"""
        with self.cache_lock:
            self.memory_cache.clear()
            
            if self.config.use_disk_cache:
                try:
                    # Clear disk cache files
                    for file_path in self.cache_dir.glob("*.pkl"):
                        file_path.unlink()
                    
                    # Clear database
                    self.conn.execute("DELETE FROM cache_metadata")
                    self.conn.commit()
                    
                except Exception as e:
                    logger.warning(f"Disk cache clear failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.cache_lock:
            total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                **self.cache_stats,
                "memory_cache_size": len(self.memory_cache),
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }


class EmbeddingOptimizer:
    """Advanced embedding optimizer with caching, batching, and performance monitoring"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.cache = EmbeddingCache(self.config)
        self.performance_metrics = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "batch_operations": 0,
            "average_batch_size": 0.0,
            "average_processing_time": 0.0,
            "memory_usage_mb": 0.0
        }
        
        # Performance monitoring
        self.processing_times = []
        self.batch_sizes = []
        
        # Adaptive batching
        self.optimal_batch_size = self.config.max_batch_size
        
        logger.info("✅ Embedding optimizer initialized")
    
    async def optimize_embedding_generation(self, embedder_func: Callable, 
                                          texts: List[str], 
                                          config_hash: str = "") -> List[Dict[str, Any]]:
        """
        Optimize embedding generation with caching and batching
        
        Args:
            embedder_func: Function to generate embeddings
            texts: List of texts to embed
            config_hash: Configuration hash for cache key
            
        Returns:
            List of embedding results
        """
        start_time = time.time()
        
        try:
            # Check cache for all texts
            cached_results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cached_result = self.cache.get(text, config_hash)
                if cached_result:
                    cached_result["cached"] = True
                    cached_results.append(cached_result)
                    self.performance_metrics["cache_hits"] += 1
                else:
                    cached_results.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                if self.config.batch_processing:
                    # Batch processing
                    batch_results = await self._process_batch(embedder_func, uncached_texts, config_hash)
                    
                    # Fill in results
                    for i, result in zip(uncached_indices, batch_results):
                        cached_results[i] = result
                else:
                    # Individual processing
                    for i, text in zip(uncached_indices, uncached_texts):
                        result = await embedder_func(text)
                        result["cached"] = False
                        cached_results[i] = result
                        
                        # Cache the result
                        self.cache.put(text, config_hash, result)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(len(texts), processing_time)
            
            return cached_results
            
        except Exception as e:
            logger.error(f"❌ Optimized embedding generation failed: {e}")
            return []
    
    async def _process_batch(self, embedder_func: Callable, texts: List[str], 
                           config_hash: str) -> List[Dict[str, Any]]:
        """Process batch of texts with adaptive batching"""
        try:
            # Determine optimal batch size
            if self.config.adaptive_batching:
                optimal_size = self._calculate_optimal_batch_size()
            else:
                optimal_size = self.config.max_batch_size
            
            # Process in batches
            results = []
            for i in range(0, len(texts), optimal_size):
                batch = texts[i:i + optimal_size]
                
                # Process batch
                batch_start = time.time()
                batch_results = await embedder_func(batch)
                batch_time = time.time() - batch_start
                
                # Handle single embedding function vs batch function
                if isinstance(batch_results, dict):
                    # Single result for batch
                    for text in batch:
                        result = batch_results.copy()
                        result["cached"] = False
                        results.append(result)
                        
                        # Cache individual results
                        self.cache.put(text, config_hash, result)
                else:
                    # Multiple results
                    for text, result in zip(batch, batch_results):
                        result["cached"] = False
                        results.append(result)
                        
                        # Cache individual results
                        self.cache.put(text, config_hash, result)
                
                # Update batch metrics
                self.batch_sizes.append(len(batch))
                self.processing_times.append(batch_time)
                
                # Adaptive adjustment
                if self.config.adaptive_batching:
                    self._adjust_batch_size(batch_time, len(batch))
            
            self.performance_metrics["batch_operations"] += 1
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            return []
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on performance metrics"""
        if not self.processing_times or not self.batch_sizes:
            return self.config.max_batch_size
        
        # Simple heuristic: if processing time is increasing, reduce batch size
        if len(self.processing_times) > 1:
            recent_time = np.mean(self.processing_times[-3:])
            older_time = np.mean(self.processing_times[:-3]) if len(self.processing_times) > 3 else recent_time
            
            if recent_time > older_time * 1.2:  # 20% increase
                return max(1, self.optimal_batch_size // 2)
            elif recent_time < older_time * 0.8:  # 20% decrease
                return min(self.config.max_batch_size, self.optimal_batch_size * 2)
        
        return self.optimal_batch_size
    
    def _adjust_batch_size(self, processing_time: float, batch_size: int):
        """Adjust batch size based on processing time"""
        # Simple adaptive strategy
        if processing_time > 5.0:  # Too slow
            self.optimal_batch_size = max(1, batch_size // 2)
        elif processing_time < 1.0 and batch_size < self.config.max_batch_size:  # Too fast
            self.optimal_batch_size = min(self.config.max_batch_size, batch_size * 2)
    
    def _update_metrics(self, count: int, processing_time: float):
        """Update performance metrics"""
        self.performance_metrics["total_embeddings"] += count
        
        # Update average processing time
        total_ops = self.performance_metrics["total_embeddings"]
        if total_ops == count:
            self.performance_metrics["average_processing_time"] = processing_time
        else:
            current_avg = self.performance_metrics["average_processing_time"]
            self.performance_metrics["average_processing_time"] = (
                (current_avg * (total_ops - count) + processing_time * count) / total_ops
            )
        
        # Update average batch size
        if self.batch_sizes:
            self.performance_metrics["average_batch_size"] = np.mean(self.batch_sizes)
        
        # Update memory usage
        import psutil
        process = psutil.Process()
        self.performance_metrics["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
    
    def create_index(self, embeddings: List[np.ndarray], texts: List[str]) -> Dict[str, Any]:
        """Create search index for embeddings"""
        try:
            if not self.config.use_indexing or not embeddings:
                return {"index": None, "type": "none"}
            
            if self.config.index_type == "faiss":
                return self._create_faiss_index(embeddings, texts)
            elif self.config.index_type == "annoy":
                return self._create_annoy_index(embeddings, texts)
            else:
                logger.warning(f"Unsupported index type: {self.config.index_type}")
                return {"index": None, "type": "none"}
                
        except Exception as e:
            logger.warning(f"Index creation failed: {e}")
            return {"index": None, "type": "none"}
    
    def _create_faiss_index(self, embeddings: List[np.ndarray], texts: List[str]) -> Dict[str, Any]:
        """Create FAISS index"""
        try:
            import faiss
            
            # Normalize embeddings
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            
            # Create index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            
            # Add embeddings
            index.add(embeddings_array)
            
            return {
                "index": index,
                "type": "faiss",
                "dimension": dimension,
                "size": len(embeddings),
                "texts": texts
            }
            
        except ImportError:
            logger.warning("FAISS not available")
            return {"index": None, "type": "none"}
        except Exception as e:
            logger.warning(f"FAISS index creation failed: {e}")
            return {"index": None, "type": "none"}
    
    def _create_annoy_index(self, embeddings: List[np.ndarray], texts: List[str]) -> Dict[str, Any]:
        """Create Annoy index"""
        try:
            from annoy import AnnoyIndex
            
            if not embeddings:
                return {"index": None, "type": "none"}
            
            dimension = len(embeddings[0])
            index = AnnoyIndex(dimension, 'angular')  # Cosine similarity
            
            # Add embeddings
            for i, embedding in enumerate(embeddings):
                index.add_item(i, embedding)
            
            # Build index
            index.build(10)  # Number of trees
            
            return {
                "index": index,
                "type": "annoy",
                "dimension": dimension,
                "size": len(embeddings),
                "texts": texts
            }
            
        except ImportError:
            logger.warning("Annoy not available")
            return {"index": None, "type": "none"}
        except Exception as e:
            logger.warning(f"Annoy index creation failed: {e}")
            return {"index": None, "type": "none"}
    
    def search_similar(self, index_data: Dict[str, Any], query_embedding: np.ndarray, 
                      top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for similar embeddings using index"""
        try:
            if not index_data.get("index"):
                return []
            
            index_type = index_data["type"]
            
            if index_type == "faiss":
                return self._search_faiss(index_data, query_embedding, top_k)
            elif index_type == "annoy":
                return self._search_annoy(index_data, query_embedding, top_k)
            else:
                return []
                
        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")
            return []
    
    def _search_faiss(self, index_data: Dict[str, Any], query_embedding: np.ndarray, 
                     top_k: int) -> List[Tuple[int, float]]:
        """Search using FAISS index"""
        try:
            import faiss
            
            index = index_data["index"]
            query = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query)
            
            # Search
            scores, indices = index.search(query, top_k)
            
            # Convert to list of tuples
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid result
                    results.append((int(idx), float(score)))
            
            return results
            
        except Exception as e:
            logger.warning(f"FAISS search failed: {e}")
            return []
    
    def _search_annoy(self, index_data: Dict[str, Any], query_embedding: np.ndarray, 
                     top_k: int) -> List[Tuple[int, float]]:
        """Search using Annoy index"""
        try:
            index = index_data["index"]
            
            # Search
            indices, distances = index.get_nns_by_vector(query_embedding, top_k, include_distances=True)
            
            # Convert distances to similarities (Annoy returns distances, we want similarities)
            results = []
            for idx, dist in zip(indices, distances):
                similarity = 1.0 / (1.0 + dist)  # Convert distance to similarity
                results.append((int(idx), float(similarity)))
            
            return results
            
        except Exception as e:
            logger.warning(f"Annoy search failed: {e}")
            return []
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        cache_stats = self.cache.get_stats()
        
        return {
            "performance_metrics": self.performance_metrics.copy(),
            "cache_stats": cache_stats,
            "optimization_config": {
                "batch_processing": self.config.batch_processing,
                "adaptive_batching": self.config.adaptive_batching,
                "optimal_batch_size": self.optimal_batch_size,
                "use_indexing": self.config.use_indexing,
                "index_type": self.config.index_type
            },
            "recent_performance": {
                "recent_batch_sizes": self.batch_sizes[-10:] if self.batch_sizes else [],
                "recent_processing_times": self.processing_times[-10:] if self.processing_times else []
            }
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.cache.clear()
        logger.info("Optimizer cache cleared")
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "batch_operations": 0,
            "average_batch_size": 0.0,
            "average_processing_time": 0.0,
            "memory_usage_mb": 0.0
        }
        self.processing_times.clear()
        self.batch_sizes.clear()
        self.optimal_batch_size = self.config.max_batch_size
        logger.info("Performance metrics reset")
