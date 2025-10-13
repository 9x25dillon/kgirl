#!/usr/bin/env python3
"""
Matrix Processor - Vector Compilation and Optimization
Handles matrix operations, vector embeddings, and knowledge compilation
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

# Try to import FAISS, fallback to numpy if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, using numpy fallback")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatrixConfig:
    """Configuration for matrix operations"""
    embedding_dim: int = 768
    use_faiss: bool = True
    index_type: str = "flat"  # flat, ivf, hnsw
    optimization_level: int = 2  # 0=basic, 1=moderate, 2=aggressive
    cache_embeddings: bool = True
    similarity_threshold: float = 0.7

class VectorEmbedder:
    """Handles vector embedding operations"""
    
    def __init__(self, config: MatrixConfig):
        self.config = config
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
    async def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to vectors"""
        embeddings = []
        
        for text in texts:
            # Check cache first
            if self.config.cache_embeddings and text in self.embedding_cache:
                embeddings.append(self.embedding_cache[text])
                continue
            
            # Generate embedding (simplified - in real implementation, use actual embedding model)
            embedding = self._generate_embedding(text)
            
            if self.config.cache_embeddings:
                self.embedding_cache[text] = embedding
            
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text (simplified implementation)"""
        # In real implementation, this would use a proper embedding model
        # For now, create a deterministic vector based on text characteristics
        words = text.lower().split()
        vector = np.zeros(self.config.embedding_dim)
        
        # Simple hash-based embedding
        for i, word in enumerate(words):
            hash_val = hash(word) % self.config.embedding_dim
            vector[hash_val] += 1.0 / (i + 1)  # Weight by position
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector

class MatrixOptimizer:
    """Optimizes matrix operations and vector spaces"""
    
    def __init__(self, config: MatrixConfig):
        self.config = config
        
    def optimize(self, vectors: np.ndarray) -> np.ndarray:
        """Optimize vector matrix based on configuration level"""
        if self.config.optimization_level == 0:
            return vectors
        
        # Level 1: Basic normalization and centering
        if self.config.optimization_level >= 1:
            # Center the vectors
            vectors = vectors - np.mean(vectors, axis=0)
            # Normalize each vector
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms!=0)
        
        # Level 2: Advanced optimization
        if self.config.optimization_level >= 2:
            # Apply PCA-like dimensionality reduction if needed
            if vectors.shape[1] > 512:
                # Simple SVD-based reduction
                U, s, Vt = np.linalg.svd(vectors, full_matrices=False)
                # Keep top 512 components
                vectors = U[:, :512] @ np.diag(s[:512])
            
            # Apply noise reduction
            vectors = self._denoise_vectors(vectors)
        
        return vectors
    
    def _denoise_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Apply noise reduction to vectors"""
        # Simple smoothing filter
        if len(vectors) > 1:
            # Apply moving average smoothing
            smoothed = np.zeros_like(vectors)
            for i in range(len(vectors)):
                start_idx = max(0, i - 1)
                end_idx = min(len(vectors), i + 2)
                smoothed[i] = np.mean(vectors[start_idx:end_idx], axis=0)
            return smoothed
        return vectors

class KnowledgeCompiler:
    """Compiles knowledge from optimized matrices"""
    
    def __init__(self, config: MatrixConfig):
        self.config = config
        self.compiled_matrices: Dict[str, Any] = {}
        
    def compile(self, optimized_vectors: np.ndarray, metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compile optimized vectors into knowledge structure"""
        if metadata is None:
            metadata = [{"id": i, "text": f"insight_{i}"} for i in range(len(optimized_vectors))]
        
        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(optimized_vectors)
        
        # Find clusters
        clusters = self._find_clusters(optimized_vectors, similarity_matrix)
        
        # Generate knowledge graph structure
        knowledge_graph = self._build_knowledge_graph(optimized_vectors, metadata, clusters)
        
        # Compile statistics
        stats = self._compile_statistics(optimized_vectors, similarity_matrix, clusters)
        
        compiled = {
            "vectors": optimized_vectors.tolist(),
            "similarity_matrix": similarity_matrix.tolist(),
            "clusters": clusters,
            "knowledge_graph": knowledge_graph,
            "statistics": stats,
            "metadata": metadata,
            "timestamp": time.time()
        }
        
        return compiled
    
    def _calculate_similarity_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix"""
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms!=0)
        
        # Calculate cosine similarity
        similarity_matrix = np.dot(normalized, normalized.T)
        return similarity_matrix
    
    def _find_clusters(self, vectors: np.ndarray, similarity_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Find clusters in the vector space"""
        clusters = []
        n_vectors = len(vectors)
        visited = set()
        
        for i in range(n_vectors):
            if i in visited:
                continue
            
            # Start new cluster
            cluster = {
                "id": len(clusters),
                "members": [i],
                "centroid": vectors[i],
                "coherence": 0.0
            }
            
            # Find similar vectors
            for j in range(i + 1, n_vectors):
                if j in visited:
                    continue
                
                similarity = similarity_matrix[i, j]
                if similarity >= self.config.similarity_threshold:
                    cluster["members"].append(j)
                    visited.add(j)
            
            # Calculate cluster coherence
            if len(cluster["members"]) > 1:
                member_vectors = vectors[cluster["members"]]
                cluster["centroid"] = np.mean(member_vectors, axis=0)
                cluster["coherence"] = np.mean(similarity_matrix[np.ix_(cluster["members"], cluster["members"])])
            
            visited.add(i)
            clusters.append(cluster)
        
        return clusters
    
    def _build_knowledge_graph(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build knowledge graph from vectors and clusters"""
        nodes = []
        edges = []
        
        # Create nodes
        for i, meta in enumerate(metadata):
            node = {
                "id": f"node_{i}",
                "vector_id": i,
                "metadata": meta,
                "cluster_id": None
            }
            
            # Assign cluster
            for cluster in clusters:
                if i in cluster["members"]:
                    node["cluster_id"] = cluster["id"]
                    break
            
            nodes.append(node)
        
        # Create edges based on similarity
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                similarity = np.dot(vectors[i], vectors[j])
                if similarity >= self.config.similarity_threshold:
                    edge = {
                        "source": f"node_{i}",
                        "target": f"node_{j}",
                        "weight": float(similarity),
                        "type": "semantic_similarity"
                    }
                    edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters
        }
    
    def _compile_statistics(self, vectors: np.ndarray, similarity_matrix: np.ndarray, clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile statistics about the knowledge matrix"""
        n_vectors = len(vectors)
        
        # Basic statistics
        stats = {
            "total_vectors": n_vectors,
            "embedding_dimension": vectors.shape[1],
            "clusters_found": len(clusters),
            "avg_cluster_size": np.mean([len(c["members"]) for c in clusters]) if clusters else 0,
            "similarity_stats": {
                "mean": float(np.mean(similarity_matrix)),
                "std": float(np.std(similarity_matrix)),
                "min": float(np.min(similarity_matrix)),
                "max": float(np.max(similarity_matrix))
            }
        }
        
        # Cluster statistics
        if clusters:
            cluster_coherences = [c["coherence"] for c in clusters if c["coherence"] > 0]
            stats["cluster_coherence"] = {
                "mean": float(np.mean(cluster_coherences)) if cluster_coherences else 0,
                "max": float(np.max(cluster_coherences)) if cluster_coherences else 0
            }
        
        return stats

class MatrixProcessor:
    """Main matrix processing orchestrator"""
    
    def __init__(self, config: MatrixConfig = None):
        self.config = config or MatrixConfig()
        self.embedder = VectorEmbedder(self.config)
        self.optimizer = MatrixOptimizer(self.config)
        self.compiler = KnowledgeCompiler(self.config)
        
    async def process_matrices(self, texts: List[str], metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main processing pipeline: embed -> optimize -> compile"""
        logger.info(f"Processing {len(texts)} texts through matrix pipeline")
        
        # Step 1: Encode texts to vectors
        start_time = time.time()
        vectors = await self.embedder.encode(texts)
        embed_time = time.time() - start_time
        
        # Step 2: Optimize vectors
        start_time = time.time()
        optimized_vectors = self.optimizer.optimize(vectors)
        optimize_time = time.time() - start_time
        
        # Step 3: Compile knowledge
        start_time = time.time()
        compiled_knowledge = self.compiler.compile(optimized_vectors, metadata)
        compile_time = time.time() - start_time
        
        # Add timing information
        compiled_knowledge["processing_times"] = {
            "embedding": embed_time,
            "optimization": optimize_time,
            "compilation": compile_time,
            "total": embed_time + optimize_time + compile_time
        }
        
        logger.info(f"Matrix processing complete in {compiled_knowledge['processing_times']['total']:.2f}s")
        
        return compiled_knowledge
    
    async def compile_matrices(self, vectors: np.ndarray, metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compile pre-existing vectors into knowledge structure"""
        return self.compiler.compile(vectors, metadata)

# FAISS-based knowledge base
class FAISSKnowledgeBase:
    """FAISS-based distributed knowledge base"""
    
    def __init__(self, index_path: str = "faiss_index", dimension: int = 768):
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self.metadata = []
        
    def initialize(self):
        """Initialize FAISS index"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, using numpy fallback")
            return False
        
        try:
            # Create FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"FAISS index initialized with dimension {self.dimension}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            return False
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]] = None):
        """Add embeddings to the index"""
        if self.index is None:
            if not self.initialize():
                return False
        
        try:
            # Ensure embeddings are float32
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            
            # Add to index
            self.index.add(embeddings)
            
            # Store metadata
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{"id": i} for i in range(len(embeddings))])
            
            logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            return False
    
    def query(self, vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Query the index for similar vectors"""
        if self.index is None or self.index.ntotal == 0:
            return np.array([]), np.array([])
        
        try:
            # Ensure vector is float32 and 2D
            if vector.dtype != np.float32:
                vector = vector.astype(np.float32)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            
            # Search
            distances, indices = self.index.search(vector, k)
            return distances[0], indices[0]
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return np.array([]), np.array([])
    
    def save_index(self):
        """Save index to disk"""
        if self.index is not None:
            try:
                faiss.write_index(self.index, self.index_path)
                logger.info(f"FAISS index saved to {self.index_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to save index: {e}")
                return False
        return False
    
    def load_index(self):
        """Load index from disk"""
        try:
            self.index = faiss.read_index(self.index_path)
            logger.info(f"FAISS index loaded from {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

# Demo function
async def demo_matrix_processing():
    """Demonstrate matrix processing capabilities"""
    config = MatrixConfig(embedding_dim=128, optimization_level=2)
    processor = MatrixProcessor(config)
    
    # Sample texts
    texts = [
        "Quantum computing uses superposition and entanglement",
        "Neural networks learn patterns from data",
        "Cognitive systems emerge from recursive processing",
        "Machine learning algorithms optimize performance",
        "Artificial intelligence mimics human cognition"
    ]
    
    metadata = [{"id": i, "text": text, "category": "ai_concept"} for i, text in enumerate(texts)]
    
    # Process matrices
    result = await processor.process_matrices(texts, metadata)
    
    print("Matrix Processing Results:")
    print(f"Total vectors: {result['statistics']['total_vectors']}")
    print(f"Clusters found: {result['statistics']['clusters_found']}")
    print(f"Processing time: {result['processing_times']['total']:.2f}s")
    
    # Test FAISS knowledge base
    kb = FAISSKnowledgeBase(dimension=128)
    if kb.initialize():
        # Add some test vectors
        test_vectors = np.random.rand(5, 128).astype(np.float32)
        kb.add_embeddings(test_vectors, metadata)
        
        # Query
        query_vector = np.random.rand(128).astype(np.float32)
        distances, indices = kb.query(query_vector, k=3)
        print(f"Query results: {len(indices)} similar vectors found")

if __name__ == "__main__":
    asyncio.run(demo_matrix_processing())