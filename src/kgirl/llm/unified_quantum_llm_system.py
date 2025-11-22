"""
Unified Quantum LLM System
Integrates Quantum Knowledge + LIMPS + NuRea_sim into a comprehensive LLM platform

This system combines:
1. Quantum Holographic Knowledge System - Higher-dimensional knowledge representation
2. LIMPS Framework - GPU-accelerated matrix optimization
3. NuRea_sim - Matrix orchestrator, entropy engine, and ChaosRAGJulia vector database

The result is a production-ready LLM platform with:
- Multi-source knowledge ingestion (PDFs, code, text, equations)
- Quantum-inspired embeddings with chaos learning
- Multi-backend matrix optimization (GPU + Julia)
- Advanced entropy analysis and adaptive transformations
- Vector similarity search with PostgreSQL + pgvector
- Temporal causality tracking for knowledge evolution
- RAG-powered natural language querying
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import asyncio
import json
import httpx
from dataclasses import dataclass, field
from enum import Enum
import time

# Import Quantum-LIMPS integration
from quantum_limps_integration import (
    QuantumLIMPSIntegration,
    QuantumLIMPSConfig,
    OptimizedQuantumState
)

# Import Quantum Knowledge components
from quantum_holographic_knowledge_synthesis import KnowledgeQuantum, DataSourceType
from quantum_knowledge_database import QuantumHolographicKnowledgeDatabase

# Add NuRea_sim paths
NUREA_PATH = Path(__file__).parent / "NuRea_sim"
sys.path.insert(0, str(NUREA_PATH))
sys.path.insert(0, str(NUREA_PATH / "entropy engine"))

# Try importing NuRea components
try:
    from matrix_orchestrator import (
        JuliaBackend, MockBackend, OptimizeRequest, OptimizeResponse,
        MatrixChunk, RunPlan, EntropyReport
    )
    from ent.entropy_engine import Token, EntropyNode, EntropyEngine
    NUREA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"NuRea_sim components not available: {e}")
    NUREA_AVAILABLE = False
    JuliaBackend = None
    MockBackend = None
    Token = None
    EntropyNode = None
    EntropyEngine = None

logger = logging.getLogger(__name__)


class OptimizationBackend(Enum):
    """Available matrix optimization backends"""
    LIMPS_GPU = "limps_gpu"  # LIMPS GPU-accelerated processor
    LIMPS_CPU = "limps_cpu"  # LIMPS CPU processor
    NUREA_JULIA = "nurea_julia"  # NuRea Julia backend (OSQP, Convex.jl)
    NUREA_MOCK = "nurea_mock"  # NuRea mock backend for testing
    HYBRID = "hybrid"  # Use multiple backends and compare


@dataclass
class UnifiedSystemConfig:
    """Configuration for unified quantum LLM system"""
    # Quantum-LIMPS config
    use_gpu: bool = True
    matrix_precision: str = "float32"
    enable_limps_optimization: bool = True
    enable_limps_entropy: bool = True

    # NuRea config
    enable_nurea_optimization: bool = True
    enable_nurea_entropy: bool = True
    nurea_julia_url: str = "http://localhost:9000"
    chaos_rag_url: str = "http://localhost:8081"

    # Optimization backend selection
    primary_backend: OptimizationBackend = OptimizationBackend.HYBRID

    # ChaosRAGJulia vector database
    enable_vector_db: bool = True
    vector_dimensions: int = 1536
    postgres_url: str = "postgres://chaos_user:chaos_pass@localhost:5432/chaos"

    # Temporal causality
    enable_temporal_tracking: bool = True
    temporal_window: int = 10  # Track last N states

    # Performance
    max_concurrency: int = 4
    request_timeout: float = 60.0

    # Debug
    debug: bool = False


@dataclass
class UnifiedOptimizationResult:
    """Result from multi-backend optimization"""
    quantum_state: OptimizedQuantumState
    limps_results: Optional[Dict[str, Any]] = None
    nurea_results: Optional[Dict[str, Any]] = None
    backend_comparison: Optional[Dict[str, Any]] = None
    vector_embedding: Optional[np.ndarray] = None
    chaos_rag_id: Optional[str] = None
    temporal_predecessor: Optional[str] = None
    optimization_time: float = 0.0


class NuReaMatrixOptimizer:
    """Interface to NuRea_sim matrix orchestrator and Julia backend"""

    def __init__(self, config: UnifiedSystemConfig):
        self.config = config
        self.julia_backend = None
        self.http_client = None

        if NUREA_AVAILABLE and config.enable_nurea_optimization:
            self.julia_backend = JuliaBackend(config.nurea_julia_url)
            self.http_client = httpx.AsyncClient(timeout=config.request_timeout)
            logger.info(f"NuRea matrix optimizer initialized (Julia: {config.nurea_julia_url})")

    async def optimize(self, matrix: np.ndarray, method: str = "sparsity",
                      params: Optional[Dict] = None) -> Dict[str, Any]:
        """Optimize matrix using NuRea Julia backend"""
        if self.julia_backend is None:
            return {"error": "NuRea backend not available"}

        params = params or {}

        # Convert numpy array to list for JSON serialization
        matrix_list = matrix.tolist()

        # Create optimization request
        request = OptimizeRequest(
            matrix=matrix_list,
            method=method,
            params=params
        )

        try:
            # Call Julia backend
            response = await self.julia_backend.optimize(request)

            # Convert response
            optimized_matrix = np.array(response.matrix_opt)

            return {
                "optimized_matrix": optimized_matrix,
                "objective": response.objective,
                "iterations": response.iterations,
                "method": method,
                "backend": "nurea_julia",
                "meta": response.meta
            }
        except Exception as e:
            logger.error(f"NuRea optimization failed: {e}")
            return {"error": str(e), "optimized_matrix": matrix}

    async def close(self):
        """Close connections"""
        if self.julia_backend:
            await self.julia_backend.aclose()
        if self.http_client:
            await self.http_client.aclose()


class NuReaEntropyAnalyzer:
    """Interface to NuRea_sim entropy engine"""

    def __init__(self, config: UnifiedSystemConfig):
        self.config = config
        self.engine = None

        if NUREA_AVAILABLE and config.enable_nurea_entropy:
            # Create entropy transformation pipeline
            root = self._create_entropy_pipeline()
            self.engine = EntropyEngine(root, max_depth=5)
            logger.info("NuRea entropy analyzer initialized")

    def _create_entropy_pipeline(self) -> 'EntropyNode':
        """Create NuRea entropy transformation pipeline"""
        # Root: Identity transform
        def identity_transform(value, entropy):
            return value

        root = EntropyNode("root", identity_transform)

        # Child 1: Normalize
        def normalize_transform(value, entropy):
            if isinstance(value, (list, np.ndarray)):
                arr = np.array(value)
                norm = np.linalg.norm(arr)
                return (arr / (norm + 1e-8)).tolist() if norm > 0 else value
            return value

        normalize_node = EntropyNode("normalize", normalize_transform, entropy_limit=10.0)
        root.add_child(normalize_node)

        # Child 2: Adaptive filtering based on entropy
        def adaptive_filter(value, entropy):
            if isinstance(value, (list, np.ndarray)):
                arr = np.array(value)
                # Higher entropy -> stronger filtering
                threshold = np.std(arr) * (entropy / 10.0)
                return (arr * (np.abs(arr) > threshold)).tolist()
            return value

        filter_node = EntropyNode("adaptive_filter", adaptive_filter, entropy_limit=9.0)
        root.add_child(filter_node)

        return root

    def analyze(self, embedding: np.ndarray) -> Dict[str, float]:
        """Analyze embedding entropy using NuRea engine"""
        if self.engine is None:
            return {"error": "NuRea entropy engine not available"}

        # Create token
        token = Token(embedding.tolist())
        initial_entropy = token.entropy

        # Run through engine
        self.engine.run(token)

        # Export results
        graph = self.engine.export_graph()

        return {
            "initial_entropy": initial_entropy,
            "final_entropy": token.entropy,
            "entropy_delta": token.entropy - initial_entropy,
            "transformation_graph": graph,
            "backend": "nurea_entropy_engine"
        }


class ChaosRAGInterface:
    """Interface to ChaosRAGJulia vector database"""

    def __init__(self, config: UnifiedSystemConfig):
        self.config = config
        self.base_url = config.chaos_rag_url
        self.http_client = None

        if config.enable_vector_db:
            self.http_client = httpx.AsyncClient(timeout=config.request_timeout)
            logger.info(f"ChaosRAG interface initialized ({config.chaos_rag_url})")

    async def ingest_vector(self, vector: np.ndarray, metadata: Dict[str, Any]) -> Optional[str]:
        """Ingest vector into ChaosRAGJulia database"""
        if self.http_client is None:
            return None

        try:
            # Prepare payload
            payload = {
                "vector": vector.tolist(),
                "metadata": metadata,
                "dimension": len(vector)
            }

            # Call ChaosRAG ingest endpoint
            response = await self.http_client.post(
                f"{self.base_url}/ingest",
                json=payload
            )
            response.raise_for_status()

            result = response.json()
            return result.get("id")

        except Exception as e:
            logger.error(f"ChaosRAG ingestion failed: {e}")
            return None

    async def query_similar(self, query_vector: np.ndarray, limit: int = 5) -> List[Dict]:
        """Query similar vectors from ChaosRAGJulia"""
        if self.http_client is None:
            return []

        try:
            payload = {
                "query_vector": query_vector.tolist(),
                "limit": limit
            }

            response = await self.http_client.post(
                f"{self.base_url}/query",
                json=payload
            )
            response.raise_for_status()

            return response.json().get("results", [])

        except Exception as e:
            logger.error(f"ChaosRAG query failed: {e}")
            return []

    async def query_rag(self, query_text: str, context_limit: int = 5) -> Dict[str, Any]:
        """Query using RAG (Retrieval-Augmented Generation)"""
        if self.http_client is None:
            return {"error": "ChaosRAG not available"}

        try:
            payload = {
                "query": query_text,
                "limit": context_limit
            }

            response = await self.http_client.post(
                f"{self.base_url}/rag_query",
                json=payload
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"ChaosRAG RAG query failed: {e}")
            return {"error": str(e)}

    async def close(self):
        """Close HTTP client"""
        if self.http_client:
            await self.http_client.aclose()


class TemporalCausalityTracker:
    """Tracks temporal relationships between knowledge states"""

    def __init__(self, config: UnifiedSystemConfig):
        self.config = config
        self.states: List[Dict[str, Any]] = []
        self.edges: List[Tuple[str, str, float]] = []  # (from_id, to_id, weight)

        logger.info(f"Temporal causality tracker initialized (window={config.temporal_window})")

    def add_state(self, quantum_id: str, metadata: Dict[str, Any]) -> None:
        """Add a new knowledge state"""
        state = {
            "quantum_id": quantum_id,
            "timestamp": time.time(),
            "metadata": metadata
        }

        self.states.append(state)

        # Create temporal edges
        if len(self.states) > 1:
            # Step-1 edge: i -> i+1 (weight 1.0)
            prev_state = self.states[-2]
            self.edges.append((prev_state["quantum_id"], quantum_id, 1.0))

            # Step-5 edge: i -> i+5 (weight 0.6) if exists
            if len(self.states) >= 6:
                prev_5_state = self.states[-6]
                self.edges.append((prev_5_state["quantum_id"], quantum_id, 0.6))

        # Maintain window size
        if len(self.states) > self.config.temporal_window:
            removed = self.states.pop(0)
            # Remove edges involving removed state
            self.edges = [(f, t, w) for f, t, w in self.edges
                         if f != removed["quantum_id"] and t != removed["quantum_id"]]

    def get_predecessor(self, quantum_id: str) -> Optional[str]:
        """Get immediate predecessor state"""
        for from_id, to_id, weight in self.edges:
            if to_id == quantum_id and weight == 1.0:  # Step-1 edge
                return from_id
        return None

    def get_causal_chain(self, quantum_id: str, max_depth: int = 5) -> List[str]:
        """Get causal chain leading to this state"""
        chain = [quantum_id]
        current = quantum_id

        for _ in range(max_depth):
            pred = self.get_predecessor(current)
            if pred is None:
                break
            chain.insert(0, pred)
            current = pred

        return chain

    def export_graph(self) -> Dict[str, Any]:
        """Export temporal graph"""
        return {
            "states": self.states,
            "edges": [{"from": f, "to": t, "weight": w} for f, t, w in self.edges],
            "window_size": self.config.temporal_window
        }


class UnifiedQuantumLLMSystem:
    """
    Main unified system integrating Quantum Knowledge + LIMPS + NuRea_sim

    This provides a comprehensive LLM platform with:
    - Multi-source knowledge ingestion
    - Quantum-inspired embeddings
    - Multi-backend optimization (LIMPS GPU + NuRea Julia)
    - Advanced entropy analysis (dual engines)
    - Vector database with RAG
    - Temporal causality tracking
    """

    def __init__(self, config: Optional[UnifiedSystemConfig] = None):
        self.config = config or UnifiedSystemConfig()

        # Initialize Quantum-LIMPS integration
        limps_config = QuantumLIMPSConfig(
            use_gpu=self.config.use_gpu,
            matrix_precision=self.config.matrix_precision,
            enable_matrix_optimization=self.config.enable_limps_optimization,
            enable_entropy_analysis=self.config.enable_limps_entropy,
            debug=self.config.debug
        )
        self.quantum_limps = QuantumLIMPSIntegration(limps_config)

        # Initialize NuRea components
        self.nurea_optimizer = NuReaMatrixOptimizer(self.config)
        self.nurea_entropy = NuReaEntropyAnalyzer(self.config)
        self.chaos_rag = ChaosRAGInterface(self.config)

        # Initialize temporal tracker
        self.temporal_tracker = None
        if self.config.enable_temporal_tracking:
            self.temporal_tracker = TemporalCausalityTracker(self.config)

        logger.info("Unified Quantum LLM System initialized")

    async def ingest_and_process(self, source: Union[str, Path]) -> UnifiedOptimizationResult:
        """
        Ingest and process knowledge with full multi-backend optimization

        Args:
            source: Path to data source (PDF, code, text, etc.)

        Returns:
            UnifiedOptimizationResult with all backends' results
        """
        start_time = time.time()

        logger.info(f"Ingesting source: {source}")

        # Step 1: Process through Quantum-LIMPS
        quantum_state = await self.quantum_limps.ingest_and_optimize(source)

        limps_results = {
            "compression_ratio": quantum_state.compression_ratio,
            "complexity_score": quantum_state.complexity_score,
            "entropy_metrics": quantum_state.entropy_metrics,
            "optimized_embeddings": len(quantum_state.optimized_embeddings)
        }

        # Step 2: Process through NuRea optimizer (if enabled and hybrid mode)
        nurea_results = None
        if self.config.primary_backend in [OptimizationBackend.NUREA_JULIA, OptimizationBackend.HYBRID]:
            nurea_results = await self._optimize_with_nurea(quantum_state)

        # Step 3: Compare backends (if hybrid mode)
        backend_comparison = None
        if self.config.primary_backend == OptimizationBackend.HYBRID:
            backend_comparison = self._compare_backends(limps_results, nurea_results)

        # Step 4: Create vector embedding for ChaosRAG
        vector_embedding = None
        chaos_rag_id = None
        if self.config.enable_vector_db and quantum_state.original_quantum.hybrid_embedding is not None:
            # Use hybrid embedding as base
            base_embedding = quantum_state.original_quantum.hybrid_embedding

            # Resize to target dimensions if needed
            if len(base_embedding) < self.config.vector_dimensions:
                # Pad with zeros
                vector_embedding = np.pad(
                    base_embedding,
                    (0, self.config.vector_dimensions - len(base_embedding))
                )
            else:
                # Truncate or interpolate
                indices = np.linspace(0, len(base_embedding) - 1, self.config.vector_dimensions).astype(int)
                vector_embedding = base_embedding[indices]

            # Normalize
            vector_embedding = vector_embedding / (np.linalg.norm(vector_embedding) + 1e-8)

            # Ingest into ChaosRAG
            metadata = {
                "quantum_id": quantum_state.original_quantum.quantum_id,
                "source_type": quantum_state.original_quantum.source_type.value,
                "coherence_resonance": quantum_state.original_quantum.coherence_resonance,
                "complexity_score": quantum_state.complexity_score,
                "timestamp": time.time()
            }

            chaos_rag_id = await self.chaos_rag.ingest_vector(vector_embedding, metadata)

        # Step 5: Track temporal causality
        temporal_predecessor = None
        if self.temporal_tracker:
            temporal_predecessor = self.temporal_tracker.get_predecessor(
                quantum_state.original_quantum.quantum_id
            )

            self.temporal_tracker.add_state(
                quantum_state.original_quantum.quantum_id,
                {"source": str(source), "chaos_rag_id": chaos_rag_id}
            )

        optimization_time = time.time() - start_time

        result = UnifiedOptimizationResult(
            quantum_state=quantum_state,
            limps_results=limps_results,
            nurea_results=nurea_results,
            backend_comparison=backend_comparison,
            vector_embedding=vector_embedding,
            chaos_rag_id=chaos_rag_id,
            temporal_predecessor=temporal_predecessor,
            optimization_time=optimization_time
        )

        logger.info(f"Processing complete in {optimization_time:.2f}s")

        return result

    async def _optimize_with_nurea(self, quantum_state: OptimizedQuantumState) -> Dict[str, Any]:
        """Optimize quantum state using NuRea backend"""
        results = {}

        # Optimize each embedding type
        for emb_type, embedding in quantum_state.optimized_embeddings.items():
            # Convert 1D to 2D matrix
            size = int(np.sqrt(len(embedding)))
            if size * size == len(embedding):
                matrix = embedding.reshape(size, size)
            else:
                # Pad to square
                target_size = int(np.ceil(np.sqrt(len(embedding))))
                padded = np.pad(embedding, (0, target_size**2 - len(embedding)))
                matrix = padded.reshape(target_size, target_size)

            # Optimize with NuRea
            result = await self.nurea_optimizer.optimize(matrix, method="sparsity")
            results[emb_type] = result

        return results

    def _compare_backends(self, limps_results: Dict, nurea_results: Dict) -> Dict[str, Any]:
        """Compare optimization results from different backends"""
        if nurea_results is None:
            return {"error": "NuRea results not available"}

        # Calculate average compression from NuRea
        nurea_compressions = []
        for result in nurea_results.values():
            if "optimized_matrix" in result and not isinstance(result.get("optimized_matrix"), str):
                # Calculate compression
                opt_matrix = result["optimized_matrix"]
                sparsity = 1.0 - np.count_nonzero(opt_matrix) / opt_matrix.size
                nurea_compressions.append(sparsity)

        avg_nurea_compression = np.mean(nurea_compressions) if nurea_compressions else 0.0

        return {
            "limps_compression": limps_results["compression_ratio"],
            "nurea_compression": avg_nurea_compression,
            "best_backend": "LIMPS" if limps_results["compression_ratio"] > avg_nurea_compression else "NuRea",
            "compression_improvement": abs(limps_results["compression_ratio"] - avg_nurea_compression)
        }

    async def query_llm(self, query: str, use_rag: bool = True, context_limit: int = 5) -> Dict[str, Any]:
        """
        Query the system using natural language with RAG

        Args:
            query: Natural language query
            use_rag: Whether to use ChaosRAG for retrieval
            context_limit: Maximum context items to retrieve

        Returns:
            Query results with relevant knowledge
        """
        logger.info(f"Processing query: {query}")

        results = {
            "query": query,
            "timestamp": time.time()
        }

        if use_rag and self.config.enable_vector_db:
            # Use ChaosRAG
            rag_results = await self.chaos_rag.query_rag(query, context_limit)
            results["rag_results"] = rag_results
        else:
            # Use quantum database
            quantum_results = await self.quantum_limps.query_with_llm(query, context_limit)
            results["quantum_results"] = quantum_results

        return results

    def get_temporal_graph(self) -> Optional[Dict[str, Any]]:
        """Export temporal causality graph"""
        if self.temporal_tracker:
            return self.temporal_tracker.export_graph()
        return None

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "quantum_limps": self.quantum_limps.get_system_status(),
            "nurea_available": NUREA_AVAILABLE,
            "nurea_optimization_enabled": self.config.enable_nurea_optimization,
            "nurea_entropy_enabled": self.config.enable_nurea_entropy,
            "vector_db_enabled": self.config.enable_vector_db,
            "temporal_tracking_enabled": self.config.enable_temporal_tracking,
            "primary_backend": self.config.primary_backend.value,
            "backends": {
                "limps_gpu": self.config.enable_limps_optimization and self.config.use_gpu,
                "limps_cpu": self.config.enable_limps_optimization and not self.config.use_gpu,
                "nurea_julia": self.config.enable_nurea_optimization,
                "chaos_rag": self.config.enable_vector_db
            }
        }

        if self.temporal_tracker:
            graph = self.temporal_tracker.export_graph()
            status["temporal_states"] = len(graph["states"])
            status["temporal_edges"] = len(graph["edges"])

        return status

    async def close(self):
        """Close all connections"""
        await self.nurea_optimizer.close()
        await self.chaos_rag.close()


# Convenience functions

async def process_with_unified_system(source: Union[str, Path],
                                     config: Optional[UnifiedSystemConfig] = None) -> UnifiedOptimizationResult:
    """
    Convenience function to process a single source with the unified system

    Example:
        >>> result = await process_with_unified_system("research_paper.pdf")
        >>> print(f"LIMPS compression: {result.limps_results['compression_ratio']:.2%}")
        >>> print(f"NuRea compression: {result.nurea_results['avg_compression']:.2%}")
    """
    system = UnifiedQuantumLLMSystem(config)
    try:
        return await system.ingest_and_process(source)
    finally:
        await system.close()


async def query_unified_llm(query: str,
                           config: Optional[UnifiedSystemConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to query the unified system

    Example:
        >>> results = await query_unified_llm("quantum entanglement patterns")
        >>> print(f"Found {len(results['rag_results'])} relevant documents")
    """
    system = UnifiedQuantumLLMSystem(config)
    try:
        return await system.query_llm(query)
    finally:
        await system.close()


def create_unified_system(primary_backend: str = "hybrid",
                         enable_all: bool = True,
                         use_gpu: bool = True) -> UnifiedQuantumLLMSystem:
    """
    Create a UnifiedQuantumLLMSystem with simplified configuration

    Args:
        primary_backend: "limps_gpu", "limps_cpu", "nurea_julia", or "hybrid"
        enable_all: Enable all features
        use_gpu: Use GPU acceleration for LIMPS

    Example:
        >>> system = create_unified_system(primary_backend="hybrid", use_gpu=True)
        >>> status = system.get_system_status()
    """
    backend_map = {
        "limps_gpu": OptimizationBackend.LIMPS_GPU,
        "limps_cpu": OptimizationBackend.LIMPS_CPU,
        "nurea_julia": OptimizationBackend.NUREA_JULIA,
        "hybrid": OptimizationBackend.HYBRID
    }

    config = UnifiedSystemConfig(
        use_gpu=use_gpu,
        primary_backend=backend_map.get(primary_backend, OptimizationBackend.HYBRID),
        enable_limps_optimization=enable_all,
        enable_limps_entropy=enable_all,
        enable_nurea_optimization=enable_all and NUREA_AVAILABLE,
        enable_nurea_entropy=enable_all and NUREA_AVAILABLE,
        enable_vector_db=enable_all,
        enable_temporal_tracking=enable_all
    )

    return UnifiedQuantumLLMSystem(config)


if __name__ == "__main__":
    # Demonstration
    logging.basicConfig(level=logging.INFO)

    async def demonstrate_unified_system():
        """Demonstrate the unified system capabilities"""
        logger.info("=" * 70)
        logger.info("UNIFIED QUANTUM LLM SYSTEM DEMONSTRATION")
        logger.info("=" * 70)

        # Create system
        system = create_unified_system(primary_backend="hybrid", use_gpu=False)

        try:
            # Show status
            logger.info("\n📊 System Status:")
            status = system.get_system_status()
            for key, value in status.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for k, v in value.items():
                        logger.info(f"    {k}: {v}")
                else:
                    logger.info(f"  {key}: {value}")

            # Test with sample data
            logger.info("\n🔬 Processing sample knowledge...")

            import tempfile
            sample_text = "Quantum mechanics and nuclear physics share fundamental principles."

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(sample_text)
                temp_file = f.name

            try:
                result = await system.ingest_and_process(temp_file)

                logger.info("\n✅ Processing Results:")
                logger.info(f"  Optimization Time: {result.optimization_time:.2f}s")

                if result.limps_results:
                    logger.info(f"  LIMPS Compression: {result.limps_results['compression_ratio']:.2%}")
                    logger.info(f"  Complexity Score: {result.limps_results['complexity_score']:.4f}")

                if result.backend_comparison:
                    logger.info(f"  Best Backend: {result.backend_comparison['best_backend']}")

                if result.chaos_rag_id:
                    logger.info(f"  ChaosRAG ID: {result.chaos_rag_id}")

                # Test query
                logger.info("\n🔍 Testing LLM Query...")
                query_result = await system.query_llm("quantum mechanics", use_rag=False)
                logger.info(f"  Query: {query_result['query']}")
                logger.info(f"  Results: {len(query_result.get('quantum_results', {}).get('results', []))}")

            finally:
                import os
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

            logger.info("\n✨ Demonstration complete!")

        finally:
            await system.close()

    # Run demonstration
    asyncio.run(demonstrate_unified_system())
