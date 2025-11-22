#!/usr/bin/env python3
"""
Complete System Integration: All LiMp + Numbskull Components
===========================================================

Master integration bringing together EVERYTHING:

LiMp Components:
- Chaos LLM API (QGI, retrieval, unitary mixer)
- AL-ULS (symbolic evaluation)
- TA ULS Transformer
- Neuro-Symbolic Engine
- Holographic Memory
- Signal Processing
- Evolutionary Communicator
- Quantum Cognitive Processor
- Entropy Engine
- Graph Store
- Vector Index

Numbskull Components:
- Semantic Embeddings (Eopiez)
- Mathematical Embeddings (LIMPS)
- Fractal Embeddings (local)
- Hybrid Fusion
- Embedding Optimizer
- Pipeline Cache

LFM2-8B-A1B:
- Local LLM inference
- Dual orchestration
- Embedding-enhanced context

Author: Assistant
License: MIT
"""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add paths
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

# Import all available components
from unified_cognitive_orchestrator import UnifiedCognitiveOrchestrator
from enhanced_vector_index import EnhancedVectorIndex
from enhanced_graph_store import EnhancedGraphStore

try:
    from src.chaos_llm.services.entropy_engine import entropy_engine
    ENTROPY_AVAILABLE = True
except:
    ENTROPY_AVAILABLE = False

try:
    from src.chaos_llm.services.al_uls import al_uls
    ALULS_AVAILABLE = True
except:
    ALULS_AVAILABLE = False

try:
    from entropy_engine import EntropyEngine as LiMpEntropyEngine
    LIMP_ENTROPY_AVAILABLE = True
except:
    LIMP_ENTROPY_AVAILABLE = False

try:
    from evolutionary_communicator import EvolutionaryCommunicator
    EVOL_COMM_AVAILABLE = True
except:
    EVOL_COMM_AVAILABLE = False

try:
    from quantum_cognitive_processor import QuantumNeuralNetwork, QuantumWalkOptimizer
    import torch
    QUANTUM_AVAILABLE = True
except:
    QUANTUM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Complete system state across all modules"""
    embeddings: Optional[Dict[str, Any]] = None
    vector_index_stats: Dict[str, Any] = field(default_factory=dict)
    graph_stats: Dict[str, Any] = field(default_factory=dict)
    cognitive_results: Dict[str, Any] = field(default_factory=dict)
    entropy_scores: Dict[str, float] = field(default_factory=dict)
    symbolic_calls: List[Dict[str, Any]] = field(default_factory=list)
    quantum_state: Optional[Dict[str, Any]] = None
    processing_history: List[Dict[str, Any]] = field(default_factory=list)


class CompleteSystemIntegration:
    """
    Master integration of ALL LiMp + Numbskull components
    
    Provides unified access to:
    - Cognitive orchestration (Numbskull + LiMp)
    - Vector indexing (embeddings + search)
    - Knowledge graphs (semantic + structural)
    - Entropy analysis (token + content)
    - Symbolic evaluation (AL-ULS)
    - Quantum processing (QNN)
    - Evolutionary communication
    - And more...
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize complete system integration
        
        Args:
            config: Optional system-wide configuration
        """
        self.config = config or self._default_config()
        self.state = SystemState()
        
        logger.info("=" * 70)
        logger.info("COMPLETE SYSTEM INTEGRATION INITIALIZING")
        logger.info("=" * 70)
        
        # Initialize all subsystems
        self.cognitive_orch = None
        self.vector_index = None
        self.graph_store = None
        self.evol_comm = None
        self.quantum_processor = None
        
        asyncio.run(self._initialize_subsystems())
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default system configuration"""
        return {
            "llm": {
                "base_url": "http://127.0.0.1:8080",
                "mode": "llama-cpp",
                "model": "LFM2-8B-A1B",
                "timeout": 120
            },
            "numbskull": {
                "use_semantic": False,
                "use_mathematical": False,
                "use_fractal": True,
                "fusion_method": "weighted_average"
            },
            "vector_index": {
                "embedding_dim": 768,
                "use_numbskull": True
            },
            "graph_store": {
                "use_numbskull": True
            },
            "enable_quantum": QUANTUM_AVAILABLE,
            "enable_evolution": EVOL_COMM_AVAILABLE
        }
    
    async def _initialize_subsystems(self):
        """Initialize all subsystems"""
        
        # 1. Unified Cognitive Orchestrator
        logger.info("\n1. Initializing Unified Cognitive Orchestrator...")
        try:
            self.cognitive_orch = UnifiedCognitiveOrchestrator(
                local_llm_config=self.config["llm"],
                numbskull_config=self.config["numbskull"],
                enable_tauls=False,  # Requires PyTorch
                enable_neurosymbolic=True,
                enable_holographic=False  # Requires PyTorch
            )
            logger.info("   ✅ Cognitive orchestrator ready")
        except Exception as e:
            logger.warning(f"   ⚠️  Cognitive orchestrator init failed: {e}")
        
        # 2. Enhanced Vector Index
        logger.info("2. Initializing Enhanced Vector Index...")
        try:
            self.vector_index = EnhancedVectorIndex(**self.config["vector_index"])
            logger.info("   ✅ Vector index ready")
        except Exception as e:
            logger.warning(f"   ⚠️  Vector index init failed: {e}")
        
        # 3. Enhanced Graph Store
        logger.info("3. Initializing Enhanced Graph Store...")
        try:
            self.graph_store = EnhancedGraphStore(**self.config["graph_store"])
            logger.info("   ✅ Graph store ready")
        except Exception as e:
            logger.warning(f"   ⚠️  Graph store init failed: {e}")
        
        # 4. Evolutionary Communicator
        if self.config.get("enable_evolution") and EVOL_COMM_AVAILABLE:
            logger.info("4. Initializing Evolutionary Communicator...")
            try:
                self.evol_comm = EvolutionaryCommunicator()
                logger.info("   ✅ Evolutionary communicator ready")
            except Exception as e:
                logger.warning(f"   ⚠️  Evolutionary communicator init failed: {e}")
        
        # 5. Quantum Processor
        if self.config.get("enable_quantum") and QUANTUM_AVAILABLE:
            logger.info("5. Initializing Quantum Processor...")
            try:
                self.quantum_processor = QuantumNeuralNetwork(num_qubits=4, num_layers=2)
                logger.info("   ✅ Quantum processor ready")
            except Exception as e:
                logger.warning(f"   ⚠️  Quantum processor init failed: {e}")
        
        logger.info("\n" + "=" * 70)
        logger.info("COMPLETE SYSTEM READY")
        logger.info("=" * 70)
        self._print_system_status()
    
    def _print_system_status(self):
        """Print complete system status"""
        logger.info("\n🎯 Active Components:")
        logger.info(f"  Cognitive Orchestrator:     {'✅ Active' if self.cognitive_orch else '❌ Inactive'}")
        logger.info(f"  Vector Index:               {'✅ Active' if self.vector_index else '❌ Inactive'}")
        logger.info(f"  Graph Store:                {'✅ Active' if self.graph_store else '❌ Inactive'}")
        logger.info(f"  Evolutionary Comm:          {'✅ Active' if self.evol_comm else '❌ Inactive'}")
        logger.info(f"  Quantum Processor:          {'✅ Active' if self.quantum_processor else '❌ Inactive'}")
        
        logger.info("\n🔧 Service Integrations:")
        logger.info(f"  AL-ULS (Symbolic):          {'✅ Available' if ALULS_AVAILABLE else '❌ Unavailable'}")
        logger.info(f"  Entropy Engine:             {'✅ Available' if ENTROPY_AVAILABLE else '❌ Unavailable'}")
        logger.info(f"  Quantum Cognitive:          {'✅ Available' if QUANTUM_AVAILABLE else '❌ Unavailable'}")
        logger.info("")
    
    async def process_complete_workflow(
        self,
        user_query: str,
        context: Optional[str] = None,
        resources: Optional[List[str]] = None,
        enable_vector_index: bool = True,
        enable_graph: bool = True,
        enable_entropy: bool = True
    ) -> Dict[str, Any]:
        """
        Execute complete integrated workflow across all systems
        
        Args:
            user_query: User's query
            context: Additional context
            resources: Resource texts or paths
            enable_vector_index: Use vector indexing
            enable_graph: Use graph operations
            enable_entropy: Use entropy analysis
        
        Returns:
            Complete workflow results
        """
        logger.info("\n" + "=" * 70)
        logger.info("COMPLETE WORKFLOW EXECUTION")
        logger.info("=" * 70)
        logger.info(f"Query: {user_query}")
        
        results = {
            "query": user_query,
            "stages": {},
            "final_output": None,
            "system_state": {}
        }
        
        resources = resources or []
        
        # Stage 1: Entropy Analysis
        if enable_entropy and (ENTROPY_AVAILABLE or LIMP_ENTROPY_AVAILABLE):
            logger.info("\n--- Stage 1: Entropy Analysis ---")
            try:
                if ENTROPY_AVAILABLE:
                    token_entropy = entropy_engine.score_token(user_query)
                    volatility = entropy_engine.get_volatility_signal(user_query)
                    self.state.entropy_scores = {
                        "token_entropy": token_entropy,
                        "volatility": volatility
                    }
                    results["stages"]["entropy"] = self.state.entropy_scores
                    logger.info(f"✅ Entropy: {token_entropy:.3f}, Volatility: {volatility:.3f}")
            except Exception as e:
                logger.warning(f"⚠️  Entropy analysis failed: {e}")
        
        # Stage 2: Symbolic Evaluation (AL-ULS)
        if ALULS_AVAILABLE:
            logger.info("\n--- Stage 2: Symbolic Evaluation ---")
            try:
                if al_uls.is_symbolic_call(user_query):
                    call = al_uls.parse_symbolic_call(user_query)
                    symbolic_result = await al_uls.eval_symbolic_call_async(call)
                    self.state.symbolic_calls.append(symbolic_result)
                    results["stages"]["symbolic"] = symbolic_result
                    logger.info(f"✅ Symbolic evaluation complete")
            except Exception as e:
                logger.warning(f"⚠️  Symbolic evaluation failed: {e}")
        
        # Stage 3: Vector Index Operations
        if enable_vector_index and self.vector_index:
            logger.info("\n--- Stage 3: Vector Index Operations ---")
            try:
                # Add query to index for future reference
                await self.vector_index.add_entry(
                    f"query_{hash(user_query) % 10000}",
                    user_query,
                    {"type": "query", "context": context}
                )
                
                # Search for similar queries if we have entries
                if len(self.vector_index.entries) > 1:
                    similar = await self.vector_index.search(user_query, top_k=3)
                    results["stages"]["vector_search"] = {
                        "similar_count": len(similar),
                        "top_match": similar[0][0].text if similar else None
                    }
                    logger.info(f"✅ Found {len(similar)} similar entries")
            except Exception as e:
                logger.warning(f"⚠️  Vector indexing failed: {e}")
        
        # Stage 4: Knowledge Graph Operations
        if enable_graph and self.graph_store:
            logger.info("\n--- Stage 4: Knowledge Graph Operations ---")
            try:
                # Add query as graph node
                node_id = f"q_{hash(user_query) % 10000}"
                await self.graph_store.add_node(
                    node_id,
                    "Query",
                    user_query,
                    {"context": context}
                )
                
                # Find semantically similar nodes
                if len(self.graph_store.nodes) > 1:
                    similar_nodes = await self.graph_store.find_similar_nodes(user_query, top_k=3)
                    results["stages"]["graph"] = {
                        "node_id": node_id,
                        "similar_count": len(similar_nodes)
                    }
                    logger.info(f"✅ Added graph node, found {len(similar_nodes)} similar")
            except Exception as e:
                logger.warning(f"⚠️  Graph operations failed: {e}")
        
        # Stage 5: Quantum Processing (if available)
        if self.quantum_processor and QUANTUM_AVAILABLE:
            logger.info("\n--- Stage 5: Quantum Processing ---")
            try:
                # Convert query to quantum representation
                import torch
                query_vec = torch.randn(1, 16)  # Simple representation
                quantum_result = self.quantum_processor(query_vec)
                
                self.state.quantum_state = {
                    "entropy": float(quantum_result["quantum_entropy"]),
                    "coherence": float(quantum_result["quantum_coherence"])
                }
                results["stages"]["quantum"] = self.state.quantum_state
                logger.info(f"✅ Quantum processing complete")
            except Exception as e:
                logger.warning(f"⚠️  Quantum processing failed: {e}")
        
        # Stage 6: Unified Cognitive Processing
        if self.cognitive_orch:
            logger.info("\n--- Stage 6: Unified Cognitive Processing ---")
            try:
                cognitive_result = await self.cognitive_orch.process_cognitive_workflow(
                    user_query=user_query,
                    context=context,
                    inline_resources=resources
                )
                
                self.state.cognitive_results = cognitive_result
                results["stages"]["cognitive"] = {
                    "stages_completed": list(cognitive_result["stages"].keys()),
                    "total_time": cognitive_result["timing"]["total"]
                }
                results["final_output"] = cognitive_result.get("final_output", "No output")
                logger.info(f"✅ Cognitive processing complete")
            except Exception as e:
                logger.warning(f"⚠️  Cognitive processing failed: {e}")
                results["final_output"] = f"Error in cognitive processing: {e}"
        
        # Compile system state
        results["system_state"] = {
            "vector_index_entries": len(self.vector_index.entries) if self.vector_index else 0,
            "graph_nodes": len(self.graph_store.nodes) if self.graph_store else 0,
            "graph_edges": len(self.graph_store.edges) if self.graph_store else 0,
            "entropy_analyzed": bool(self.state.entropy_scores),
            "symbolic_calls": len(self.state.symbolic_calls),
            "quantum_processed": self.state.quantum_state is not None
        }
        
        logger.info("\n" + "=" * 70)
        logger.info("COMPLETE WORKFLOW FINISHED")
        logger.info("=" * 70)
        
        return results
    
    async def batch_process(
        self,
        queries: List[str],
        contexts: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of queries
            contexts: Optional list of contexts
        
        Returns:
            List of results
        """
        contexts = contexts or [None] * len(queries)
        results = []
        
        logger.info(f"\nProcessing {len(queries)} queries in batch...")
        
        for i, (query, context) in enumerate(zip(queries, contexts), 1):
            logger.info(f"\n--- Batch {i}/{len(queries)} ---")
            result = await self.process_complete_workflow(query, context)
            results.append(result)
        
        return results
    
    def get_complete_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all systems"""
        stats = {
            "cognitive": {},
            "vector_index": {},
            "graph": {},
            "entropy": self.state.entropy_scores,
            "symbolic": {"total_calls": len(self.state.symbolic_calls)},
            "quantum": self.state.quantum_state
        }
        
        if self.cognitive_orch:
            stats["cognitive"] = self.cognitive_orch.get_cognitive_metrics()
        
        if self.vector_index:
            stats["vector_index"] = self.vector_index.get_stats()
        
        if self.graph_store:
            stats["graph"] = self.graph_store.get_stats()
        
        return stats
    
    async def close_all(self):
        """Close all subsystems"""
        logger.info("\nClosing all subsystems...")
        
        if self.cognitive_orch:
            await self.cognitive_orch.close()
        
        if self.vector_index:
            await self.vector_index.close()
        
        if self.graph_store:
            await self.graph_store.close()
        
        logger.info("✅ All subsystems closed")


async def demo_complete_integration():
    """Comprehensive demonstration of complete system integration"""
    
    print("\n" + "=" * 70)
    print("COMPLETE SYSTEM INTEGRATION DEMO")
    print("LiMp + Numbskull - All Components")
    print("=" * 70)
    
    # Create complete system
    system = CompleteSystemIntegration()
    
    # Test queries
    test_queries = [
        {
            "query": "What is the relationship between entropy and information?",
            "context": "Focus on information theory and thermodynamics",
            "resources": ["Information theory connects entropy to data compression"]
        },
        {
            "query": "Explain machine learning fundamentals",
            "context": "Cover supervised, unsupervised, and reinforcement learning",
            "resources": ["ML uses statistical methods to learn from data"]
        }
    ]
    
    # Process queries
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST QUERY {i}/{len(test_queries)}")
        print(f"{'='*70}")
        
        result = await system.process_complete_workflow(
            user_query=test["query"],
            context=test["context"],
            resources=test["resources"]
        )
        
        print(f"\n--- Results ---")
        print(f"Stages completed: {list(result['stages'].keys())}")
        print(f"System state: {result['system_state']}")
        print(f"Output length: {len(result.get('final_output', ''))} chars")
    
    # Get comprehensive stats
    print(f"\n{'='*70}")
    print("COMPLETE SYSTEM STATISTICS")
    print(f"{'='*70}")
    stats = system.get_complete_stats()
    print(json.dumps(stats, indent=2, default=str))
    
    # Cleanup
    await system.close_all()
    
    print(f"\n{'='*70}")
    print("✅ COMPLETE DEMO FINISHED")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(demo_complete_integration())

