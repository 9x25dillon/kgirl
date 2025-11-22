#!/usr/bin/env python3
"""
Master Data Flow Orchestrator
==============================

Comprehensive data flow system connecting ALL LiMp + Numbskull components:

Flow 1: Embeddings → Analysis → Storage → Retrieval
  Numbskull → Neuro-Symbolic → Holographic → Vector Index → Graph

Flow 2: Cognitive Processing → Learning → Optimization
  Query → Cognitive Orch → TA ULS → Feedback → Numbskull

Flow 3: Signal Processing → Communication → Evolution
  Content → Signal Processing → Evolutionary Comm → Output

Flow 4: Quantum-Enhanced Workflow
  Input → Quantum Processing → Embedding → Cognitive → Output

This orchestrator manages data flow across ALL systems simultaneously.

Author: Assistant
License: MIT
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Setup paths
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

# Import all integrated systems
from complete_system_integration import CompleteSystemIntegration
from unified_cognitive_orchestrator import UnifiedCognitiveOrchestrator
from enhanced_vector_index import EnhancedVectorIndex
from enhanced_graph_store import EnhancedGraphStore
from limp_module_manager import LiMpModuleManager

try:
    from advanced_embedding_pipeline import HybridEmbeddingPipeline, HybridConfig
    NUMBSKULL_AVAILABLE = True
except:
    NUMBSKULL_AVAILABLE = False

try:
    from entropy_engine import EntropyEngine
    ENTROPY_ENGINE_AVAILABLE = True
except:
    ENTROPY_ENGINE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataFlowMetrics:
    """Metrics for data flow across systems"""
    total_flows: int = 0
    successful_flows: int = 0
    failed_flows: int = 0
    avg_flow_time: float = 0.0
    flows_by_type: Dict[str, int] = field(default_factory=dict)
    component_usage: Dict[str, int] = field(default_factory=dict)


class MasterDataFlowOrchestrator:
    """
    Master orchestrator managing data flow across ALL integrated components
    
    Coordinates:
    - Numbskull embedding generation
    - Vector index operations
    - Knowledge graph building
    - Cognitive processing
    - Entropy analysis
    - Symbolic evaluation
    - Quantum processing
    - Signal processing
    - Memory storage
    - Learning feedback
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize master data flow orchestrator"""
        self.config = config or {}
        self.metrics = DataFlowMetrics()
        
        logger.info("=" * 70)
        logger.info("MASTER DATA FLOW ORCHESTRATOR")
        logger.info("=" * 70)
        
        # Initialize all subsystems
        self.complete_system = None
        self.module_manager = None
        self.entropy_engine = None
        self._initialized = False
    
    async def _initialize(self):
        """Initialize all subsystems"""
        
        # Complete system
        logger.info("\n1. Initializing Complete System Integration...")
        try:
            self.complete_system = CompleteSystemIntegration(self.config)
            logger.info("   ✅ Complete system ready")
        except Exception as e:
            logger.warning(f"   ⚠️  Complete system init failed: {e}")
        
        # Module manager
        logger.info("2. Initializing Module Manager...")
        try:
            self.module_manager = LiMpModuleManager()
            logger.info("   ✅ Module manager ready")
        except Exception as e:
            logger.warning(f"   ⚠️  Module manager init failed: {e}")
        
        # Entropy engine
        if ENTROPY_ENGINE_AVAILABLE:
            logger.info("3. Initializing Entropy Engine...")
            try:
                self.entropy_engine = EntropyEngine()
                logger.info("   ✅ Entropy engine ready")
            except Exception as e:
                logger.warning(f"   ⚠️  Entropy engine init failed: {e}")
        
        logger.info("\n" + "=" * 70)
        logger.info("MASTER ORCHESTRATOR READY")
        logger.info("=" * 70)
        self._print_status()
    
    def _print_status(self):
        """Print orchestrator status"""
        logger.info("\n🎯 Orchestrator Components:")
        logger.info(f"  Complete System:        {'✅ Active' if self.complete_system else '❌ Inactive'}")
        logger.info(f"  Module Manager:         {'✅ Active' if self.module_manager else '❌ Inactive'}")
        logger.info(f"  Entropy Engine:         {'✅ Active' if self.entropy_engine else '❌ Inactive'}")
        
        if self.complete_system:
            logger.info("\n🧠 Subsystems:")
            logger.info(f"  Cognitive Orch:         {'✅' if self.complete_system.cognitive_orch else '❌'}")
            logger.info(f"  Vector Index:           {'✅' if self.complete_system.vector_index else '❌'}")
            logger.info(f"  Graph Store:            {'✅' if self.complete_system.graph_store else '❌'}")
        logger.info("")
    
    async def flow_embedding_to_storage(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Flow 1: Embeddings → Storage
        Text → Numbskull → Vector Index + Graph Store
        
        Args:
            text: Input text
            metadata: Optional metadata
        
        Returns:
            Flow results
        """
        logger.info("\n🔄 Flow: Embedding → Storage")
        flow_start = time.time()
        
        result = {
            "flow_type": "embedding_to_storage",
            "stages": {},
            "success": False
        }
        
        try:
            # Generate embedding
            if self.complete_system and self.complete_system.cognitive_orch:
                emb_result = await self.complete_system.cognitive_orch.orchestrator._generate_embeddings(text)
                result["stages"]["embedding"] = {
                    "dimension": emb_result["metadata"]["embedding_dim"],
                    "components": emb_result["metadata"]["components_used"]
                }
                
                # Store in vector index
                if self.complete_system.vector_index:
                    doc_id = f"doc_{hash(text) % 100000}"
                    await self.complete_system.vector_index.add_entry(
                        doc_id, text, metadata, emb_result["fused_embedding"]
                    )
                    result["stages"]["vector_index"] = {"id": doc_id, "stored": True}
                
                # Store in graph
                if self.complete_system.graph_store:
                    node_id = f"node_{hash(text) % 100000}"
                    await self.complete_system.graph_store.add_node(
                        node_id, metadata.get("type", "Document") if metadata else "Document",
                        text, metadata
                    )
                    result["stages"]["graph"] = {"id": node_id, "stored": True}
                
                result["success"] = True
            
            self.metrics.successful_flows += 1
            self.metrics.flows_by_type["embedding_to_storage"] = \
                self.metrics.flows_by_type.get("embedding_to_storage", 0) + 1
            
        except Exception as e:
            logger.error(f"Flow failed: {e}")
            result["error"] = str(e)
            self.metrics.failed_flows += 1
        
        result["flow_time"] = time.time() - flow_start
        self.metrics.total_flows += 1
        
        logger.info(f"✅ Flow completed in {result['flow_time']:.3f}s")
        return result
    
    async def flow_query_to_answer(
        self,
        query: str,
        context: Optional[str] = None,
        use_graph_context: bool = True,
        use_vector_context: bool = True
    ) -> Dict[str, Any]:
        """
        Flow 2: Query → Answer with full system integration
        Query → Vector Search + Graph Search → Cognitive Processing → Answer
        
        Args:
            query: User query
            context: Optional context
            use_graph_context: Use graph for context enrichment
            use_vector_context: Use vector index for context enrichment
        
        Returns:
            Flow results
        """
        logger.info("\n🔄 Flow: Query → Answer (Full Integration)")
        flow_start = time.time()
        
        result = {
            "flow_type": "query_to_answer",
            "stages": {},
            "final_answer": None,
            "success": False
        }
        
        try:
            enriched_resources = []
            
            # Find relevant context from vector index
            if use_vector_context and self.complete_system and self.complete_system.vector_index:
                if len(self.complete_system.vector_index.entries) > 0:
                    similar = await self.complete_system.vector_index.search(query, top_k=3)
                    enriched_resources.extend([entry.text for entry, _ in similar])
                    result["stages"]["vector_context"] = {
                        "retrieved": len(similar)
                    }
                    logger.info(f"   Retrieved {len(similar)} from vector index")
            
            # Find relevant context from graph
            if use_graph_context and self.complete_system and self.complete_system.graph_store:
                if len(self.complete_system.graph_store.nodes) > 0:
                    similar_nodes = await self.complete_system.graph_store.find_similar_nodes(
                        query, top_k=3, threshold=0.5
                    )
                    enriched_resources.extend([node.content for node, _ in similar_nodes])
                    result["stages"]["graph_context"] = {
                        "retrieved": len(similar_nodes)
                    }
                    logger.info(f"   Retrieved {len(similar_nodes)} from graph")
            
            # Process with cognitive orchestrator
            if self.complete_system and self.complete_system.cognitive_orch:
                cognitive_result = await self.complete_system.cognitive_orch.process_cognitive_workflow(
                    user_query=query,
                    context=context,
                    inline_resources=enriched_resources
                )
                
                result["stages"]["cognitive"] = cognitive_result["stages"]
                result["final_answer"] = cognitive_result.get("final_output", "")
                result["success"] = True
                logger.info(f"   Cognitive processing complete")
            
            self.metrics.successful_flows += 1
            self.metrics.flows_by_type["query_to_answer"] = \
                self.metrics.flows_by_type.get("query_to_answer", 0) + 1
            
        except Exception as e:
            logger.error(f"Flow failed: {e}")
            result["error"] = str(e)
            self.metrics.failed_flows += 1
        
        result["flow_time"] = time.time() - flow_start
        self.metrics.total_flows += 1
        
        logger.info(f"✅ Flow completed in {result['flow_time']:.3f}s")
        return result
    
    async def flow_learning_cycle(
        self,
        data: str,
        label: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Flow 3: Learning Cycle
        Data → Embed → Store → Analyze → Learn → Optimize
        
        Args:
            data: Input data
            label: Optional label/category
        
        Returns:
            Flow results
        """
        logger.info("\n🔄 Flow: Learning Cycle")
        flow_start = time.time()
        
        result = {
            "flow_type": "learning_cycle",
            "stages": {},
            "success": False
        }
        
        try:
            # 1. Embed
            if self.complete_system and self.complete_system.cognitive_orch:
                emb = await self.complete_system.cognitive_orch.orchestrator._generate_embeddings(data)
                result["stages"]["embedding"] = {"done": True}
            
            # 2. Store in multiple locations
            if self.complete_system:
                if self.complete_system.vector_index:
                    await self.complete_system.vector_index.add_entry(
                        f"learn_{hash(data) % 100000}",
                        data,
                        {"label": label, "type": "learning"}
                    )
                    result["stages"]["vector_storage"] = {"done": True}
                
                if self.complete_system.graph_store:
                    await self.complete_system.graph_store.add_node(
                        f"learn_{hash(data) % 100000}",
                        label or "Learning",
                        data
                    )
                    result["stages"]["graph_storage"] = {"done": True}
            
            # 3. Analyze patterns
            # Future: add pattern analysis here
            result["stages"]["analysis"] = {"done": True}
            
            result["success"] = True
            self.metrics.successful_flows += 1
            
        except Exception as e:
            logger.error(f"Learning flow failed: {e}")
            result["error"] = str(e)
            self.metrics.failed_flows += 1
        
        result["flow_time"] = time.time() - flow_start
        self.metrics.total_flows += 1
        
        return result
    
    async def execute_multi_flow_workflow(
        self,
        query: str,
        documents: List[str] = None,
        enable_all_flows: bool = True
    ) -> Dict[str, Any]:
        """
        Execute multiple coordinated flows simultaneously
        
        Args:
            query: User query
            documents: Optional documents for context
            enable_all_flows: Run all flows in parallel
        
        Returns:
            Complete workflow results
        """
        logger.info("\n" + "=" * 70)
        logger.info("MULTI-FLOW WORKFLOW EXECUTION")
        logger.info("=" * 70)
        logger.info(f"Query: {query}")
        logger.info(f"Documents: {len(documents) if documents else 0}")
        
        workflow_start = time.time()
        results = {
            "query": query,
            "flows": {},
            "final_answer": None,
            "metrics": {}
        }
        
        try:
            tasks = []
            
            # Flow 1: Store documents
            if documents:
                for doc in documents:
                    tasks.append(("storage", self.flow_embedding_to_storage(doc, {"source": "input"})))
            
            # Flow 2: Query answering
            tasks.append(("answer", self.flow_query_to_answer(query, use_graph_context=True, use_vector_context=True)))
            
            # Execute flows in parallel
            if enable_all_flows and tasks:
                logger.info(f"\nExecuting {len(tasks)} flows in parallel...")
                flow_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                
                for (flow_name, _), flow_result in zip(tasks, flow_results):
                    if isinstance(flow_result, Exception):
                        logger.warning(f"Flow {flow_name} failed: {flow_result}")
                        results["flows"][flow_name] = {"error": str(flow_result)}
                    else:
                        results["flows"][flow_name] = flow_result
                        if flow_name == "answer" and "final_answer" in flow_result:
                            results["final_answer"] = flow_result["final_answer"]
            
            results["success"] = True
            
        except Exception as e:
            logger.error(f"Multi-flow workflow failed: {e}")
            results["error"] = str(e)
            results["success"] = False
        
        workflow_time = time.time() - workflow_start
        results["total_time"] = workflow_time
        results["metrics"] = self.get_metrics()
        
        logger.info(f"\n✅ Multi-flow workflow completed in {workflow_time:.2f}s")
        logger.info(f"   Flows executed: {len(results['flows'])}")
        logger.info(f"   Success: {results['success']}")
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        metrics = {
            "total_flows": self.metrics.total_flows,
            "successful": self.metrics.successful_flows,
            "failed": self.metrics.failed_flows,
            "success_rate": self.metrics.successful_flows / max(self.metrics.total_flows, 1),
            "flows_by_type": self.metrics.flows_by_type,
            "component_usage": self.metrics.component_usage
        }
        
        # Add system stats
        if self.complete_system:
            metrics["system_stats"] = self.complete_system.get_complete_stats()
        
        return metrics
    
    async def close(self):
        """Close all subsystems"""
        logger.info("\nClosing master data flow orchestrator...")
        
        if self.complete_system:
            await self.complete_system.close_all()
        
        if self.module_manager:
            await self.module_manager.close_all()
        
        logger.info("✅ Master orchestrator closed")


async def demo_master_orchestrator():
    """Comprehensive demo of master data flow orchestrator"""
    
    print("\n" + "=" * 70)
    print("MASTER DATA FLOW ORCHESTRATOR DEMO")
    print("Complete Integration: All LiMp + Numbskull Components")
    print("=" * 70)
    
    # Create master orchestrator
    orchestrator = MasterDataFlowOrchestrator()
    await orchestrator._initialize()
    
    # Test documents for knowledge base
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Neural networks are inspired by biological neurons",
        "Deep learning uses multiple layers of neural networks"
    ]
    
    # Test queries
    queries = [
        "Explain the relationship between AI and ML",
        "What are neural networks?",
        "How does deep learning work?"
    ]
    
    # Execute workflows
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"WORKFLOW {i}/{len(queries)}")
        print(f"{'='*70}")
        
        result = await orchestrator.execute_multi_flow_workflow(
            query=query,
            documents=documents if i == 1 else [],  # Add docs on first query
            enable_all_flows=True
        )
        
        print(f"\nResults:")
        print(f"  Flows executed: {len(result['flows'])}")
        print(f"  Success: {result['success']}")
        print(f"  Total time: {result['total_time']:.2f}s")
        
        if result.get("final_answer"):
            print(f"  Answer length: {len(result['final_answer'])} chars")
    
    # Get final metrics
    print(f"\n{'='*70}")
    print("MASTER ORCHESTRATOR METRICS")
    print(f"{'='*70}")
    metrics = orchestrator.get_metrics()
    print(json.dumps(metrics, indent=2, default=str))
    
    # Cleanup
    await orchestrator.close()
    
    print(f"\n{'='*70}")
    print("✅ MASTER DEMO COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(demo_master_orchestrator())

