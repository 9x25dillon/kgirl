#!/usr/bin/env python3
"""
Unified Cognitive Orchestrator: Numbskull + LiMp Integration
=============================================================

Comprehensive integration bringing together:
- Numbskull: Hybrid embeddings (semantic, mathematical, fractal)
- LiMp TA ULS: Transformer with KFP layers and stability
- LiMp Neuro-Symbolic: 9 analytical modules
- LiMp Holographic Memory: Advanced memory storage
- LFM2-8B-A1B: Local LLM inference
- Signal Processing: Advanced modulation and processing

This creates a complete cognitive architecture for AI workflows.

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

# Add numbskull to path
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

# Numbskull imports
try:
    from advanced_embedding_pipeline import (
        HybridEmbeddingPipeline,
        HybridConfig
    )
    NUMBSKULL_AVAILABLE = True
except ImportError as e:
    NUMBSKULL_AVAILABLE = False
    logging.warning(f"Numbskull not available: {e}")

# LiMp imports
from numbskull_dual_orchestrator import (
    create_numbskull_orchestrator,
    NumbskullDualOrchestrator
)

try:
    from neuro_symbolic_engine import (
        EntropyAnalyzer,
        DianneReflector,
        MatrixTransformer,
        JuliaSymbolEngine,
        ChoppyProcessor,
        NeuroSymbolicEngine
    )
    NEUROSYMBOLIC_AVAILABLE = True
except ImportError:
    NEUROSYMBOLIC_AVAILABLE = False
    logging.warning("Neuro-symbolic engine not available")

try:
    from holographic_memory_system import (
        HolographicAssociativeMemory,
        FractalEncoder,
        QuantumEnhancedMemory
    )
    HOLOGRAPHIC_AVAILABLE = True
except ImportError:
    HOLOGRAPHIC_AVAILABLE = False
    logging.warning("Holographic memory not available")

try:
    import torch
    from tauls_transformer import (
        TAULSControlUnit,
        KFPLayer,
        EntropyRegulator
    )
    TAULS_AVAILABLE = True
except ImportError:
    TAULS_AVAILABLE = False
    logging.warning("TA ULS transformer not available")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CognitiveState:
    """State of the unified cognitive system"""
    embeddings: Optional[Dict[str, Any]] = None
    neuro_symbolic_analysis: Optional[Dict[str, Any]] = None
    holographic_traces: List[str] = field(default_factory=list)
    tauls_control: Optional[Dict[str, Any]] = None
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    cognitive_metrics: Dict[str, float] = field(default_factory=dict)


class UnifiedCognitiveOrchestrator:
    """
    Master orchestrator integrating Numbskull + LiMp modules
    
    Provides a complete cognitive workflow:
    1. Input → Numbskull embeddings (semantic, math, fractal)
    2. Embeddings → Neuro-symbolic analysis (9 modules)
    3. Analysis → Holographic memory storage
    4. Memory + Context → TA ULS transformation
    5. Transformed → LFM2-8B-A1B inference
    6. Output + learning feedback
    """
    
    def __init__(
        self,
        local_llm_config: Dict[str, Any],
        remote_llm_config: Optional[Dict[str, Any]] = None,
        numbskull_config: Optional[Dict[str, Any]] = None,
        enable_tauls: bool = True,
        enable_neurosymbolic: bool = True,
        enable_holographic: bool = True
    ):
        """
        Initialize the unified cognitive orchestrator
        
        Args:
            local_llm_config: Configuration for LFM2-8B-A1B
            remote_llm_config: Optional remote LLM for summarization
            numbskull_config: Configuration for embedding pipeline
            enable_tauls: Enable TA ULS transformer
            enable_neurosymbolic: Enable neuro-symbolic analysis
            enable_holographic: Enable holographic memory
        """
        logger.info("=" * 70)
        logger.info("INITIALIZING UNIFIED COGNITIVE ORCHESTRATOR")
        logger.info("=" * 70)
        
        self.cognitive_state = CognitiveState()
        
        # 1. Numbskull + Dual LLM Orchestration
        logger.info("1. Initializing Numbskull + Dual LLM...")
        if NUMBSKULL_AVAILABLE:
            settings = {
                "use_numbskull": True,
                "use_semantic": numbskull_config.get("use_semantic", False),
                "use_mathematical": numbskull_config.get("use_mathematical", False),
                "use_fractal": numbskull_config.get("use_fractal", True),
                "fusion_method": numbskull_config.get("fusion_method", "weighted_average"),
                "embedding_enhancement": "metadata",
                "temperature": 0.7,
                "max_tokens": 512
            }
            
            self.orchestrator = create_numbskull_orchestrator(
                local_configs=[local_llm_config],
                remote_config=remote_llm_config,
                settings=settings,
                numbskull_config=numbskull_config
            )
            logger.info("   ✅ Numbskull + Dual LLM ready")
        else:
            self.orchestrator = None
            logger.warning("   ⚠️  Numbskull not available")
        
        # 2. Neuro-Symbolic Engine
        logger.info("2. Initializing Neuro-Symbolic Engine...")
        if NEUROSYMBOLIC_AVAILABLE and enable_neurosymbolic:
            try:
                self.neuro_symbolic = NeuroSymbolicEngine()
                logger.info("   ✅ Neuro-symbolic engine ready (9 modules)")
            except Exception as e:
                self.neuro_symbolic = None
                logger.warning(f"   ⚠️  Neuro-symbolic init failed: {e}")
        else:
            self.neuro_symbolic = None
            logger.warning("   ⚠️  Neuro-symbolic not available")
        
        # 3. Holographic Memory
        logger.info("3. Initializing Holographic Memory...")
        if HOLOGRAPHIC_AVAILABLE and enable_holographic:
            try:
                self.holographic_memory = HolographicAssociativeMemory(
                    memory_size=1024,
                    hologram_dim=256
                )
                logger.info("   ✅ Holographic memory ready")
            except Exception as e:
                self.holographic_memory = None
                logger.warning(f"   ⚠️  Holographic memory init failed: {e}")
        else:
            self.holographic_memory = None
            logger.warning("   ⚠️  Holographic memory not available")
        
        # 4. TA ULS Transformer
        logger.info("4. Initializing TA ULS Transformer...")
        if TAULS_AVAILABLE and enable_tauls:
            try:
                self.tauls_unit = TAULSControlUnit(
                    input_dim=768,  # Match embedding dimension
                    hidden_dim=512,
                    control_dim=256
                )
                logger.info("   ✅ TA ULS transformer ready")
            except Exception as e:
                self.tauls_unit = None
                logger.warning(f"   ⚠️  TA ULS init failed: {e}")
        else:
            self.tauls_unit = None
            logger.warning("   ⚠️  TA ULS not available")
        
        logger.info("=" * 70)
        logger.info("UNIFIED COGNITIVE ORCHESTRATOR READY")
        logger.info("=" * 70)
        self._print_system_status()
    
    def _print_system_status(self):
        """Print status of all integrated systems"""
        logger.info("\nSystem Components Status:")
        logger.info(f"  Numbskull Embeddings:    {'✅ Active' if self.orchestrator else '❌ Inactive'}")
        logger.info(f"  Neuro-Symbolic Engine:   {'✅ Active' if self.neuro_symbolic else '❌ Inactive'}")
        logger.info(f"  Holographic Memory:      {'✅ Active' if self.holographic_memory else '❌ Inactive'}")
        logger.info(f"  TA ULS Transformer:      {'✅ Active' if self.tauls_unit else '❌ Inactive'}")
        logger.info("")
    
    async def process_cognitive_workflow(
        self,
        user_query: str,
        context: Optional[str] = None,
        resource_paths: List[str] = None,
        inline_resources: List[str] = None
    ) -> Dict[str, Any]:
        """
        Complete cognitive processing workflow
        
        Args:
            user_query: User's query or task
            context: Additional context
            resource_paths: Paths to resource files
            inline_resources: Inline resource strings
        
        Returns:
            Complete cognitive processing results
        """
        resource_paths = resource_paths or []
        inline_resources = inline_resources or []
        
        logger.info("\n" + "=" * 70)
        logger.info("STARTING COGNITIVE WORKFLOW")
        logger.info("=" * 70)
        logger.info(f"Query: {user_query}")
        
        start_time = time.time()
        workflow_results = {
            "query": user_query,
            "context": context,
            "stages": {},
            "final_output": None,
            "cognitive_state": {},
            "timing": {}
        }
        
        # Stage 1: Numbskull Embeddings
        logger.info("\n--- Stage 1: Numbskull Embedding Generation ---")
        stage_start = time.time()
        
        if self.orchestrator:
            try:
                # Generate embeddings for query + context
                combined_text = f"{user_query}\n{context if context else ''}"
                embedding_result = await self.orchestrator._generate_embeddings(combined_text)
                
                self.cognitive_state.embeddings = embedding_result
                workflow_results["stages"]["embeddings"] = {
                    "components": embedding_result["metadata"]["components_used"],
                    "dimension": embedding_result["metadata"]["embedding_dim"],
                    "processing_time": embedding_result["metadata"]["processing_time"]
                }
                logger.info(f"✅ Embeddings generated: {embedding_result['metadata']['components_used']}")
            except Exception as e:
                logger.warning(f"⚠️  Embedding generation failed: {e}")
                workflow_results["stages"]["embeddings"] = {"error": str(e)}
        else:
            logger.warning("⚠️  Numbskull not available, skipping embeddings")
        
        workflow_results["timing"]["embeddings"] = time.time() - stage_start
        
        # Stage 2: Neuro-Symbolic Analysis
        logger.info("\n--- Stage 2: Neuro-Symbolic Analysis ---")
        stage_start = time.time()
        
        if self.neuro_symbolic:
            try:
                analysis_input = {
                    "text": user_query,
                    "embeddings": self.cognitive_state.embeddings,
                    "context": context
                }
                
                neuro_analysis = await self.neuro_symbolic.analyze_async(analysis_input)
                self.cognitive_state.neuro_symbolic_analysis = neuro_analysis
                workflow_results["stages"]["neuro_symbolic"] = {
                    "modules_activated": len(neuro_analysis.get("modules", [])),
                    "insights": neuro_analysis.get("insights", [])[:3],  # Top 3
                    "complexity": neuro_analysis.get("complexity_score", 0)
                }
                logger.info(f"✅ Neuro-symbolic analysis complete")
            except Exception as e:
                logger.warning(f"⚠️  Neuro-symbolic analysis failed: {e}")
                workflow_results["stages"]["neuro_symbolic"] = {"error": str(e)}
        else:
            logger.warning("⚠️  Neuro-symbolic engine not available")
        
        workflow_results["timing"]["neuro_symbolic"] = time.time() - stage_start
        
        # Stage 3: Holographic Memory Storage
        logger.info("\n--- Stage 3: Holographic Memory Storage ---")
        stage_start = time.time()
        
        if self.holographic_memory and self.cognitive_state.embeddings:
            try:
                # Store embeddings in holographic memory
                embedding_vector = self.cognitive_state.embeddings["fused_embedding"]
                
                if isinstance(embedding_vector, np.ndarray):
                    memory_key = self.holographic_memory.store(
                        embedding_vector,
                        metadata={
                            "query": user_query,
                            "timestamp": time.time(),
                            "emotional_valence": 0.5,
                            "cognitive_significance": 0.8
                        }
                    )
                    
                    self.cognitive_state.holographic_traces.append(memory_key)
                    workflow_results["stages"]["holographic_memory"] = {
                        "memory_key": memory_key,
                        "stored": True
                    }
                    logger.info(f"✅ Stored in holographic memory: {memory_key}")
            except Exception as e:
                logger.warning(f"⚠️  Holographic storage failed: {e}")
                workflow_results["stages"]["holographic_memory"] = {"error": str(e)}
        else:
            logger.warning("⚠️  Holographic memory not available")
        
        workflow_results["timing"]["holographic_memory"] = time.time() - stage_start
        
        # Stage 4: TA ULS Transformation
        logger.info("\n--- Stage 4: TA ULS Transformation ---")
        stage_start = time.time()
        
        if self.tauls_unit and self.cognitive_state.embeddings:
            try:
                # Convert embedding to torch tensor
                embedding_vector = self.cognitive_state.embeddings["fused_embedding"]
                
                if isinstance(embedding_vector, np.ndarray):
                    # Ensure correct dimension (768)
                    if len(embedding_vector) < 768:
                        embedding_vector = np.pad(
                            embedding_vector,
                            (0, 768 - len(embedding_vector)),
                            mode='constant'
                        )
                    elif len(embedding_vector) > 768:
                        embedding_vector = embedding_vector[:768]
                    
                    tensor_input = torch.from_numpy(embedding_vector).float().unsqueeze(0)
                    
                    # Apply TA ULS transformation
                    with torch.no_grad():
                        control_output, stability_metrics = self.tauls_unit(
                            tensor_input,
                            torch.zeros(1, 256)  # Initial control state
                        )
                    
                    self.cognitive_state.tauls_control = {
                        "transformed": control_output.numpy(),
                        "stability": stability_metrics
                    }
                    
                    workflow_results["stages"]["tauls"] = {
                        "transformed": True,
                        "stability_score": float(stability_metrics.mean()) if torch.is_tensor(stability_metrics) else 0.0
                    }
                    logger.info(f"✅ TA ULS transformation applied")
            except Exception as e:
                logger.warning(f"⚠️  TA ULS transformation failed: {e}")
                workflow_results["stages"]["tauls"] = {"error": str(e)}
        else:
            logger.warning("⚠️  TA ULS not available")
        
        workflow_results["timing"]["tauls"] = time.time() - stage_start
        
        # Stage 5: LFM2-8B-A1B Inference
        logger.info("\n--- Stage 5: LFM2-8B-A1B Final Inference ---")
        stage_start = time.time()
        
        if self.orchestrator:
            try:
                # Run full orchestration with enriched context
                result = await self.orchestrator.run_with_embeddings(
                    user_prompt=user_query,
                    resource_paths=resource_paths,
                    inline_resources=inline_resources + ([context] if context else [])
                )
                
                workflow_results["stages"]["llm_inference"] = {
                    "summary_length": len(result.get("summary", "")),
                    "answer_length": len(result.get("final", "")),
                    "embedding_enhanced": result.get("embedding_result") is not None
                }
                workflow_results["final_output"] = result.get("final", "")
                logger.info(f"✅ LFM2 inference complete ({len(result.get('final', ''))} chars)")
            except Exception as e:
                logger.warning(f"⚠️  LFM2 inference failed: {e}")
                workflow_results["stages"]["llm_inference"] = {"error": str(e)}
                workflow_results["final_output"] = f"Error: {e}"
        else:
            logger.warning("⚠️  LLM orchestrator not available")
            workflow_results["final_output"] = "No LLM available for inference"
        
        workflow_results["timing"]["llm_inference"] = time.time() - stage_start
        
        # Complete workflow
        total_time = time.time() - start_time
        workflow_results["timing"]["total"] = total_time
        workflow_results["cognitive_state"] = {
            "embeddings_generated": self.cognitive_state.embeddings is not None,
            "neuro_analysis_complete": self.cognitive_state.neuro_symbolic_analysis is not None,
            "holographic_traces": len(self.cognitive_state.holographic_traces),
            "tauls_applied": self.cognitive_state.tauls_control is not None
        }
        
        logger.info("\n" + "=" * 70)
        logger.info(f"COGNITIVE WORKFLOW COMPLETE ({total_time:.2f}s)")
        logger.info("=" * 70)
        
        return workflow_results
    
    def get_cognitive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from the cognitive system"""
        metrics = {
            "embedding_stats": {},
            "memory_stats": {},
            "system_health": {}
        }
        
        # Embedding stats
        if self.orchestrator:
            metrics["embedding_stats"] = self.orchestrator.get_embedding_stats()
        
        # Memory stats
        if self.holographic_memory:
            metrics["memory_stats"] = {
                "total_traces": len(self.holographic_memory.memory_traces),
                "memory_size": self.holographic_memory.memory_size,
                "hologram_dim": self.holographic_memory.hologram_dim
            }
        
        # System health
        metrics["system_health"] = {
            "numbskull": self.orchestrator is not None,
            "neuro_symbolic": self.neuro_symbolic is not None,
            "holographic": self.holographic_memory is not None,
            "tauls": self.tauls_unit is not None
        }
        
        return metrics
    
    async def close(self):
        """Clean up resources"""
        if self.orchestrator:
            await self.orchestrator.close()
        logger.info("✅ Unified cognitive orchestrator closed")


async def demo_unified_system():
    """Demonstration of the unified cognitive system"""
    
    print("\n" + "=" * 70)
    print("UNIFIED COGNITIVE ORCHESTRATOR DEMO")
    print("Numbskull + LiMp Full Integration")
    print("=" * 70)
    
    # Configuration
    local_llm_config = {
        "base_url": "http://127.0.0.1:8080",
        "mode": "llama-cpp",
        "model": "LFM2-8B-A1B",
        "timeout": 120
    }
    
    numbskull_config = {
        "use_semantic": False,  # Set to True if Eopiez available
        "use_mathematical": False,  # Set to True if LIMPS available
        "use_fractal": True,  # Always available
        "fusion_method": "weighted_average"
    }
    
    # Create orchestrator
    orchestrator = UnifiedCognitiveOrchestrator(
        local_llm_config=local_llm_config,
        numbskull_config=numbskull_config,
        enable_tauls=TAULS_AVAILABLE,
        enable_neurosymbolic=NEUROSYMBOLIC_AVAILABLE,
        enable_holographic=HOLOGRAPHIC_AVAILABLE
    )
    
    # Test queries
    test_queries = [
        {
            "query": "Explain the concept of quantum entanglement",
            "context": "Focus on practical applications and experimental verification"
        },
        {
            "query": "Analyze the efficiency of different sorting algorithms",
            "context": "Consider time complexity, space complexity, and practical use cases"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST QUERY {i}")
        print(f"{'=' * 70}")
        
        result = await orchestrator.process_cognitive_workflow(
            user_query=test["query"],
            context=test["context"]
        )
        
        print(f"\n--- Results ---")
        print(f"Stages completed: {list(result['stages'].keys())}")
        print(f"Total time: {result['timing']['total']:.2f}s")
        print(f"Final output length: {len(result.get('final_output', ''))} chars")
    
    # Get metrics
    print(f"\n{'=' * 70}")
    print("SYSTEM METRICS")
    print(f"{'=' * 70}")
    metrics = orchestrator.get_cognitive_metrics()
    print(json.dumps(metrics, indent=2))
    
    # Cleanup
    await orchestrator.close()
    
    print(f"\n{'=' * 70}")
    print("✅ DEMO COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    if not NUMBSKULL_AVAILABLE:
        print("❌ Numbskull not available. Please install numbskull package.")
        sys.exit(1)
    
    asyncio.run(demo_unified_system())

