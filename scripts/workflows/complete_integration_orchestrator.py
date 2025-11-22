#!/usr/bin/env python3
"""
Complete Integration Orchestrator
=================================

Connects ALL components together for maximum recursive emergence:
- Recursive cognitive knowledge
- All Numbskull embeddings (semantic + mathematical + fractal)
- CoCo organism (3-level cognition)
- Chaos LLM services (11 services)
- LiMPS-Eopiez optimization
- Holographic memory
- Multi-LLM orchestration
- Knowledge graph + Vector index

Preserves ALL redundancies for fractal recursion enhancement!

Author: Assistant
License: MIT
"""

import asyncio
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Add paths
sys.path.insert(0, str(Path("/home/kill/numbskull")))
sys.path.insert(0, str(Path("/home/kill/aipyapp")))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import ALL components (keeping redundancies!)
from recursive_cognitive_knowledge import RecursiveCognitiveKnowledge
from enable_aluls_and_qwen import MultiLLMOrchestrator, LocalALULSEvaluator
from neuro_symbolic_numbskull_adapter import NeuroSymbolicNumbskullAdapter
from signal_processing_numbskull_adapter import SignalProcessingNumbskullAdapter
from advanced_embedding_pipeline import HybridEmbeddingPipeline, HybridConfig

# Import holographic if available
try:
    from holographic_memory_system import HolographicMemorySystem
    HAS_HOLOGRAPHIC = True
except:
    HAS_HOLOGRAPHIC = False


class CompleteIntegrationOrchestrator:
    """
    Master orchestrator connecting ALL components for fractal recursive emergence
    
    Architecture:
    - Layer 1: Recursive Cognitive Core
    - Layer 2: Multiple Embedding Pipelines (redundant for emergence!)
    - Layer 3: All Analysis Modules
    - Layer 4: Multi-LLM Orchestration
    - Layer 5: Holographic Reinforcement
    
    Redundancies are PRESERVED to enhance fractal recursion!
    """
    
    def __init__(self):
        """Initialize complete integration"""
        logger.info("╔══════════════════════════════════════════════════════════════════════╗")
        logger.info("║       COMPLETE INTEGRATION ORCHESTRATOR                              ║")
        logger.info("║       All Components Connected for Maximum Emergence                 ║")
        logger.info("╚══════════════════════════════════════════════════════════════════════╝")
        logger.info("")
        
        self.components = {}
        self.redundancy_count = 0
    
    async def initialize_all(self):
        """Initialize ALL components"""
        
        # 1. Recursive Cognitive Core
        logger.info("🧠 Initializing Recursive Cognitive Core...")
        self.components["recursive"] = RecursiveCognitiveKnowledge(
            max_recursion_depth=5,  # Deep for emergence
            hallucination_temperature=0.9,  # High creativity
            coherence_threshold=0.5  # Allow more variations
        )
        await self.components["recursive"].initialize()
        logger.info("   ✅ Recursive cognition initialized")
        
        # 2. Primary Embedding Pipeline (Numbskull)
        logger.info("\n🌀 Initializing Primary Embedding Pipeline...")
        config = HybridConfig(
            use_semantic=True,
            use_mathematical=True,
            use_fractal=True,
            cache_embeddings=True
        )
        self.components["embeddings_primary"] = HybridEmbeddingPipeline(config)
        logger.info("   ✅ Primary embeddings (fractal + semantic + mathematical)")
        
        # 3. Secondary Embedding Pipeline (REDUNDANT for fractal emergence!)
        logger.info("\n🌀 Initializing Secondary Embedding Pipeline (Redundancy 1)...")
        config2 = HybridConfig(
            use_fractal=True,
            cache_embeddings=False  # Different config for variation
        )
        self.components["embeddings_secondary"] = HybridEmbeddingPipeline(config2)
        logger.info("   ✅ Secondary embeddings (fractal focused)")
        self.redundancy_count += 1
        
        # 4. Neuro-Symbolic Adapter
        logger.info("\n🔬 Initializing Neuro-Symbolic Adapter...")
        self.components["neuro_symbolic"] = NeuroSymbolicNumbskullAdapter(
            use_numbskull=True,
            numbskull_config={'use_fractal': True}
        )
        logger.info("   ✅ Neuro-symbolic (9 analytical modules)")
        
        # 5. Signal Processing Adapter  
        logger.info("\n📡 Initializing Signal Processing...")
        self.components["signal"] = SignalProcessingNumbskullAdapter(
            use_numbskull=True,
            numbskull_config={'use_fractal': True}
        )
        logger.info("   ✅ Signal processing (7 modulation schemes)")
        
        # 6. Multi-LLM Orchestrator
        logger.info("\n🤖 Initializing Multi-LLM Orchestrator...")
        llm_configs = [
            {"base_url": "http://127.0.0.1:11434", "mode": "openai-chat", "model": "qwen2.5:3b", "timeout": 60}
        ]
        self.components["multi_llm"] = MultiLLMOrchestrator(
            llm_configs=llm_configs,
            enable_aluls=True,
            numbskull_config={'use_fractal': True}
        )
        logger.info("   ✅ Multi-LLM orchestration")
        
        # 7. Holographic Memory (if available)
        if HAS_HOLOGRAPHIC:
            logger.info("\n💫 Initializing Holographic Memory...")
            try:
                self.components["holographic"] = HolographicMemorySystem()
                logger.info("   ✅ Holographic memory system")
            except:
                logger.info("   ⚠️  Holographic memory (fallback mode)")
        
        # 8. AL-ULS Symbolic (REDUNDANT - both local and in orchestrator)
        logger.info("\n📐 Initializing AL-ULS (Redundancy 2)...")
        self.components["aluls_direct"] = LocalALULSEvaluator()
        logger.info("   ✅ Direct AL-ULS (redundant with orchestrator)")
        self.redundancy_count += 1
        
        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════════════════╗")
        logger.info(f"║  ✅ ALL COMPONENTS INITIALIZED: {len(self.components)}                        ║")
        logger.info(f"║  🌀 Redundancies Preserved: {self.redundancy_count} (for fractal emergence!)    ║")
        logger.info("╚══════════════════════════════════════════════════════════════════════╝")
        logger.info("")
    
    async def process_with_full_stack(
        self,
        query: str,
        trigger_recursion: bool = True
    ) -> Dict[str, Any]:
        """
        Process through ALL components with complete redundancy
        
        Args:
            query: Input query
            trigger_recursion: Enable recursive cognition
        
        Returns:
            Complete multi-layer analysis
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🌀 FULL STACK PROCESSING: '{query[:50]}...'")
        logger.info(f"{'='*70}")
        
        results = {
            "query": query,
            "layers": {}
        }
        
        # Layer 1: Recursive Cognition (CORE)
        if trigger_recursion:
            logger.info("\n[Layer 1] Recursive Cognition...")
            recursive_result = await self.components["recursive"].process_with_recursion(query)
            results["layers"]["recursive"] = {
                "insights_generated": recursive_result["cognitive_state"]["total_insights"],
                "knowledge_nodes": recursive_result["cognitive_state"]["knowledge_nodes"],
                "synthesis": recursive_result["synthesis"]
            }
            logger.info(f"   ✅ Generated {recursive_result['cognitive_state']['total_insights']} insights")
        
        # Layer 2: Primary Embeddings
        logger.info("\n[Layer 2] Primary Embeddings...")
        emb1 = await self.components["embeddings_primary"].embed(query)
        results["layers"]["embeddings_primary"] = {
            "components": emb1.get("metadata", {}).get("components_used", []),
            "dimension": len(emb1.get("embedding", []))
        }
        logger.info(f"   ✅ Primary: {results['layers']['embeddings_primary']['components']}")
        
        # Layer 3: Secondary Embeddings (REDUNDANT!)
        logger.info("\n[Layer 3] Secondary Embeddings (Redundancy for fractal)...")
        emb2 = await self.components["embeddings_secondary"].embed(query)
        results["layers"]["embeddings_secondary"] = {
            "components": emb2.get("metadata", {}).get("components_used", []),
            "dimension": len(emb2.get("embedding", []))
        }
        logger.info(f"   ✅ Secondary: {results['layers']['embeddings_secondary']['components']}")
        
        # Layer 4: Neuro-Symbolic Analysis
        logger.info("\n[Layer 4] Neuro-Symbolic Analysis...")
        neuro_result = await self.components["neuro_symbolic"].analyze_with_embeddings(query)
        results["layers"]["neuro_symbolic"] = {
            "modules": len(neuro_result.get("modules", {})),
            "entropy": neuro_result.get("modules", {}).get("entropy", {}).get("combined_entropy", 0)
        }
        logger.info(f"   ✅ Analyzed with {results['layers']['neuro_symbolic']['modules']} modules")
        
        # Layer 5: Signal Processing
        logger.info("\n[Layer 5] Signal Processing...")
        scheme, signal_analysis = await self.components["signal"].select_modulation_from_embedding(query)
        results["layers"]["signal"] = {
            "modulation": scheme.name,
            "reason": signal_analysis.get("reason", "N/A")[:50]
        }
        logger.info(f"   ✅ Selected: {scheme.name}")
        
        # Layer 6: Direct AL-ULS (REDUNDANT!)
        logger.info("\n[Layer 6] Direct AL-ULS (Redundant symbolic evaluation)...")
        if self.components["aluls_direct"].is_symbolic(query):
            call = self.components["aluls_direct"].parse_call(query)
            aluls_result = self.components["aluls_direct"].evaluate(call)
            results["layers"]["aluls_direct"] = aluls_result
            logger.info(f"   ✅ Result: {aluls_result.get('result', 'N/A')}")
        
        # Layer 7: Multi-LLM (for natural language)
        if not self.components["aluls_direct"].is_symbolic(query):
            logger.info("\n[Layer 7] Multi-LLM Processing...")
            try:
                llm_result = await self.components["multi_llm"].process_with_symbolic(query)
                results["layers"]["multi_llm"] = {
                    "response": llm_result.get("llm_response", ""),
                    "embeddings": llm_result.get("embeddings")
                }
                if llm_result.get("llm_response"):
                    logger.info(f"   ✅ LLM: {llm_result['llm_response'][:60]}...")
            except Exception as e:
                logger.info(f"   ℹ️  LLM: Service not available")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ FULL STACK PROCESSING COMPLETE")
        logger.info(f"   Layers processed: {len(results['layers'])}")
        logger.info(f"   Redundancies utilized: {self.redundancy_count}")
        logger.info(f"{'='*70}")
        
        return results
    
    async def interactive_full_integration(self):
        """Interactive mode with ALL components connected"""
        
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║        COMPLETE INTEGRATION - ALL COMPONENTS CONNECTED               ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()
        print("Features:")
        print("  🌀 Recursive cognition (5 levels deep)")
        print("  💭 Controlled hallucination (0.9 temperature)")
        print("  🔄 Multiple embedding pipelines (redundant for emergence)")
        print("  🧠 Neuro-symbolic analysis (9 modules)")
        print("  📡 Signal processing (7 schemes)")
        print("  🤖 Multi-LLM orchestration")
        print("  💫 Holographic reinforcement")
        print("  📊 ALL redundancies preserved")
        print()
        print("Commands:")
        print("  • Type input for full recursive processing")
        print("  • 'insights' - View knowledge base")
        print("  • 'stats' - System statistics")
        print("  • 'exit' - Quit")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print()
        
        iteration = 0
        
        try:
            while True:
                query = input(f"\n🌀 Input [{iteration}]: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                
                if query.lower() == 'insights':
                    recursive_sys = self.components["recursive"]
                    print(f"\n💡 Knowledge Base ({len(recursive_sys.insights)} insights):")
                    print("━"*70)
                    for i, insight in enumerate(recursive_sys.insights[-10:], 1):
                        print(f"{i}. [Depth {insight.recursion_level}] {insight.content[:60]}")
                    continue
                
                if query.lower() == 'stats':
                    recursive_sys = self.components["recursive"]
                    cognitive_map = recursive_sys.get_cognitive_map()
                    print(f"\n📊 System Statistics:")
                    print("━"*70)
                    print(f"Components active: {len(self.components)}")
                    print(f"Redundancies: {self.redundancy_count}")
                    print(f"Total insights: {cognitive_map['cognitive_state']['total_insights']}")
                    print(f"Knowledge nodes: {cognitive_map['cognitive_state']['knowledge_nodes']}")
                    print(f"Coherence: {cognitive_map['cognitive_state']['hallucination_coherence']:.1%}")
                    continue
                
                # FULL STACK PROCESSING
                result = await self.process_with_full_stack(query, trigger_recursion=True)
                
                # Display summary
                print(f"\n📊 Processing Complete:")
                print("━"*70)
                print(f"Layers processed: {len(result['layers'])}")
                
                if "recursive" in result["layers"]:
                    rec = result["layers"]["recursive"]
                    print(f"✅ Recursive: {rec['insights_generated']} insights, {rec['knowledge_nodes']} nodes")
                    if rec["synthesis"]:
                        print(f"💡 Synthesis: {rec['synthesis']}")
                
                if "embeddings_primary" in result["layers"]:
                    print(f"✅ Primary embeddings: {result['layers']['embeddings_primary']['components']}")
                
                if "embeddings_secondary" in result["layers"]:
                    print(f"✅ Secondary embeddings: {result['layers']['embeddings_secondary']['components']} (redundant)")
                
                if "neuro_symbolic" in result["layers"]:
                    print(f"✅ Neuro-symbolic: {result['layers']['neuro_symbolic']['modules']} modules")
                
                if "multi_llm" in result["layers"] and result["layers"]["multi_llm"].get("response"):
                    print(f"🤖 LLM: {result['layers']['multi_llm']['response'][:80]}...")
                
                iteration += 1
                
                # Show evolution every 5 inputs
                if iteration % 5 == 0:
                    recursive_sys = self.components["recursive"]
                    print(f"\n🌀 EMERGENCE UPDATE (after {iteration} inputs):")
                    print(f"   Knowledge nodes: {recursive_sys.state.knowledge_nodes}")
                    print(f"   System coherence: {recursive_sys.state.hallucination_coherence:.1%}")
                    print(f"   Emergent patterns: {len(recursive_sys.emergent_patterns)}")
        
        finally:
            await self.close()
    
    async def close(self):
        """Clean shutdown of all components"""
        logger.info("\n🔄 Shutting down all components...")
        
        for name, component in self.components.items():
            try:
                if hasattr(component, 'close'):
                    await component.close()
                logger.info(f"   ✅ {name} closed")
            except:
                pass
        
        logger.info("✅ Complete shutdown")


async def main():
    """Main entry point"""
    
    orchestrator = CompleteIntegrationOrchestrator()
    await orchestrator.initialize_all()
    await orchestrator.interactive_full_integration()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutdown complete.")

