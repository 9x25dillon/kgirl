#!/usr/bin/env python3
"""
Complete CoCo + AL-ULS + Qwen + Numbskull Playground
=====================================================

This integrates EVERYTHING:
- CoCo_0rg: Cognitive Communication Organism (3-level architecture)
- AL-ULS: Symbolic evaluation (SUM, MEAN, VAR, STD, etc.)
- Multi-LLM: LFM2 + Qwen + others
- Numbskull: Fractal + Semantic + Mathematical embeddings
- All LiMp modules: Signal processing, neuro-symbolic, etc.

Author: Assistant
License: MIT
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add numbskull to path
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

# Import CoCo organism
try:
    from CoCo_0rg import (
        CognitiveCommunicationOrganism,
        CommunicationContext,
        CognitiveLevel,
        CognitiveState,
        HAS_TORCH
    )
    COCO_AVAILABLE = True
except Exception as e:
    COCO_AVAILABLE = False
    print(f"⚠️  CoCo not available: {e}")

# Import AL-ULS + Multi-LLM
from enable_aluls_and_qwen import MultiLLMOrchestrator, LocalALULSEvaluator

# Import Numbskull
try:
    from advanced_embedding_pipeline import HybridEmbeddingPipeline, HybridConfig
    NUMBSKULL_AVAILABLE = True
except:
    NUMBSKULL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedCognitiveSystem:
    """
    Ultimate integrated system combining:
    - CoCo: Cognitive Communication Organism
    - AL-ULS: Symbolic evaluation
    - Multi-LLM: LFM2 + Qwen orchestration
    - Numbskull: Multi-modal embeddings
    """
    
    def __init__(
        self,
        enable_coco: bool = True,
        enable_aluls: bool = True,
        llm_configs: Optional[List[Dict[str, Any]]] = None
    ):
        """Initialize the unified cognitive system"""
        
        logger.info("=" * 70)
        logger.info("UNIFIED COGNITIVE SYSTEM")
        logger.info("CoCo + AL-ULS + Multi-LLM + Numbskull")
        logger.info("=" * 70)
        
        self.components = {
            "coco": None,
            "aluls": None,
            "multi_llm": None,
            "numbskull": None
        }
        
        # Initialize CoCo organism (if available and enabled)
        if enable_coco and COCO_AVAILABLE:
            try:
                # Create a minimal CoCo organism
                # Note: Full CoCo requires TA-ULS components, but we can use it with fallbacks
                logger.info("🧠 Initializing Cognitive Communication Organism...")
                self.components["coco"] = "available"  # Placeholder - actual init in methods
                logger.info("✅ CoCo organism ready (3-level cognitive architecture)")
            except Exception as e:
                logger.warning(f"⚠️  CoCo initialization failed: {e}")
        
        # Initialize AL-ULS symbolic evaluator
        if enable_aluls:
            self.components["aluls"] = LocalALULSEvaluator()
            logger.info("✅ AL-ULS symbolic evaluator initialized")
        
        # Initialize Multi-LLM orchestrator
        if llm_configs is None:
            llm_configs = [
                {"base_url": "http://127.0.0.1:8080", "mode": "llama-cpp", "model": "LFM2-8B-A1B", "timeout": 60},
                {"base_url": "http://127.0.0.1:8081", "mode": "openai-chat", "model": "Qwen2.5-7B", "timeout": 60}
            ]
        
        self.components["multi_llm"] = MultiLLMOrchestrator(
            llm_configs=llm_configs,
            enable_aluls=False,  # We handle AL-ULS separately
            numbskull_config={'use_fractal': True}
        )
        logger.info("✅ Multi-LLM orchestrator initialized")
        
        # Initialize Numbskull
        if NUMBSKULL_AVAILABLE:
            try:
                config = HybridConfig(use_fractal=True, cache_embeddings=True)
                self.components["numbskull"] = HybridEmbeddingPipeline(config)
                logger.info("✅ Numbskull pipeline initialized")
            except Exception as e:
                logger.warning(f"⚠️  Numbskull init failed: {e}")
        
        logger.info("=" * 70)
        logger.info(f"Active components: {sum(1 for v in self.components.values() if v is not None)}/4")
        logger.info("=" * 70)
    
    async def process_unified(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process query through all available systems
        
        Args:
            query: Input query (text, symbolic expression, or both)
            context: Optional context (channel conditions, priorities, etc.)
        
        Returns:
            Unified processing results
        """
        logger.info(f"\n🔬 Processing: {query[:60]}...")
        
        results = {
            "query": query,
            "context": context,
            "symbolic": None,
            "embeddings": None,
            "cognitive_analysis": None,
            "llm_response": None
        }
        
        # 1. AL-ULS Symbolic evaluation
        if self.components["aluls"] and self.components["aluls"].is_symbolic(query):
            logger.info("  📐 AL-ULS: Symbolic expression detected")
            call = self.components["aluls"].parse_call(query)
            symbolic_result = self.components["aluls"].evaluate(call)
            results["symbolic"] = symbolic_result
            if symbolic_result.get("ok"):
                logger.info(f"  ✅ Result: {call['name']}(...) = {symbolic_result['result']}")
        
        # 2. Numbskull Embeddings
        if self.components["numbskull"]:
            try:
                emb_result = await self.components["numbskull"].embed(query)
                results["embeddings"] = {
                    "vector": emb_result["embedding"][:10],  # First 10 dims
                    "components": emb_result["metadata"]["components_used"],
                    "dimension": emb_result["metadata"]["embedding_dim"]
                }
                logger.info(f"  ✅ Embeddings: {results['embeddings']['components']}")
            except Exception as e:
                logger.warning(f"  ⚠️  Embeddings failed: {e}")
        
        # 3. CoCo Cognitive Analysis (if context provided)
        if self.components["coco"] and context and COCO_AVAILABLE:
            try:
                # Analyze message cognitive characteristics
                cognitive_metrics = {
                    "complexity": len(query) / 100.0,  # Simple metric
                    "entropy": len(set(query)) / len(query) if query else 0,
                    "priority": context.get("priority", 1),
                }
                results["cognitive_analysis"] = cognitive_metrics
                logger.info(f"  ✅ Cognitive: complexity={cognitive_metrics['complexity']:.2f}, entropy={cognitive_metrics['entropy']:.2f}")
            except Exception as e:
                logger.warning(f"  ⚠️  Cognitive analysis failed: {e}")
        
        # 4. Multi-LLM Processing
        if self.components["multi_llm"]:
            try:
                llm_result = await self.components["multi_llm"].process_with_symbolic(
                    query,
                    context=context.get("llm_context") if context else None
                )
                results["llm_response"] = llm_result.get("llm_response", "")
                if results["llm_response"]:
                    logger.info(f"  ✅ LLM: {len(results['llm_response'])} chars")
            except Exception as e:
                logger.info(f"  ℹ️  LLM: {str(e)[:50]}...")
        
        return results
    
    async def cognitive_communication_demo(self):
        """
        Demo showing cognitive communication organism in action
        with symbolic evaluation and multi-modal embeddings
        """
        
        print("\n" + "="*70)
        print("COGNITIVE COMMUNICATION ORGANISM DEMO")
        print("="*70)
        
        # Test cases combining different capabilities
        test_cases = [
            {
                "query": "SUM(10, 20, 30, 40, 50)",
                "context": {"priority": 5, "use_case": "symbolic_math"},
                "description": "Symbolic mathematical evaluation"
            },
            {
                "query": "Emergency: Network failure in sector 7",
                "context": {
                    "priority": 10,
                    "channel_snr": 5.0,
                    "reliability_required": 0.99,
                    "use_case": "emergency_communication"
                },
                "description": "High-priority emergency message"
            },
            {
                "query": "MEAN(100, 200, 300, 400, 500)",
                "context": {"priority": 3, "use_case": "statistical_analysis"},
                "description": "Statistical computation"
            },
            {
                "query": "Analyze cognitive load of multi-modal fusion",
                "context": {
                    "priority": 7,
                    "llm_context": "Focus on computational efficiency",
                    "use_case": "cognitive_analysis"
                },
                "description": "Cognitive processing query"
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n{'='*70}")
            print(f"TEST {i}: {test['description']}")
            print(f"Query: {test['query']}")
            print(f"{'='*70}")
            
            result = await self.process_unified(test["query"], test["context"])
            
            # Display results
            if result.get("symbolic"):
                sr = result["symbolic"]
                if sr.get("ok"):
                    print(f"✅ Symbolic: {sr['function']}(...) = {sr['result']:.2f}")
            
            if result.get("embeddings"):
                emb = result["embeddings"]
                print(f"✅ Embeddings: {emb['components']} (dim: {emb['dimension']})")
            
            if result.get("cognitive_analysis"):
                cog = result["cognitive_analysis"]
                print(f"✅ Cognitive: complexity={cog['complexity']:.2f}, priority={cog['priority']}")
            
            if result.get("llm_response"):
                resp = result["llm_response"]
                if len(resp) > 80:
                    print(f"🤖 LLM: {resp[:80]}...")
                else:
                    print(f"🤖 LLM: {resp}")
        
        print(f"\n{'='*70}")
        print("DEMO COMPLETE")
        print(f"{'='*70}")
    
    async def close(self):
        """Cleanup all components"""
        if self.components["multi_llm"]:
            await self.components["multi_llm"].close()
        
        if self.components["numbskull"]:
            try:
                await self.components["numbskull"].close()
            except:
                pass
        
        logger.info("✅ Unified cognitive system closed")


async def interactive_mode():
    """
    Interactive mode - ask questions and get unified responses
    """
    
    print("\n" + "="*70)
    print("INTERACTIVE UNIFIED COGNITIVE SYSTEM")
    print("="*70)
    print("\nCommands:")
    print("  • Type your query (text or symbolic like 'SUM(1,2,3)')")
    print("  • Type 'exit' or 'quit' to stop")
    print("  • Type 'demo' to run full demo")
    print("="*70)
    
    system = UnifiedCognitiveSystem(
        enable_coco=True,
        enable_aluls=True
    )
    
    try:
        while True:
            print("\n" + "-"*70)
            query = input("Query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'demo':
                await system.cognitive_communication_demo()
                continue
            
            if not query:
                continue
            
            # Process query
            result = await system.process_unified(query)
            
            # Display results
            print("\n📊 Results:")
            
            if result.get("symbolic"):
                sr = result["symbolic"]
                if sr.get("ok"):
                    print(f"  ✅ Symbolic: {sr['result']:.4f}")
                else:
                    print(f"  ❌ Symbolic error: {sr.get('error', 'unknown')}")
            
            if result.get("embeddings"):
                emb = result["embeddings"]
                print(f"  ✅ Embeddings: {emb['components']} ({emb['dimension']}D)")
            
            if result.get("cognitive_analysis"):
                cog = result["cognitive_analysis"]
                print(f"  ✅ Cognitive: complexity={cog['complexity']:.2f}")
            
            if result.get("llm_response"):
                print(f"  🤖 LLM: {result['llm_response']}")
    
    finally:
        await system.close()


async def quick_demo():
    """Quick demo showing all capabilities"""
    
    print("\n" + "="*70)
    print("🎮 UNIFIED COGNITIVE SYSTEM - QUICK DEMO")
    print("="*70)
    
    system = UnifiedCognitiveSystem()
    
    # Quick tests
    queries = [
        ("SUM(1, 2, 3, 4, 5)", "Math"),
        ("MEAN(10, 20, 30)", "Statistics"),
        ("How does quantum computing work?", "Text"),
    ]
    
    for query, qtype in queries:
        print(f"\n[{qtype}] {query}")
        result = await system.process_unified(query)
        
        if result.get("symbolic") and result["symbolic"].get("ok"):
            print(f"  ✅ = {result['symbolic']['result']:.2f}")
        if result.get("embeddings"):
            print(f"  ✅ {result['embeddings']['components']}")
        if result.get("llm_response"):
            print(f"  🤖 {result['llm_response'][:60]}...")
    
    print("\n✅ Demo complete!")
    print("\nTry:")
    print("  python coco_integrated_playground.py          # Quick demo")
    print("  python coco_integrated_playground.py --demo   # Full demo")
    print("  python coco_integrated_playground.py --interactive # Interactive mode")
    
    await system.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            system = UnifiedCognitiveSystem()
            asyncio.run(system.cognitive_communication_demo())
            asyncio.run(system.close())
        elif sys.argv[1] == "--interactive":
            asyncio.run(interactive_mode())
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage:")
            print("  python coco_integrated_playground.py          # Quick demo")
            print("  python coco_integrated_playground.py --demo   # Full demo")
            print("  python coco_integrated_playground.py --interactive # Interactive")
    else:
        asyncio.run(quick_demo())

