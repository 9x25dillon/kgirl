#!/usr/bin/env python3
"""
Complete aipyapp Integration Playground
=======================================

Interactive playground showcasing ALL integrated components from aipyapp:
- 11 Chaos LLM services (QGI, Entropy, Retrieval, etc.)
- LiMPS-Eopiez optimization system
- LLM training system
- BLOOM model backend
- Complete integration with existing LiMp components

Author: Assistant
License: MIT
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add paths
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

# Import integrated components
from chaos_llm_integration import ChaosLLMIntegration
from limps_eopiez_adapter import LiMPSEopiezAdapter
from llm_training_adapter import LLMTrainingAdapter
from bloom_backend import BLOOMBackend

# Import existing LiMp components
try:
    from enable_aluls_and_qwen import LocalALULSEvaluator
    from neuro_symbolic_numbskull_adapter import NeuroSymbolicNumbskullAdapter
    LIMP_AVAILABLE = True
except:
    LIMP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIPyAppPlayground:
    """
    Comprehensive playground for all aipyapp integrations
    
    Combines:
    - Chaos LLM services
    - LiMPS-Eopiez optimization
    - Training systems
    - BLOOM backend
    - Existing LiMp modules
    """
    
    def __init__(self):
        """Initialize the complete playground"""
        logger.info("="*70)
        logger.info("AIPYAPP COMPLETE INTEGRATION PLAYGROUND")
        logger.info("="*70)
        
        # Initialize all systems
        self.chaos = ChaosLLMIntegration()
        self.limps = LiMPSEopiezAdapter()
        self.training = LLMTrainingAdapter()
        self.bloom = BLOOMBackend()
        
        # Initialize LiMp components if available
        if LIMP_AVAILABLE:
            self.aluls = LocalALULSEvaluator()
            self.neuro = NeuroSymbolicNumbskullAdapter(use_numbskull=True)
            logger.info("✅ LiMp components integrated")
        else:
            self.aluls = None
            self.neuro = None
        
        logger.info("="*70)
        logger.info("READY! All systems initialized")
        logger.info("="*70)
    
    async def process_query(
        self,
        query: str,
        use_all_systems: bool = True
    ) -> Dict[str, Any]:
        """
        Process query through all available systems
        
        Args:
            query: Input query
            use_all_systems: Use all systems or just primary ones
        
        Returns:
            Complete processing results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {query}")
        logger.info(f"{'='*70}")
        
        results = {
            "query": query,
            "chaos_analysis": None,
            "limps_optimization": None,
            "aluls_symbolic": None,
            "neuro_symbolic": None
        }
        
        # 1. Chaos LLM comprehensive analysis
        if self.chaos.available:
            results["chaos_analysis"] = await self.chaos.comprehensive_analysis(query)
            logger.info("✅ Chaos LLM analysis complete")
        
        # 2. LiMPS-Eopiez optimization
        if self.limps.available and use_all_systems:
            results["limps_optimization"] = await self.limps.comprehensive_optimization(query)
            logger.info("✅ LiMPS-Eopiez optimization complete")
        
        # 3. AL-ULS symbolic evaluation
        if self.aluls and self.aluls.is_symbolic(query):
            call = self.aluls.parse_call(query)
            results["aluls_symbolic"] = self.aluls.evaluate(call)
            logger.info(f"✅ AL-ULS evaluation: {results['aluls_symbolic'].get('result')}")
        
        # 4. Neuro-symbolic analysis
        if self.neuro and use_all_systems:
            results["neuro_symbolic"] = await self.neuro.analyze_with_embeddings(query)
            logger.info("✅ Neuro-symbolic analysis complete")
        
        return results
    
    async def demo_chaos_services(self):
        """Demo Chaos LLM services"""
        print(f"\n{'='*70}")
        print("CHAOS LLM SERVICES DEMO")
        print(f"{'='*70}")
        
        queries = [
            "SUM(10, 20, 30, 40, 50)",
            "What is quantum computing?",
            "SELECT * FROM data WHERE value > 100"
        ]
        
        for query in queries:
            result = await self.chaos.comprehensive_analysis(query)
            
            print(f"\nQuery: {query}")
            if result.get("entropy"):
                print(f"  Entropy: {result['entropy']['entropy']:.3f}")
            if result.get("motifs"):
                print(f"  Motifs: {result['motifs']}")
            if result.get("symbolic"):
                print(f"  Symbolic: {result['symbolic']}")
    
    async def demo_limps_optimization(self):
        """Demo LiMPS-Eopiez optimization"""
        print(f"\n{'='*70}")
        print("LIMPS-EOPIEZ OPTIMIZATION DEMO")
        print(f"{'='*70}")
        
        text = "Advanced cognitive processing integrates multiple AI modalities"
        parameters = {
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        result = await self.limps.comprehensive_optimization(text, parameters)
        
        print(f"\nText: {text}")
        if result.get("linguistic"):
            ling = result["linguistic"]
            print(f"  Words: {ling.get('word_count')}, Richness: {ling.get('vocabulary_richness', 0):.2f}")
        if result.get("fractal"):
            print(f"  Fractal dimension: {result['fractal'].get('fractal_dimension', 0):.3f}")
    
    async def demo_training_system(self):
        """Demo LLM training system"""
        print(f"\n{'='*70}")
        print("LLM TRAINING SYSTEM DEMO")
        print(f"{'='*70}")
        
        # Resource estimation
        resources = await self.training.estimate_training_resources("7B")
        print(f"\n7B Model Resources:")
        print(f"  RAM: {resources['resources']['ram_gb']}GB")
        print(f"  Feasible: {resources['feasible']}")
        
        # Workflow creation
        workflow = await self.training.create_training_workflow(10000, epochs=3)
        print(f"\nWorkflow: {len(workflow['stages'])} stages")
        print(f"  Duration: {workflow['estimated_duration_hours']:.1f}h")
    
    async def demo_bloom_backend(self):
        """Demo BLOOM model backend"""
        print(f"\n{'='*70}")
        print("BLOOM MODEL BACKEND DEMO")
        print(f"{'='*70}")
        
        stats = self.bloom.get_stats()
        print(f"\nBLOOM Model:")
        print(f"  Available: {stats['model_available']}")
        print(f"  Files: {stats['model_files']}")
        print(f"  Path: {stats['model_path']}")
    
    async def demo_complete_integration(self):
        """Demo complete integration with all systems"""
        print(f"\n{'='*70}")
        print("COMPLETE INTEGRATION DEMO")
        print(f"{'='*70}")
        
        queries = [
            "SUM(100, 200, 300)",
            "Explain neural networks"
        ]
        
        for query in queries:
            result = await self.process_query(query, use_all_systems=True)
            
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print(f"{'='*70}")
            
            if result.get("aluls_symbolic") and result["aluls_symbolic"].get("ok"):
                print(f"✅ Symbolic: {result['aluls_symbolic']['result']}")
            
            if result.get("chaos_analysis"):
                chaos = result["chaos_analysis"]
                if chaos.get("entropy"):
                    print(f"✅ Entropy: {chaos['entropy']['entropy']:.3f}")
            
            if result.get("limps_optimization"):
                limps = result["limps_optimization"]
                if limps.get("linguistic"):
                    print(f"✅ Linguistic: {limps['linguistic'].get('word_count')} words")
    
    async def interactive_mode(self):
        """Interactive playground mode"""
        print(f"\n{'='*70}")
        print("AIPYAPP INTERACTIVE PLAYGROUND")
        print(f"{'='*70}")
        print("\nCommands:")
        print("  • Type your query (text or symbolic)")
        print("  • 'demo' - Run all demos")
        print("  • 'stats' - Show statistics")
        print("  • 'exit' - Quit")
        print(f"{'='*70}")
        
        while True:
            print(f"\n{'-'*70}")
            query = input("Query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'demo':
                await self.demo_complete_integration()
                continue
            
            if query.lower() == 'stats':
                self.show_stats()
                continue
            
            if not query:
                continue
            
            # Process query
            result = await self.process_query(query, use_all_systems=False)
            
            # Display results
            print("\n📊 Results:")
            
            if result.get("aluls_symbolic") and result["aluls_symbolic"].get("ok"):
                print(f"  ✅ Symbolic: {result['aluls_symbolic']['result']:.4f}")
            
            if result.get("chaos_analysis"):
                chaos = result["chaos_analysis"]
                if chaos.get("entropy"):
                    print(f"  ✅ Entropy: {chaos['entropy']['entropy']:.3f}")
                if chaos.get("motifs"):
                    print(f"  ✅ Motifs: {chaos['motifs']}")
    
    def show_stats(self):
        """Show system statistics"""
        print(f"\n{'='*70}")
        print("SYSTEM STATISTICS")
        print(f"{'='*70}")
        
        # Chaos stats
        if self.chaos.available:
            chaos_stats = self.chaos.get_stats()
            print("\nChaos LLM Services:")
            for key, value in chaos_stats.items():
                if key != "available":
                    print(f"  {key}: {value}")
        
        # BLOOM stats
        bloom_stats = self.bloom.get_stats()
        print("\nBLOOM Backend:")
        print(f"  Available: {bloom_stats['model_available']}")
        print(f"  Model files: {bloom_stats['model_files']}")
    
    async def close(self):
        """Cleanup all systems"""
        if self.chaos:
            await self.chaos.close()
        if self.limps:
            await self.limps.close()
        if self.training:
            await self.training.close()
        if self.neuro:
            await self.neuro.close()
        
        logger.info("✅ All systems closed")


async def main():
    """Main entry point"""
    import sys
    
    playground = AIPyAppPlayground()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--demo":
            await playground.demo_complete_integration()
        elif command == "--chaos":
            await playground.demo_chaos_services()
        elif command == "--limps":
            await playground.demo_limps_optimization()
        elif command == "--training":
            await playground.demo_training_system()
        elif command == "--bloom":
            await playground.demo_bloom_backend()
        elif command == "--interactive":
            await playground.interactive_mode()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python aipyapp_playground.py [--demo|--chaos|--limps|--training|--bloom|--interactive]")
    else:
        # Default: run complete demo
        await playground.demo_complete_integration()
    
    await playground.close()


if __name__ == "__main__":
    asyncio.run(main())

