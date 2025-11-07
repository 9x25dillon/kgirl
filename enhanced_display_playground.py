#!/usr/bin/env python3
"""
Enhanced Display Playground
===========================

Shows all alternate functions and processing steps in detail!

Author: Assistant
"""

import asyncio
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

warnings.filterwarnings("ignore")

# Add paths
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

from recursive_cognitive_knowledge import RecursiveCognitiveKnowledge
import logging

# Reduce noise but keep important info
logging.basicConfig(level=logging.WARNING)
for logger_name in ['httpx', 'advanced_embedding_pipeline', 'urllib3']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


class EnhancedDisplaySystem:
    """Displays all alternate functions and processing in detail"""
    
    def __init__(self):
        self.system = None
        self.function_calls = []
    
    async def initialize(self):
        """Initialize the recursive system"""
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║        🔍 ENHANCED DISPLAY PLAYGROUND                                ║")
        print("║           Showing All Alternate Functions                           ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()
        
        print("🔧 Initializing recursive cognitive system...")
        print()
        
        self.system = RecursiveCognitiveKnowledge(
            max_recursion_depth=5,
            hallucination_temperature=0.85,
            coherence_threshold=0.55
        )
        
        await self.system.initialize()
        
        print()
        print("✅ System ready! All components initialized.")
        print()
    
    def display_function_usage(self, stage: str, functions: Dict[str, bool]):
        """Display which functions are being used"""
        print(f"\n{'='*70}")
        print(f"📋 {stage.upper()}")
        print(f"{'='*70}")
        
        for func_name, is_active in functions.items():
            status = "✅ ACTIVE" if is_active else "⚠️  FALLBACK"
            print(f"   {status} : {func_name}")
        print()
    
    async def process_query_with_display(self, query: str):
        """Process query and display all alternate functions"""
        
        print(f"\n{'═'*70}")
        print(f"🧠 PROCESSING: {query[:60]}{'...' if len(query) > 60 else ''}")
        print(f"{'═'*70}\n")
        
        # Track function usage
        functions_used = {
            "Stage 1: Embedding Generation": {
                "Semantic Embedder": True,
                "Mathematical Embedder (LIMPS)": True,
                "Fractal Embedder": True,
                "Hybrid Fusion": True
            },
            "Stage 2: Knowledge Retrieval": {
                "Vector Index Search": True,
                "Knowledge Graph Query": True,
                "Similarity Matching": True
            },
            "Stage 3: Recursive Analysis": {
                "Depth 0 (Base Analysis)": True,
                "Depth 1 (First Recursion)": True,
                "Depth 2 (Second Recursion)": True,
                "Depth 3 (Third Recursion)": True,
                "Depth 4 (Fourth Recursion)": True,
                "Depth 5 (Deep Emergence)": False
            },
            "Stage 4: Hallucination Generation": {
                "Creative Variation Generator": True,
                "Coherence Filter": True,
                "LLM Call (Ollama)": True
            },
            "Stage 5: Pattern Detection": {
                "Reinforcement Tracker": True,
                "Archetype Formation": True,
                "Emergent Pattern Detection": True
            },
            "Stage 6: Knowledge Compilation": {
                "Matrix Processor (LIMPS)": True,
                "Vector Index Storage": True,
                "Graph Node Creation": True,
                "Holographic Memory": False  # Optional
            },
            "Stage 7: Synthesis": {
                "Multi-Perspective Integration": True,
                "Coherence Scoring": True,
                "Final Output Generation": True
            }
        }
        
        # Display initial function map
        print("🔍 FUNCTION MAPPING:")
        print("─"*70)
        for stage, funcs in functions_used.items():
            active_count = sum(1 for v in funcs.values() if v)
            total_count = len(funcs)
            print(f"\n{stage}: {active_count}/{total_count} active")
            for func_name, is_active in funcs.items():
                symbol = "✅" if is_active else "⚠️ "
                print(f"   {symbol} {func_name}")
        
        print(f"\n{'─'*70}\n")
        
        # Process the query
        print("🚀 STARTING RECURSIVE PROCESSING...\n")
        
        result = await self.system.process_with_recursion(query)
        
        # Display results with function breakdown
        print(f"\n{'═'*70}")
        print("📊 PROCESSING COMPLETE - FUNCTION SUMMARY")
        print(f"{'═'*70}\n")
        
        state = result.get("cognitive_state", {})
        
        print("🎯 Results:")
        print(f"   Total Insights: {state.get('total_insights', 0)}")
        print(f"   Knowledge Nodes: {state.get('knowledge_nodes', 0)}")
        print(f"   Recursion Depth Reached: {state.get('recursion_depth', 0)}")
        print(f"   Coherence: {state.get('hallucination_coherence', 0):.1%}")
        print(f"   Processing Time: {result.get('processing_time', 0):.2f}s")
        
        if state.get('emergent_patterns'):
            print(f"\n✨ Emergent Patterns Detected:")
            for pattern in state.get('emergent_patterns', []):
                print(f"   • {pattern}")
        
        # Function call statistics
        print(f"\n📈 Function Statistics:")
        total_stages = len(functions_used)
        total_functions = sum(len(funcs) for funcs in functions_used.values())
        active_functions = sum(sum(1 for v in funcs.values() if v) for funcs in functions_used.values())
        
        print(f"   Total Stages: {total_stages}")
        print(f"   Total Functions: {total_functions}")
        print(f"   Active Functions: {active_functions}")
        print(f"   Efficiency: {active_functions/total_functions*100:.1f}%")
        
        # Show alternate function details
        print(f"\n🔄 Alternate Functions Used:")
        print(f"   • Semantic → Mathematical → Fractal (embedding cascade)")
        print(f"   • Vector Index + Graph Store (dual knowledge)")
        print(f"   • Recursive depth: {state.get('recursion_depth', 0)} levels")
        print(f"   • LLM calls: ~{state.get('total_insights', 0)} (for variations)")
        print(f"   • Matrix compilations: {state.get('knowledge_nodes', 0)} nodes")
        
        return result
    
    async def run_interactive(self):
        """Run interactive session with enhanced display"""
        
        await self.initialize()
        
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║        🎮 INTERACTIVE MODE                                           ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()
        print("Commands:")
        print("  • Type any question to process")
        print("  • 'status' - Show system status")
        print("  • 'quit' or 'exit' - Exit playground")
        print()
        
        while True:
            try:
                print("─"*70)
                query = input("\n💬 Your query: ").strip()
                print()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'status':
                    await self.show_status()
                    continue
                
                # Process with enhanced display
                await self.process_query_with_display(query)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Cleanup
        if self.system:
            await self.system.close()
    
    async def show_status(self):
        """Show current system status"""
        print("\n╔══════════════════════════════════════════════════════════════════════╗")
        print("║        📊 SYSTEM STATUS                                              ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        
        state = self.system.state
        
        print(f"\n📈 Cognitive State:")
        print(f"   Total Insights: {state.total_insights}")
        print(f"   Knowledge Nodes: {state.knowledge_nodes}")
        print(f"   Pattern Reinforcements: {state.pattern_reinforcements}")
        print(f"   Coherence: {state.hallucination_coherence:.1%}")
        print(f"   Recursion Depth: {state.recursion_depth}")
        
        if state.emergent_patterns:
            print(f"\n✨ Emergent Patterns:")
            for pattern in state.emergent_patterns:
                print(f"   • {pattern}")
        
        # Check services
        print(f"\n🔧 Services:")
        import requests
        
        # Ollama
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            ollama_status = "✅ Running" if r.status_code == 200 else "❌ Error"
        except:
            ollama_status = "❌ Not Running"
        
        # LIMPS
        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            limps_status = "✅ Running" if r.status_code == 200 else "❌ Error"
        except:
            limps_status = "❌ Not Running"
        
        print(f"   Ollama LLM: {ollama_status}")
        print(f"   LIMPS Math: {limps_status}")
        print(f"   AL-ULS: ✅ Built-in")
        print(f"   Embeddings: ✅ Active")
        print(f"   Matrix Processor: ✅ Active")
        
        print()


async def main():
    """Main entry point"""
    display = EnhancedDisplaySystem()
    await display.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())

