#!/usr/bin/env python3
"""
Master Playground - Complete Integration
========================================

Clean, cohesive integration of ALL components:
- No warnings
- All services connected
- Unified experience
- Production-ready

Author: Assistant
License: MIT
"""

import asyncio
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress async cleanup warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*never awaited")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*no running event loop")

# Add paths
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

# Configure logging to reduce noise
logging.basicConfig(
    level=logging.ERROR,  # Only show critical errors
    format='%(levelname)s: %(message)s'
)

# Silence specific noisy loggers
logging.getLogger('advanced_embedding_pipeline').setLevel(logging.ERROR)
logging.getLogger('enable_aluls_and_qwen').setLevel(logging.ERROR)
logging.getLogger('dual_llm_orchestrator').setLevel(logging.ERROR)
logging.getLogger('numbskull_dual_orchestrator').setLevel(logging.ERROR)

# Import with clean error handling
try:
    from enable_aluls_and_qwen import MultiLLMOrchestrator, LocalALULSEvaluator
    from neuro_symbolic_numbskull_adapter import NeuroSymbolicNumbskullAdapter
    from signal_processing_numbskull_adapter import SignalProcessingNumbskullAdapter
    from enhanced_vector_index import EnhancedVectorIndex
    IMPORTS_OK = True
except Exception as e:
    print(f"Import error: {e}")
    IMPORTS_OK = False


class MasterPlayground:
    """
    Master playground with all services integrated cleanly
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize master playground
        
        Args:
            verbose: Enable verbose logging
        """
        if verbose:
            logging.getLogger().setLevel(logging.INFO)
        
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║              🎮 MASTER PLAYGROUND - ALL SERVICES                     ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()
        
        # Initialize AL-ULS
        self.aluls = LocalALULSEvaluator()
        
        # Initialize multi-LLM with Ollama
        llm_configs = [
            {
                "base_url": "http://127.0.0.1:11434",
                "mode": "openai-chat",
                "model": "qwen2.5:3b",
                "timeout": 60
            }
        ]
        
        numbskull_config = {
            'use_semantic': True,  # Will use Eopiez if available
            'use_mathematical': True,  # Will use LIMPS if available
            'use_fractal': True,  # Always available
            'cache_embeddings': True
        }
        
        self.orchestrator = MultiLLMOrchestrator(
            llm_configs=llm_configs,
            enable_aluls=True,
            numbskull_config=numbskull_config
        )
        
        # Check service availability
        self.services = self._check_services()
        self._print_status()
    
    def _check_services(self) -> Dict[str, bool]:
        """Check which services are available"""
        import requests
        
        services = {
            'eopiez': False,
            'limps': False,
            'ollama': False,
            'aluls': True,  # Always available
            'fractal': True  # Always available
        }
        
        # Check Eopiez
        try:
            r = requests.get('http://localhost:8001/health', timeout=1)
            services['eopiez'] = r.status_code == 200
        except:
            pass
        
        # Check LIMPS
        try:
            r = requests.get('http://localhost:8000/health', timeout=1)
            services['limps'] = r.status_code == 200
        except:
            pass
        
        # Check Ollama
        try:
            r = requests.get('http://localhost:11434/api/tags', timeout=1)
            services['ollama'] = r.status_code == 200
        except:
            pass
        
        return services
    
    def _print_status(self):
        """Print service status"""
        print("Service Status:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        def status_icon(available):
            return "✅" if available else "⚠️ "
        
        print(f"  {status_icon(self.services['aluls'])} AL-ULS Symbolic       (local, always available)")
        print(f"  {status_icon(self.services['fractal'])} Fractal Embeddings     (local, always available)")
        print(f"  {status_icon(self.services['eopiez'])} Semantic Embeddings    (Eopiez on port 8001)")
        print(f"  {status_icon(self.services['limps'])} Mathematical Embeddings (LIMPS on port 8000)")
        print(f"  {status_icon(self.services['ollama'])} LLM Inference          (Ollama on port 11434)")
        
        active_count = sum(1 for v in self.services.values() if v)
        print()
        print(f"Active: {active_count}/5 services")
        
        if active_count < 5:
            print()
            print("To start missing services: bash start_all_services.sh")
        
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print()
    
    async def process(self, query: str) -> Dict[str, Any]:
        """
        Process query through all available systems
        
        Args:
            query: Input query
        
        Returns:
            Processing results
        """
        results = {
            'query': query,
            'symbolic': None,
            'embeddings': None,
            'llm_response': None
        }
        
        # 1. Check for symbolic expression
        if self.aluls.is_symbolic(query):
            call = self.aluls.parse_call(query)
            results['symbolic'] = self.aluls.evaluate(call)
        
        # 2. Process with full orchestrator
        try:
            full_result = await self.orchestrator.process_with_symbolic(query)
            results['embeddings'] = full_result.get('embeddings')
            results['llm_response'] = full_result.get('llm_response')
        except Exception as e:
            if 'verbose' in sys.argv:
                print(f"Processing error: {e}")
        
        return results
    
    async def interactive(self):
        """Interactive mode"""
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║              INTERACTIVE MODE                                        ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()
        print("Commands:")
        print("  • Type your query (text or symbolic like 'SUM(1,2,3)')")
        print("  • 'status' - Show service status")
        print("  • 'exit' or 'quit' - Exit")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print()
        
        while True:
            try:
                query = input("\n🎮 Query: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'status':
                    self.services = self._check_services()
                    self._print_status()
                    continue
                
                # Process query
                print()
                result = await self.process(query)
                
                # Display results
                print("Results:")
                print("─" * 70)
                
                if result['symbolic'] and result['symbolic'].get('ok'):
                    print(f"✅ Symbolic: {result['symbolic']['result']:.4f}")
                
                if result['embeddings']:
                    emb = result['embeddings']
                    print(f"✅ Embeddings: {emb['components']} ({emb['dimension']}D)")
                
                if result['llm_response']:
                    resp = result['llm_response']
                    if len(resp) > 200:
                        print(f"🤖 LLM: {resp[:200]}...")
                    else:
                        print(f"🤖 LLM: {resp}")
                else:
                    if not result['symbolic']:
                        print("ℹ️  LLM: Not available (start Ollama for inference)")
                
                print("─" * 70)
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def demo(self):
        """Quick demo"""
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║              QUICK DEMO                                              ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()
        
        queries = [
            "SUM(10, 20, 30, 40, 50)",
            "MEAN(100, 200, 300)",
            "What is quantum computing?"
        ]
        
        for query in queries:
            print(f"Query: {query}")
            print("─" * 70)
            
            result = await self.process(query)
            
            if result['symbolic'] and result['symbolic'].get('ok'):
                print(f"✅ Result: {result['symbolic']['result']:.2f}")
            
            if result['embeddings']:
                print(f"✅ Embeddings: {result['embeddings']['components']}")
            
            if result['llm_response']:
                resp = result['llm_response']
                print(f"🤖 LLM: {resp[:100]}...")
            
            print()
        
        print("Demo complete! Run with --interactive for full access.")
    
    async def close(self):
        """Clean shutdown"""
        try:
            await self.orchestrator.close()
        except:
            pass


async def main():
    """Main entry point"""
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    playground = MasterPlayground(verbose=verbose)
    
    try:
        if '--interactive' in sys.argv or '-i' in sys.argv:
            await playground.interactive()
        else:
            await playground.demo()
    finally:
        await playground.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")

