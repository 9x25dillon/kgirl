#!/usr/bin/env python3
"""
Enable AL-ULS Symbolic + Qwen Integration
=========================================

This module:
1. Enables AL-ULS symbolic evaluation (local fallback if service unavailable)
2. Adds Qwen as an additional LLM option in dual orchestration
3. Creates a complete multi-LLM + symbolic evaluation system

Author: Assistant
License: MIT
"""

import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add numbskull to path
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

from numbskull_dual_orchestrator import create_numbskull_orchestrator
from advanced_embedding_pipeline import HybridEmbeddingPipeline, HybridConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalALULSEvaluator:
    """
    Local AL-ULS symbolic evaluator (works without service)
    
    Provides basic symbolic evaluation for common operations:
    - SUM, MEAN, VAR, STD
    - MIN, MAX
    - Simple mathematical expressions
    """
    
    def __init__(self):
        self.call_pattern = re.compile(r'([A-Z_]+)\((.*?)\)')
        logger.info("✅ Local AL-ULS evaluator initialized")
    
    def is_symbolic(self, text: str) -> bool:
        """Check if text is a symbolic call"""
        return bool(self.call_pattern.search(text))
    
    def parse_call(self, text: str) -> Dict[str, Any]:
        """Parse symbolic call"""
        match = self.call_pattern.search(text)
        if not match:
            return {"name": None, "args": []}
        
        name = match.group(1)
        args_str = match.group(2)
        args = [a.strip() for a in args_str.split(',') if a.strip()]
        
        return {"name": name, "args": args}
    
    def evaluate(self, call: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate symbolic call"""
        name = call.get("name", "")
        args_str = call.get("args", [])
        
        try:
            # Convert args to numbers
            args = [float(a) for a in args_str]
            
            # Evaluate based on function name
            if name == "SUM":
                result = sum(args)
            elif name == "MEAN":
                result = sum(args) / len(args) if args else 0
            elif name == "VAR":
                mean = sum(args) / len(args) if args else 0
                result = sum((x - mean)**2 for x in args) / len(args) if args else 0
            elif name == "STD":
                mean = sum(args) / len(args) if args else 0
                var = sum((x - mean)**2 for x in args) / len(args) if args else 0
                result = var ** 0.5
            elif name == "MIN":
                result = min(args) if args else 0
            elif name == "MAX":
                result = max(args) if args else 0
            elif name == "PROD":
                result = 1
                for a in args:
                    result *= a
            else:
                return {"ok": False, "error": f"Unknown function: {name}"}
            
            return {
                "ok": True,
                "result": result,
                "function": name,
                "args": args,
                "local_evaluation": True
            }
            
        except Exception as e:
            return {"ok": False, "error": str(e)}


class MultiLLMOrchestrator:
    """
    Extended orchestrator supporting multiple LLM backends:
    - LFM2-8B-A1B (local, primary)
    - Qwen (local/remote, fallback)
    - Any other OpenAI-compatible LLM
    
    With integrated AL-ULS symbolic evaluation
    """
    
    def __init__(
        self,
        llm_configs: List[Dict[str, Any]],
        enable_aluls: bool = True,
        numbskull_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-LLM orchestrator
        
        Args:
            llm_configs: List of LLM configurations (LFM2, Qwen, etc.)
            enable_aluls: Enable AL-ULS symbolic evaluation
            numbskull_config: Numbskull configuration
        """
        logger.info("=" * 70)
        logger.info("MULTI-LLM ORCHESTRATOR (LFM2 + Qwen + AL-ULS)")
        logger.info("=" * 70)
        
        # Create numbskull orchestrator with all LLMs
        settings = {
            'use_numbskull': True,
            'use_fractal': True,
            'temperature': 0.7,
            'max_tokens': 512
        }
        
        self.orchestrator = create_numbskull_orchestrator(
            local_configs=llm_configs,
            remote_config=None,
            settings=settings,
            numbskull_config=numbskull_config or {'use_fractal': True}
        )
        
        logger.info(f"✅ Multi-LLM orchestrator with {len(llm_configs)} backends")
        
        # AL-ULS evaluator
        self.aluls = None
        if enable_aluls:
            self.aluls = LocalALULSEvaluator()
            logger.info("✅ AL-ULS symbolic evaluator enabled")
        
        logger.info("=" * 70)
    
    async def process_with_symbolic(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process query with symbolic evaluation and multi-LLM
        
        Args:
            query: User query (may contain symbolic calls)
            context: Optional context
        
        Returns:
            Processing results
        """
        logger.info(f"\n🔬 Processing: {query[:60]}...")
        
        results = {
            "query": query,
            "symbolic_result": None,
            "embeddings": None,
            "llm_response": None
        }
        
        # Check for symbolic expressions
        if self.aluls and self.aluls.is_symbolic(query):
            logger.info("  📐 Symbolic expression detected")
            call = self.aluls.parse_call(query)
            symbolic_result = self.aluls.evaluate(call)
            results["symbolic_result"] = symbolic_result
            logger.info(f"  ✅ Evaluated: {call['name']}({','.join(call['args'])}) = {symbolic_result.get('result', 'error')}")
        
        # Generate embeddings
        try:
            emb = await self.orchestrator._generate_embeddings(query)
            results["embeddings"] = {
                "components": emb["metadata"]["components_used"],
                "dimension": emb["metadata"]["embedding_dim"]
            }
            logger.info(f"  ✅ Embeddings: {emb['metadata']['components_used']}")
        except Exception as e:
            logger.warning(f"  ⚠️  Embeddings failed: {e}")
        
        # Try LLM generation (will use fallback if server not available)
        if context or not results["symbolic_result"]:
            try:
                llm_result = await self.orchestrator.run_with_embeddings(
                    user_prompt=query,
                    resource_paths=[],
                    inline_resources=[context] if context else []
                )
                results["llm_response"] = llm_result.get("final", "")
                logger.info(f"  ✅ LLM response: {len(results['llm_response'])} chars")
            except Exception as e:
                logger.info(f"  ℹ️  LLM not available (server not running): {str(e)[:50]}...")
                if results.get("symbolic_result") and results["symbolic_result"].get("ok"):
                    results["llm_response"] = f"Symbolic result: {results['symbolic_result'].get('result', 'N/A')}"
                else:
                    results["llm_response"] = "LLM server not available (start llama-server to enable)"
        
        return results
    
    async def close(self):
        """Cleanup"""
        await self.orchestrator.close()
        logger.info("✅ Multi-LLM orchestrator closed")


async def demo_aluls_and_qwen():
    """Demo AL-ULS + Qwen integration"""
    
    print("\n" + "=" * 70)
    print("AL-ULS SYMBOLIC + MULTI-LLM (LFM2 + Qwen) DEMO")
    print("=" * 70)
    
    # Configure multiple LLM backends
    llm_configs = [
        {
            "base_url": "http://127.0.0.1:8080",
            "mode": "llama-cpp",
            "model": "LFM2-8B-A1B",
            "timeout": 60
        },
        {
            "base_url": "http://127.0.0.1:8081",  # Qwen on different port
            "mode": "openai-chat",
            "model": "Qwen2.5-7B",
            "timeout": 60
        },
        {
            "base_url": "http://127.0.0.1:8082",  # Another option
            "mode": "llama-cpp",
            "model": "Qwen2.5-Coder",
            "timeout": 60
        }
    ]
    
    # Create multi-LLM system
    system = MultiLLMOrchestrator(
        llm_configs=llm_configs,
        enable_aluls=True,
        numbskull_config={'use_fractal': True, 'cache_embeddings': True}
    )
    
    # Test symbolic expressions
    test_cases = [
        {"query": "SUM(1, 2, 3, 4, 5)", "context": None},
        {"query": "MEAN(10, 20, 30, 40, 50)", "context": None},
        {"query": "VAR(1, 2, 3, 4, 5)", "context": None},
        {"query": "What is quantum computing?", "context": "Focus on practical applications"},
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {test['query']}")
        print(f"{'='*70}")
        
        result = await system.process_with_symbolic(test["query"], test["context"])
        
        if result.get("symbolic_result"):
            sr = result["symbolic_result"]
            if sr.get("ok"):
                print(f"✅ Symbolic: {sr['function']}({','.join(map(str, sr['args']))}) = {sr['result']}")
        
        if result.get("embeddings"):
            print(f"✅ Embeddings: {result['embeddings']['components']}")
        
        if result.get("llm_response"):
            print(f"ℹ️  LLM: {result['llm_response'][:80]}...")
    
    # Show LLM backend info
    print(f"\n{'='*70}")
    print("MULTI-LLM CONFIGURATION")
    print(f"{'='*70}")
    print(f"Configured backends: {len(llm_configs)}")
    for i, config in enumerate(llm_configs, 1):
        print(f"  {i}. {config['model']} @ {config['base_url']} ({config['mode']})")
    
    print(f"\n💡 Start any of these LLM servers to enable full inference:")
    print(f"   llama-server --model LFM2-8B-A1B.gguf --port 8080")
    print(f"   llama-server --model Qwen2.5-7B.gguf --port 8081")
    print(f"   llama-server --model Qwen2.5-Coder.gguf --port 8082")
    
    await system.close()
    
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(demo_aluls_and_qwen())

