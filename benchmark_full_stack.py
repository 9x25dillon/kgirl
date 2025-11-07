#!/usr/bin/env python3
"""
Full Stack Benchmark: LFM2-8B-A1B + Numbskull + Services
==========================================================

Comprehensive end-to-end benchmarks including:
- Semantic embeddings (Eopiez service if available)
- Mathematical embeddings (LIMPS service if available)
- Fractal embeddings (always available)
- LFM2-8B-A1B integration (if server running)
- Complete dual LLM orchestration pipeline

Usage:
    python benchmark_full_stack.py
    python benchmark_full_stack.py --with-llm
    python benchmark_full_stack.py --services-only
    python benchmark_full_stack.py --all

Author: Assistant
License: MIT
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import statistics

# Add numbskull to path
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

from advanced_embedding_pipeline import (
    HybridEmbeddingPipeline,
    HybridConfig,
    SemanticConfig,
    MathematicalConfig,
    FractalConfig
)

from numbskull_dual_orchestrator import (
    create_numbskull_orchestrator,
    NUMBSKULL_AVAILABLE
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServiceChecker:
    """Check availability of external services"""
    
    @staticmethod
    async def check_service(url: str, name: str) -> bool:
        """Check if a service is available"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{url}/health")
                if response.status_code < 500:
                    logger.info(f"✅ {name} service available at {url}")
                    return True
        except Exception as e:
            logger.warning(f"⚠️  {name} service not available at {url}: {type(e).__name__}")
        return False
    
    @staticmethod
    async def check_llm(url: str, mode: str = "llama-cpp") -> bool:
        """Check if LLM server is available"""
        try:
            import httpx
            
            # Different endpoints for different modes
            if mode == "llama-cpp":
                endpoint = "/health"
            elif mode == "openai-chat":
                endpoint = "/v1/models"
            else:
                endpoint = "/api/v1/model"
            
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{url}{endpoint}")
                if response.status_code < 500:
                    logger.info(f"✅ LLM server available at {url}")
                    return True
        except Exception as e:
            logger.warning(f"⚠️  LLM server not available at {url}: {type(e).__name__}")
        return False


class FullStackBenchmark:
    """Full stack benchmark including all services"""
    
    def __init__(self):
        self.services = {
            "eopiez": False,
            "limps": False,
            "lfm2": False
        }
        self.results = []
        self.test_data = self._prepare_test_data()
    
    def _prepare_test_data(self) -> Dict[str, Any]:
        """Prepare diverse test data"""
        return {
            "semantic_texts": [
                "The rapid advancement of artificial intelligence is transforming industries.",
                "Climate change poses significant challenges to global ecosystems.",
                "Quantum computing promises exponential speedups for certain problems.",
            ],
            "mathematical_texts": [
                "Solve the equation: x^2 - 5x + 6 = 0",
                "Calculate the derivative of f(x) = 3x^3 + 2x^2 - x + 5",
                "Find the integral of sin(x)cos(x) dx",
            ],
            "technical_texts": [
                "The LFM2-8B-A1B model provides efficient local inference for decision-making tasks.",
                "Hybrid embedding systems combine multiple representation techniques for richer context.",
                "Dual LLM orchestration separates resource summarization from final inference.",
            ],
            "queries": [
                "Summarize the key concepts and their relationships.",
                "What are the main technical challenges mentioned?",
                "Explain the mathematical relationships in the context.",
            ]
        }
    
    async def check_services(self):
        """Check which services are available"""
        logger.info("\n" + "=" * 70)
        logger.info("CHECKING SERVICE AVAILABILITY")
        logger.info("=" * 70)
        
        checker = ServiceChecker()
        
        # Check Eopiez (semantic embeddings)
        self.services["eopiez"] = await checker.check_service(
            "http://127.0.0.1:8001",
            "Eopiez (Semantic)"
        )
        
        # Check LIMPS (mathematical embeddings)
        self.services["limps"] = await checker.check_service(
            "http://127.0.0.1:8000",
            "LIMPS (Mathematical)"
        )
        
        # Check LFM2-8B-A1B
        self.services["lfm2"] = await checker.check_llm(
            "http://127.0.0.1:8080",
            "llama-cpp"
        )
        
        logger.info("\nService Summary:")
        logger.info(f"  Eopiez (Semantic):     {'✅ Available' if self.services['eopiez'] else '❌ Unavailable'}")
        logger.info(f"  LIMPS (Mathematical):  {'✅ Available' if self.services['limps'] else '❌ Unavailable'}")
        logger.info(f"  LFM2-8B-A1B (LLM):     {'✅ Available' if self.services['lfm2'] else '❌ Unavailable'}")
        logger.info(f"  Fractal (Local):       ✅ Always available")
    
    async def benchmark_semantic_embeddings(self) -> Dict[str, Any]:
        """Benchmark semantic embeddings with Eopiez service"""
        
        if not self.services["eopiez"]:
            logger.info("\n⚠️  Skipping semantic benchmark (Eopiez not available)")
            return None
        
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARKING SEMANTIC EMBEDDINGS (Eopiez)")
        logger.info("=" * 70)
        
        config = HybridConfig(
            use_semantic=True,
            use_mathematical=False,
            use_fractal=False,
            cache_embeddings=False,
            semantic_config=SemanticConfig(
                api_url="http://127.0.0.1:8001",
                timeout=30.0
            )
        )
        
        pipeline = HybridEmbeddingPipeline(config)
        texts = self.test_data["semantic_texts"]
        times = []
        successes = 0
        
        for text in texts:
            try:
                start = time.time()
                result = await pipeline.embed(text)
                elapsed = time.time() - start
                times.append(elapsed)
                successes += 1
                logger.info(f"  ✅ Embedded ({elapsed*1000:.2f}ms): {text[:50]}...")
            except Exception as e:
                logger.warning(f"  ❌ Failed: {e}")
        
        await pipeline.close()
        
        if times:
            result = {
                "component": "semantic",
                "num_samples": len(texts),
                "successes": successes,
                "avg_time_ms": statistics.mean(times) * 1000,
                "min_time_ms": min(times) * 1000,
                "max_time_ms": max(times) * 1000,
                "throughput": len(texts) / sum(times),
                "success_rate": successes / len(texts)
            }
            
            logger.info(f"\n  Results:")
            logger.info(f"    Average: {result['avg_time_ms']:.2f}ms")
            logger.info(f"    Throughput: {result['throughput']:.2f} samples/s")
            logger.info(f"    Success Rate: {result['success_rate']*100:.1f}%")
            
            return result
        
        return None
    
    async def benchmark_mathematical_embeddings(self) -> Dict[str, Any]:
        """Benchmark mathematical embeddings with LIMPS service"""
        
        if not self.services["limps"]:
            logger.info("\n⚠️  Skipping mathematical benchmark (LIMPS not available)")
            return None
        
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARKING MATHEMATICAL EMBEDDINGS (LIMPS)")
        logger.info("=" * 70)
        
        config = HybridConfig(
            use_semantic=False,
            use_mathematical=True,
            use_fractal=False,
            cache_embeddings=False,
            mathematical_config=MathematicalConfig(
                limps_url="http://127.0.0.1:8000",
                timeout=30.0
            )
        )
        
        pipeline = HybridEmbeddingPipeline(config)
        texts = self.test_data["mathematical_texts"]
        times = []
        successes = 0
        
        for text in texts:
            try:
                start = time.time()
                result = await pipeline.embed(text)
                elapsed = time.time() - start
                times.append(elapsed)
                successes += 1
                logger.info(f"  ✅ Embedded ({elapsed*1000:.2f}ms): {text[:50]}...")
            except Exception as e:
                logger.warning(f"  ❌ Failed: {e}")
        
        await pipeline.close()
        
        if times:
            result = {
                "component": "mathematical",
                "num_samples": len(texts),
                "successes": successes,
                "avg_time_ms": statistics.mean(times) * 1000,
                "min_time_ms": min(times) * 1000,
                "max_time_ms": max(times) * 1000,
                "throughput": len(texts) / sum(times),
                "success_rate": successes / len(texts)
            }
            
            logger.info(f"\n  Results:")
            logger.info(f"    Average: {result['avg_time_ms']:.2f}ms")
            logger.info(f"    Throughput: {result['throughput']:.2f} samples/s")
            logger.info(f"    Success Rate: {result['success_rate']*100:.1f}%")
            
            return result
        
        return None
    
    async def benchmark_full_hybrid(self) -> Dict[str, Any]:
        """Benchmark full hybrid system with all available components"""
        
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARKING FULL HYBRID SYSTEM")
        logger.info("=" * 70)
        
        # Use all available components
        config = HybridConfig(
            use_semantic=self.services["eopiez"],
            use_mathematical=self.services["limps"],
            use_fractal=True,  # Always available
            fusion_method="weighted_average",
            cache_embeddings=False
        )
        
        if self.services["eopiez"]:
            config.semantic_config = SemanticConfig(
                api_url="http://127.0.0.1:8001",
                timeout=30.0
            )
        
        if self.services["limps"]:
            config.mathematical_config = MathematicalConfig(
                limps_url="http://127.0.0.1:8000",
                timeout=30.0
            )
        
        pipeline = HybridEmbeddingPipeline(config)
        texts = self.test_data["technical_texts"]
        times = []
        successes = 0
        components_used = []
        
        for text in texts:
            try:
                start = time.time()
                result = await pipeline.embed(text)
                elapsed = time.time() - start
                times.append(elapsed)
                successes += 1
                components_used.append(result["metadata"]["components_used"])
                logger.info(f"  ✅ Embedded ({elapsed*1000:.2f}ms): {text[:50]}...")
                logger.info(f"     Components: {result['metadata']['components_used']}")
            except Exception as e:
                logger.warning(f"  ❌ Failed: {e}")
        
        await pipeline.close()
        
        if times:
            result = {
                "component": "hybrid_full",
                "num_samples": len(texts),
                "successes": successes,
                "avg_time_ms": statistics.mean(times) * 1000,
                "min_time_ms": min(times) * 1000,
                "max_time_ms": max(times) * 1000,
                "throughput": len(texts) / sum(times),
                "success_rate": successes / len(texts),
                "components_used": components_used[0] if components_used else []
            }
            
            logger.info(f"\n  Results:")
            logger.info(f"    Average: {result['avg_time_ms']:.2f}ms")
            logger.info(f"    Throughput: {result['throughput']:.2f} samples/s")
            logger.info(f"    Components: {result['components_used']}")
            logger.info(f"    Success Rate: {result['success_rate']*100:.1f}%")
            
            return result
        
        return None
    
    async def benchmark_llm_integration(self) -> Dict[str, Any]:
        """Benchmark end-to-end with LFM2-8B-A1B"""
        
        if not self.services["lfm2"]:
            logger.info("\n⚠️  Skipping LLM integration benchmark (LFM2-8B-A1B not available)")
            return None
        
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARKING END-TO-END LLM INTEGRATION")
        logger.info("=" * 70)
        
        # Create orchestrator with all available components
        settings = {
            "use_numbskull": True,
            "use_semantic": self.services["eopiez"],
            "use_mathematical": self.services["limps"],
            "use_fractal": True,
            "fusion_method": "weighted_average",
            "embedding_enhancement": "metadata",
            "temperature": 0.7,
            "max_tokens": 256
        }
        
        numbskull_config = {
            "use_semantic": self.services["eopiez"],
            "use_mathematical": self.services["limps"],
            "use_fractal": True,
            "cache_embeddings": False
        }
        
        orchestrator = create_numbskull_orchestrator(
            local_configs=[{
                "base_url": "http://127.0.0.1:8080",
                "mode": "llama-cpp",
                "model": "LFM2-8B-A1B",
                "timeout": 60
            }],
            remote_config=None,  # Use local fallback
            settings=settings,
            numbskull_config=numbskull_config
        )
        
        queries = self.test_data["queries"][:2]  # Test 2 queries
        times = []
        embedding_times = []
        successes = 0
        
        for query in queries:
            try:
                logger.info(f"\n  Query: {query}")
                
                start = time.time()
                result = await orchestrator.run_with_embeddings(
                    user_prompt=query,
                    resource_paths=[],
                    inline_resources=self.test_data["technical_texts"][:1]
                )
                total_time = time.time() - start
                
                times.append(total_time)
                successes += 1
                
                # Extract embedding time
                if result.get("embedding_result"):
                    emb_time = result["embedding_result"]["metadata"]["processing_time"]
                    embedding_times.append(emb_time)
                
                logger.info(f"  ✅ Completed in {total_time:.2f}s")
                logger.info(f"     Embedding: {emb_time*1000:.2f}ms")
                logger.info(f"     LLM: {(total_time - emb_time):.2f}s")
                logger.info(f"     Answer length: {len(result['final'])} chars")
                
            except Exception as e:
                logger.warning(f"  ❌ Failed: {e}")
        
        await orchestrator.close()
        
        if times:
            result = {
                "component": "end_to_end_llm",
                "num_samples": len(queries),
                "successes": successes,
                "avg_total_time_s": statistics.mean(times),
                "avg_embedding_time_ms": statistics.mean(embedding_times) * 1000 if embedding_times else 0,
                "avg_llm_time_s": statistics.mean([t - e for t, e in zip(times, embedding_times)]) if embedding_times else 0,
                "embedding_overhead_pct": (statistics.mean(embedding_times) / statistics.mean(times) * 100) if embedding_times else 0,
                "success_rate": successes / len(queries)
            }
            
            logger.info(f"\n  End-to-End Results:")
            logger.info(f"    Total Time: {result['avg_total_time_s']:.2f}s")
            logger.info(f"    Embedding Time: {result['avg_embedding_time_ms']:.2f}ms")
            logger.info(f"    LLM Time: {result['avg_llm_time_s']:.2f}s")
            logger.info(f"    Embedding Overhead: {result['embedding_overhead_pct']:.2f}%")
            logger.info(f"    Success Rate: {result['success_rate']*100:.1f}%")
            
            return result
        
        return None
    
    async def run_all(self, services_only: bool = False, llm_only: bool = False):
        """Run all available benchmarks"""
        
        logger.info("\n" + "=" * 70)
        logger.info("FULL STACK BENCHMARK SUITE")
        logger.info("=" * 70)
        
        # Check services
        await self.check_services()
        
        if not services_only:
            # Test individual components
            sem_result = await self.benchmark_semantic_embeddings()
            if sem_result:
                self.results.append(sem_result)
            
            math_result = await self.benchmark_mathematical_embeddings()
            if math_result:
                self.results.append(math_result)
        
        if not llm_only:
            # Test hybrid system
            hybrid_result = await self.benchmark_full_hybrid()
            if hybrid_result:
                self.results.append(hybrid_result)
        
        # Test LLM integration
        llm_result = await self.benchmark_llm_integration()
        if llm_result:
            self.results.append(llm_result)
        
        # Generate report
        self.generate_report()
        
        # Save results
        self.save_results()
    
    def generate_report(self):
        """Generate summary report"""
        
        logger.info("\n" + "=" * 70)
        logger.info("FULL STACK BENCHMARK RESULTS")
        logger.info("=" * 70)
        
        if not self.results:
            logger.info("No results to report")
            return
        
        for result in self.results:
            logger.info(f"\n{result['component'].upper()}")
            logger.info("-" * 70)
            for key, value in result.items():
                if key != "component":
                    logger.info(f"  {key}: {value}")
    
    def save_results(self):
        """Save results to file"""
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "services": self.services,
            "results": self.results
        }
        
        filename = "benchmark_full_stack_results.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"\n✅ Results saved to {filename}")


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Full stack benchmark with all services"
    )
    parser.add_argument(
        '--with-llm',
        action='store_true',
        help='Include LLM integration tests'
    )
    parser.add_argument(
        '--services-only',
        action='store_true',
        help='Only benchmark external services'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all benchmarks'
    )
    
    args = parser.parse_args()
    
    benchmark = FullStackBenchmark()
    
    try:
        await benchmark.run_all(
            services_only=args.services_only,
            llm_only=not args.all and args.with_llm
        )
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ FULL STACK BENCHMARK COMPLETED")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Benchmark interrupted")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

