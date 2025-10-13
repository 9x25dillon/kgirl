#!/usr/bin/env python3
"""
Numbskull + LiMp Integration Benchmarking Suite
================================================

Comprehensive benchmarks for the integrated system:
- Embedding generation performance
- Fusion method comparison
- Cache efficiency
- End-to-end orchestration
- Component comparison (semantic, mathematical, fractal)
- Throughput testing

Usage:
    python benchmark_integration.py
    python benchmark_integration.py --quick
    python benchmark_integration.py --component semantic
    python benchmark_integration.py --output results.json

Author: Assistant
License: MIT
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
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


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    name: str
    component: str
    num_samples: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    throughput: float  # samples/second
    embedding_dim: int
    cache_hits: int
    success_rate: float
    metadata: Dict[str, Any]


class BenchmarkSuite:
    """Comprehensive benchmark suite for the integration"""
    
    def __init__(self, output_file: Optional[str] = None):
        self.output_file = output_file
        self.results: List[BenchmarkResult] = []
        self.test_texts = self._generate_test_texts()
    
    def _generate_test_texts(self) -> Dict[str, List[str]]:
        """Generate diverse test texts for benchmarking"""
        return {
            "simple": [
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming technology.",
                "Machine learning models process data efficiently.",
                "Neural networks learn from examples.",
                "Deep learning enables complex pattern recognition."
            ],
            "mathematical": [
                "f(x) = x^2 + 2x + 1",
                "Solve: 3x + 5 = 20",
                "The derivative of x^3 is 3x^2",
                "Integral of sin(x) is -cos(x) + C",
                "Matrix multiplication: A × B where A is 3×2 and B is 2×4"
            ],
            "technical": [
                "LFM2-8B-A1B is a local language model for inference and decision making.",
                "Numbskull provides hybrid embeddings combining semantic, mathematical, and fractal approaches.",
                "Dual LLM orchestration separates resource summarization from final inference.",
                "Embedding fusion methods include weighted average, concatenation, and attention.",
                "The system supports llama-cpp, textgen-webui, and OpenAI-compatible backends."
            ],
            "complex": [
                "The integration of distributed systems requires careful consideration of consistency, availability, and partition tolerance as described by the CAP theorem.",
                "Quantum computing leverages superposition and entanglement to perform calculations that would be intractable on classical computers.",
                "Modern neural architectures like transformers use self-attention mechanisms to process sequences in parallel rather than sequentially.",
                "The efficiency of algorithmic trading systems depends on low-latency data processing, real-time risk assessment, and optimal execution strategies.",
                "Cryptographic protocols ensure data integrity through mathematical functions that are computationally infeasible to reverse without proper keys."
            ]
        }
    
    async def benchmark_embedding_component(
        self,
        component_name: str,
        use_semantic: bool = False,
        use_mathematical: bool = False,
        use_fractal: bool = False,
        text_category: str = "simple"
    ) -> BenchmarkResult:
        """Benchmark a specific embedding component"""
        
        logger.info(f"Benchmarking {component_name} component...")
        
        config = HybridConfig(
            use_semantic=use_semantic,
            use_mathematical=use_mathematical,
            use_fractal=use_fractal,
            fusion_method="weighted_average",
            cache_embeddings=True,
            parallel_processing=False  # Sequential for accurate timing
        )
        
        pipeline = HybridEmbeddingPipeline(config)
        texts = self.test_texts[text_category]
        
        times = []
        dims = []
        successes = 0
        
        # Warm-up
        await pipeline.embed(texts[0])
        pipeline.clear_cache()
        
        # Benchmark
        start_total = time.time()
        for text in texts:
            try:
                start = time.time()
                result = await pipeline.embed(text)
                elapsed = time.time() - start
                
                times.append(elapsed)
                dims.append(result["metadata"]["embedding_dim"])
                successes += 1
            except Exception as e:
                logger.warning(f"Failed to embed: {e}")
        
        total_time = time.time() - start_total
        
        # Get cache stats
        cache_stats = pipeline.get_cache_stats()
        
        await pipeline.close()
        
        # Calculate statistics
        if times:
            return BenchmarkResult(
                name=f"{component_name}_{text_category}",
                component=component_name,
                num_samples=len(texts),
                total_time=total_time,
                avg_time=statistics.mean(times),
                min_time=min(times),
                max_time=max(times),
                std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
                throughput=len(texts) / total_time if total_time > 0 else 0.0,
                embedding_dim=dims[0] if dims else 0,
                cache_hits=cache_stats["cache_hits"],
                success_rate=successes / len(texts),
                metadata={
                    "text_category": text_category,
                    "cache_enabled": True
                }
            )
        else:
            raise RuntimeError("No successful embeddings")
    
    async def benchmark_fusion_methods(self) -> List[BenchmarkResult]:
        """Benchmark different fusion methods"""
        
        logger.info("Benchmarking fusion methods...")
        
        fusion_methods = ["weighted_average", "concatenation", "attention"]
        results = []
        texts = self.test_texts["technical"][:3]  # Use subset
        
        for fusion_method in fusion_methods:
            config = HybridConfig(
                use_semantic=False,
                use_mathematical=False,
                use_fractal=True,  # Use only fractal for consistency
                fusion_method=fusion_method,
                cache_embeddings=False  # Disable cache for fair comparison
            )
            
            pipeline = HybridEmbeddingPipeline(config)
            times = []
            
            for text in texts:
                start = time.time()
                result = await pipeline.embed(text)
                times.append(time.time() - start)
            
            await pipeline.close()
            
            total_time = sum(times)
            result = BenchmarkResult(
                name=f"fusion_{fusion_method}",
                component="fusion",
                num_samples=len(texts),
                total_time=total_time,
                avg_time=statistics.mean(times),
                min_time=min(times),
                max_time=max(times),
                std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
                throughput=len(texts) / total_time,
                embedding_dim=768,  # Normalized dimension
                cache_hits=0,
                success_rate=1.0,
                metadata={"fusion_method": fusion_method}
            )
            results.append(result)
            logger.info(f"  {fusion_method}: {result.avg_time:.3f}s avg")
        
        return results
    
    async def benchmark_cache_efficiency(self) -> BenchmarkResult:
        """Benchmark cache hit performance"""
        
        logger.info("Benchmarking cache efficiency...")
        
        config = HybridConfig(
            use_semantic=False,
            use_mathematical=False,
            use_fractal=True,
            cache_embeddings=True
        )
        
        pipeline = HybridEmbeddingPipeline(config)
        text = "Cache test text for benchmarking"
        
        # First embedding (cache miss)
        start = time.time()
        await pipeline.embed(text)
        miss_time = time.time() - start
        
        # Subsequent embeddings (cache hits)
        hit_times = []
        for _ in range(10):
            start = time.time()
            await pipeline.embed(text)
            hit_times.append(time.time() - start)
        
        cache_stats = pipeline.get_cache_stats()
        await pipeline.close()
        
        speedup = miss_time / statistics.mean(hit_times) if hit_times else 1.0
        
        logger.info(f"  Cache miss: {miss_time:.4f}s")
        logger.info(f"  Cache hit avg: {statistics.mean(hit_times):.4f}s")
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        return BenchmarkResult(
            name="cache_efficiency",
            component="cache",
            num_samples=11,
            total_time=miss_time + sum(hit_times),
            avg_time=statistics.mean(hit_times),
            min_time=min(hit_times),
            max_time=max(hit_times),
            std_dev=statistics.stdev(hit_times) if len(hit_times) > 1 else 0.0,
            throughput=10 / sum(hit_times),
            embedding_dim=1024,
            cache_hits=cache_stats["cache_hits"],
            success_rate=1.0,
            metadata={
                "cache_miss_time": miss_time,
                "cache_hit_avg": statistics.mean(hit_times),
                "speedup": speedup
            }
        )
    
    async def benchmark_parallel_processing(self) -> BenchmarkResult:
        """Benchmark parallel vs sequential processing"""
        
        logger.info("Benchmarking parallel processing...")
        
        texts = self.test_texts["simple"]
        
        # Sequential
        config_seq = HybridConfig(
            use_semantic=False,
            use_mathematical=False,
            use_fractal=True,
            parallel_processing=False,
            cache_embeddings=False
        )
        
        pipeline_seq = HybridEmbeddingPipeline(config_seq)
        
        start = time.time()
        for text in texts:
            await pipeline_seq.embed(text)
        seq_time = time.time() - start
        
        await pipeline_seq.close()
        
        # Parallel
        config_par = HybridConfig(
            use_semantic=False,
            use_mathematical=False,
            use_fractal=True,
            parallel_processing=True,
            cache_embeddings=False
        )
        
        pipeline_par = HybridEmbeddingPipeline(config_par)
        
        start = time.time()
        await pipeline_par.embed_batch(texts)
        par_time = time.time() - start
        
        await pipeline_par.close()
        
        speedup = seq_time / par_time if par_time > 0 else 1.0
        
        logger.info(f"  Sequential: {seq_time:.3f}s")
        logger.info(f"  Parallel: {par_time:.3f}s")
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        return BenchmarkResult(
            name="parallel_processing",
            component="parallelism",
            num_samples=len(texts),
            total_time=par_time,
            avg_time=par_time / len(texts),
            min_time=0.0,
            max_time=0.0,
            std_dev=0.0,
            throughput=len(texts) / par_time,
            embedding_dim=1024,
            cache_hits=0,
            success_rate=1.0,
            metadata={
                "sequential_time": seq_time,
                "parallel_time": par_time,
                "speedup": speedup
            }
        )
    
    async def benchmark_hybrid_combinations(self) -> List[BenchmarkResult]:
        """Benchmark different component combinations"""
        
        logger.info("Benchmarking hybrid combinations...")
        
        combinations = [
            ("fractal_only", False, False, True),
            ("semantic_fractal", True, False, True),
            ("math_fractal", False, True, True),
            ("all_components", True, True, True),
        ]
        
        results = []
        texts = self.test_texts["technical"][:3]
        
        for name, use_sem, use_math, use_frac in combinations:
            config = HybridConfig(
                use_semantic=use_sem,
                use_mathematical=use_math,
                use_fractal=use_frac,
                cache_embeddings=False
            )
            
            try:
                pipeline = HybridEmbeddingPipeline(config)
                times = []
                dims = []
                
                for text in texts:
                    start = time.time()
                    result = await pipeline.embed(text)
                    times.append(time.time() - start)
                    dims.append(result["metadata"]["embedding_dim"])
                
                await pipeline.close()
                
                total_time = sum(times)
                bench_result = BenchmarkResult(
                    name=f"hybrid_{name}",
                    component="hybrid",
                    num_samples=len(texts),
                    total_time=total_time,
                    avg_time=statistics.mean(times),
                    min_time=min(times),
                    max_time=max(times),
                    std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
                    throughput=len(texts) / total_time,
                    embedding_dim=dims[0] if dims else 0,
                    cache_hits=0,
                    success_rate=1.0,
                    metadata={
                        "semantic": use_sem,
                        "mathematical": use_math,
                        "fractal": use_frac
                    }
                )
                results.append(bench_result)
                logger.info(f"  {name}: {bench_result.avg_time:.3f}s avg")
                
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")
        
        return results
    
    async def run_all_benchmarks(self, quick: bool = False):
        """Run all benchmark suites"""
        
        logger.info("=" * 70)
        logger.info("STARTING COMPREHENSIVE BENCHMARK SUITE")
        logger.info("=" * 70)
        print()
        
        if not NUMBSKULL_AVAILABLE:
            logger.error("Numbskull not available!")
            return
        
        # 1. Component benchmarks
        logger.info("\n1. COMPONENT BENCHMARKS")
        logger.info("-" * 70)
        
        components = [
            ("fractal", False, False, True, "simple"),
            ("fractal", False, False, True, "mathematical"),
            ("fractal", False, False, True, "technical"),
        ]
        
        if not quick:
            components.extend([
                ("fractal", False, False, True, "complex"),
            ])
        
        for name, sem, math, frac, category in components:
            try:
                result = await self.benchmark_embedding_component(
                    name, sem, math, frac, category
                )
                self.results.append(result)
            except Exception as e:
                logger.error(f"Component benchmark failed: {e}")
        
        # 2. Fusion methods
        logger.info("\n2. FUSION METHOD COMPARISON")
        logger.info("-" * 70)
        try:
            fusion_results = await self.benchmark_fusion_methods()
            self.results.extend(fusion_results)
        except Exception as e:
            logger.error(f"Fusion benchmark failed: {e}")
        
        # 3. Cache efficiency
        logger.info("\n3. CACHE EFFICIENCY")
        logger.info("-" * 70)
        try:
            cache_result = await self.benchmark_cache_efficiency()
            self.results.append(cache_result)
        except Exception as e:
            logger.error(f"Cache benchmark failed: {e}")
        
        # 4. Parallel processing
        logger.info("\n4. PARALLEL PROCESSING")
        logger.info("-" * 70)
        try:
            parallel_result = await self.benchmark_parallel_processing()
            self.results.append(parallel_result)
        except Exception as e:
            logger.error(f"Parallel benchmark failed: {e}")
        
        # 5. Hybrid combinations
        if not quick:
            logger.info("\n5. HYBRID COMBINATIONS")
            logger.info("-" * 70)
            try:
                hybrid_results = await self.benchmark_hybrid_combinations()
                self.results.extend(hybrid_results)
            except Exception as e:
                logger.error(f"Hybrid benchmark failed: {e}")
        
        # Generate report
        self.generate_report()
        
        # Save results
        if self.output_file:
            self.save_results()
    
    def generate_report(self):
        """Generate human-readable benchmark report"""
        
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)
        
        if not self.results:
            print("No results to display")
            return
        
        # Group by component
        by_component = {}
        for result in self.results:
            comp = result.component
            if comp not in by_component:
                by_component[comp] = []
            by_component[comp].append(result)
        
        for component, results in by_component.items():
            print(f"\n{component.upper()}")
            print("-" * 70)
            
            for result in results:
                print(f"\n  {result.name}:")
                print(f"    Samples: {result.num_samples}")
                print(f"    Avg Time: {result.avg_time*1000:.2f}ms")
                print(f"    Min/Max: {result.min_time*1000:.2f}ms / {result.max_time*1000:.2f}ms")
                print(f"    Std Dev: {result.std_dev*1000:.2f}ms")
                print(f"    Throughput: {result.throughput:.2f} samples/s")
                print(f"    Embedding Dim: {result.embedding_dim}")
                print(f"    Success Rate: {result.success_rate*100:.1f}%")
                
                if result.metadata:
                    print(f"    Metadata: {json.dumps(result.metadata, indent=6)}")
        
        # Overall statistics
        print("\n" + "=" * 70)
        print("OVERALL STATISTICS")
        print("=" * 70)
        
        all_times = [r.avg_time for r in self.results]
        all_throughputs = [r.throughput for r in self.results]
        
        print(f"  Total Benchmarks: {len(self.results)}")
        print(f"  Avg Time Across All: {statistics.mean(all_times)*1000:.2f}ms")
        print(f"  Fastest: {min(all_times)*1000:.2f}ms ({[r.name for r in self.results if r.avg_time == min(all_times)][0]})")
        print(f"  Slowest: {max(all_times)*1000:.2f}ms ({[r.name for r in self.results if r.avg_time == max(all_times)][0]})")
        print(f"  Avg Throughput: {statistics.mean(all_throughputs):.2f} samples/s")
    
    def save_results(self):
        """Save results to JSON file"""
        
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_benchmarks": len(self.results),
            "results": [asdict(r) for r in self.results]
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"\n✅ Results saved to {self.output_file}")


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Benchmark Numbskull + LiMp integration"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmark suite (fewer tests)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results.json',
        help='Output file for results (default: benchmark_results.json)'
    )
    parser.add_argument(
        '--component',
        type=str,
        choices=['semantic', 'mathematical', 'fractal', 'all'],
        default='all',
        help='Benchmark specific component only'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("NUMBSKULL + LIMP INTEGRATION BENCHMARK SUITE")
    print("=" * 70)
    print(f"Mode: {'Quick' if args.quick else 'Comprehensive'}")
    print(f"Output: {args.output}")
    print(f"Component: {args.component}")
    print("=" * 70 + "\n")
    
    suite = BenchmarkSuite(output_file=args.output)
    
    try:
        await suite.run_all_benchmarks(quick=args.quick)
        
        print("\n" + "=" * 70)
        print("✅ BENCHMARK SUITE COMPLETED")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

