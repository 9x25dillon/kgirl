#!/usr/bin/env python3
"""
Advanced Embedding Pipeline Demo
Demonstrates the sophisticated multi-modal embedding system
"""

import asyncio
import logging
import numpy as np
import json
import time
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# Import our embedding pipeline components
from .semantic_embedder import SemanticEmbedder, SemanticConfig
from .mathematical_embedder import MathematicalEmbedder, MathematicalConfig
from .fractal_cascade_embedder import FractalCascadeEmbedder, FractalConfig
from .hybrid_pipeline import HybridEmbeddingPipeline, HybridConfig
from .optimizer import EmbeddingOptimizer, OptimizationConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("embedding_demo")


class EmbeddingPipelineDemo:
    """Demo class for the advanced embedding pipeline"""
    
    def __init__(self):
        self.results = {}
        
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of all embedding capabilities"""
        logger.info("🚀 Starting Advanced Embedding Pipeline Demo")
        
        # Demo texts covering different domains
        demo_texts = [
            "The quick brown fox jumps over the lazy dog",
            "x^2 + y^2 = z^2",  # Mathematical
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",  # Code
            "Fractal geometry reveals infinite complexity in finite space",
            "Artificial intelligence and machine learning revolutionize data processing",
            "Quantum mechanics describes the behavior of matter at atomic scales",
            "sin(x) * cos(y) + e^(i*pi) = -1",  # Complex mathematical
            "Neural networks learn hierarchical representations through backpropagation",
            "The Mandelbrot set exhibits self-similarity at all scales",
            "Natural language processing combines linguistics with computational methods"
        ]
        
        # Run individual component demos
        await self.demo_semantic_embedder(demo_texts[:5])
        await self.demo_mathematical_embedder(demo_texts[1:6])
        await self.demo_fractal_embedder(demo_texts[3:8])
        await self.demo_hybrid_pipeline(demo_texts)
        await self.demo_optimization(demo_texts)
        
        # Generate comprehensive report
        self.generate_demo_report()
        
        logger.info("✅ Demo completed successfully!")
    
    async def demo_semantic_embedder(self, texts: List[str]):
        """Demo semantic embedding capabilities"""
        logger.info("📝 Demo: Semantic Embedder")
        
        try:
            config = SemanticConfig(
                embedding_dim=768,
                batch_size=16,
                use_cache=True
            )
            embedder = SemanticEmbedder(config)
            
            start_time = time.time()
            embeddings = await embedder.embed_batch(texts)
            processing_time = time.time() - start_time
            
            # Analyze embeddings
            embedding_analysis = self._analyze_embeddings(embeddings, "semantic")
            
            self.results["semantic"] = {
                "embeddings": [emb.tolist() for emb in embeddings],
                "analysis": embedding_analysis,
                "processing_time": processing_time,
                "config": config.__dict__
            }
            
            logger.info(f"✅ Semantic embeddings generated: {len(embeddings)} vectors")
            logger.info(f"   Processing time: {processing_time:.2f}s")
            logger.info(f"   Average embedding norm: {embedding_analysis['average_norm']:.4f}")
            
            await embedder.close()
            
        except Exception as e:
            logger.error(f"❌ Semantic embedder demo failed: {e}")
    
    async def demo_mathematical_embedder(self, texts: List[str]):
        """Demo mathematical embedding capabilities"""
        logger.info("🔢 Demo: Mathematical Embedder")
        
        try:
            config = MathematicalConfig(
                max_dimension=1024,
                polynomial_degree=3,
                use_matrix_optimization=False  # Disable LIMPS for demo
            )
            embedder = MathematicalEmbedder(config)
            
            start_time = time.time()
            embeddings = []
            for text in texts:
                embedding = await embedder.embed_mathematical_expression(text)
                embeddings.append(embedding)
            processing_time = time.time() - start_time
            
            # Analyze embeddings
            embedding_analysis = self._analyze_embeddings(embeddings, "mathematical")
            
            self.results["mathematical"] = {
                "embeddings": [emb.tolist() for emb in embeddings],
                "analysis": embedding_analysis,
                "processing_time": processing_time,
                "config": config.__dict__
            }
            
            logger.info(f"✅ Mathematical embeddings generated: {len(embeddings)} vectors")
            logger.info(f"   Processing time: {processing_time:.2f}s")
            logger.info(f"   Average embedding norm: {embedding_analysis['average_norm']:.4f}")
            
            await embedder.close()
            
        except Exception as e:
            logger.error(f"❌ Mathematical embedder demo failed: {e}")
    
    async def demo_fractal_embedder(self, texts: List[str]):
        """Demo fractal cascade embedding capabilities"""
        logger.info("🌀 Demo: Fractal Cascade Embedder")
        
        try:
            config = FractalConfig(
                max_depth=5,
                branching_factor=3,
                embedding_dim=1024,
                fractal_type="mandelbrot",
                use_entropy=True,
                visualization=False
            )
            embedder = FractalCascadeEmbedder(config)
            
            start_time = time.time()
            embeddings = embedder.embed_batch(texts)
            processing_time = time.time() - start_time
            
            # Analyze embeddings
            embedding_analysis = self._analyze_embeddings(embeddings, "fractal")
            
            self.results["fractal"] = {
                "embeddings": [emb.tolist() for emb in embeddings],
                "analysis": embedding_analysis,
                "processing_time": processing_time,
                "config": config.__dict__
            }
            
            logger.info(f"✅ Fractal embeddings generated: {len(embeddings)} vectors")
            logger.info(f"   Processing time: {processing_time:.2f}s")
            logger.info(f"   Average embedding norm: {embedding_analysis['average_norm']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Fractal embedder demo failed: {e}")
    
    async def demo_hybrid_pipeline(self, texts: List[str]):
        """Demo hybrid embedding pipeline"""
        logger.info("🔗 Demo: Hybrid Embedding Pipeline")
        
        try:
            # Configure components
            semantic_config = SemanticConfig(embedding_dim=512, batch_size=8)
            mathematical_config = MathematicalConfig(max_dimension=512, use_matrix_optimization=False)
            fractal_config = FractalConfig(embedding_dim=512, max_depth=4)
            
            hybrid_config = HybridConfig(
                semantic_config=semantic_config,
                mathematical_config=mathematical_config,
                fractal_config=fractal_config,
                use_semantic=True,
                use_mathematical=True,
                use_fractal=True,
                fusion_method="weighted_average",
                semantic_weight=0.4,
                mathematical_weight=0.3,
                fractal_weight=0.3,
                parallel_processing=True
            )
            
            pipeline = HybridEmbeddingPipeline(hybrid_config)
            
            start_time = time.time()
            results = await pipeline.embed_batch(texts)
            processing_time = time.time() - start_time
            
            # Analyze results
            hybrid_analysis = self._analyze_hybrid_results(results)
            
            self.results["hybrid"] = {
                "results": results,
                "analysis": hybrid_analysis,
                "processing_time": processing_time,
                "config": hybrid_config.__dict__,
                "metrics": pipeline.get_metrics()
            }
            
            logger.info(f"✅ Hybrid embeddings generated: {len(results)} results")
            logger.info(f"   Processing time: {processing_time:.2f}s")
            logger.info(f"   Average processing time per text: {processing_time/len(texts):.3f}s")
            
            await pipeline.close()
            
        except Exception as e:
            logger.error(f"❌ Hybrid pipeline demo failed: {e}")
    
    async def demo_optimization(self, texts: List[str]):
        """Demo embedding optimization capabilities"""
        logger.info("⚡ Demo: Embedding Optimization")
        
        try:
            config = OptimizationConfig(
                use_disk_cache=True,
                batch_processing=True,
                max_batch_size=16,
                adaptive_batching=True,
                use_indexing=True,
                index_type="faiss"
            )
            
            optimizer = EmbeddingOptimizer(config)
            
            # Create a simple embedder function for demo
            async def demo_embedder(texts_batch):
                # Simulate embedding generation
                embeddings = []
                for text in texts_batch:
                    # Simple hash-based embedding for demo
                    import hashlib
                    hash_val = hashlib.md5(text.encode()).digest()
                    embedding = np.frombuffer(hash_val, dtype=np.float32)
                    embedding = np.tile(embedding, 24)[:768]  # Pad to 768 dimensions
                    embedding = embedding / np.linalg.norm(embedding)  # Normalize
                    
                    result = {
                        "text": text,
                        "embedding": embedding,
                        "metadata": {"created_at": time.time()}
                    }
                    embeddings.append(result)
                
                return embeddings
            
            # Test optimization
            start_time = time.time()
            
            # First run (no cache)
            results1 = await optimizer.optimize_embedding_generation(
                demo_embedder, texts, "demo_config_1"
            )
            
            # Second run (with cache)
            results2 = await optimizer.optimize_embedding_generation(
                demo_embedder, texts, "demo_config_1"
            )
            
            processing_time = time.time() - start_time
            
            # Create index
            embeddings = [result["embedding"] for result in results1]
            index_data = optimizer.create_index(embeddings, texts)
            
            # Test search
            if index_data["index"]:
                query_embedding = embeddings[0]
                search_results = optimizer.search_similar(index_data, query_embedding, top_k=5)
            
            optimization_analysis = {
                "first_run_time": processing_time / 2,
                "cache_effectiveness": optimizer.cache.get_stats(),
                "index_created": index_data["type"] != "none",
                "search_results": search_results if index_data["index"] else []
            }
            
            self.results["optimization"] = {
                "results": results1,
                "analysis": optimization_analysis,
                "processing_time": processing_time,
                "config": config.__dict__,
                "performance_report": optimizer.get_performance_report()
            }
            
            logger.info(f"✅ Optimization demo completed")
            logger.info(f"   Processing time: {processing_time:.2f}s")
            logger.info(f"   Cache hit rate: {optimization_analysis['cache_effectiveness']['hit_rate']:.2%}")
            logger.info(f"   Index created: {optimization_analysis['index_created']}")
            
        except Exception as e:
            logger.error(f"❌ Optimization demo failed: {e}")
    
    def _analyze_embeddings(self, embeddings: List[np.ndarray], embedder_type: str) -> Dict[str, Any]:
        """Analyze embedding properties"""
        if not embeddings:
            return {}
        
        # Convert to numpy array for analysis
        embedding_matrix = np.array(embeddings)
        
        analysis = {
            "count": len(embeddings),
            "dimension": embedding_matrix.shape[1],
            "average_norm": np.mean([np.linalg.norm(emb) for emb in embeddings]),
            "std_norm": np.std([np.linalg.norm(emb) for emb in embeddings]),
            "mean_values": np.mean(embedding_matrix, axis=0).tolist(),
            "std_values": np.std(embedding_matrix, axis=0).tolist(),
            "embedder_type": embedder_type
        }
        
        # Calculate pairwise similarities
        if len(embeddings) > 1:
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            analysis["average_similarity"] = np.mean(similarities)
            analysis["std_similarity"] = np.std(similarities)
            analysis["min_similarity"] = np.min(similarities)
            analysis["max_similarity"] = np.max(similarities)
        
        return analysis
    
    def _analyze_hybrid_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze hybrid pipeline results"""
        if not results:
            return {}
        
        analysis = {
            "total_results": len(results),
            "cached_results": sum(1 for r in results if r.get("cached", False)),
            "average_processing_time": np.mean([r["metadata"]["processing_time"] for r in results]),
            "components_used": set(),
            "fusion_methods": set(),
            "embedding_dimensions": []
        }
        
        for result in results:
            if "embeddings" in result:
                analysis["components_used"].update(result["embeddings"].keys())
            
            if "metadata" in result:
                fusion_method = result["metadata"].get("fusion_method", "unknown")
                analysis["fusion_methods"].add(fusion_method)
                
                embedding_dim = result["metadata"].get("embedding_dim", 0)
                analysis["embedding_dimensions"].append(embedding_dim)
        
        # Convert sets to lists for JSON serialization
        analysis["components_used"] = list(analysis["components_used"])
        analysis["fusion_methods"] = list(analysis["fusion_methods"])
        
        return analysis
    
    def generate_demo_report(self):
        """Generate comprehensive demo report"""
        logger.info("📊 Generating Demo Report")
        
        report = {
            "demo_info": {
                "timestamp": time.time(),
                "components_tested": list(self.results.keys()),
                "total_components": len(self.results)
            },
            "results": self.results,
            "summary": self._generate_summary()
        }
        
        # Save report to file
        report_file = "embedding_pipeline_demo_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Demo report saved to: {report_file}")
        
        # Print summary
        self._print_summary(report["summary"])
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of demo results"""
        summary = {
            "total_tests": len(self.results),
            "successful_tests": 0,
            "total_processing_time": 0.0,
            "embedding_types": [],
            "performance_metrics": {}
        }
        
        for component, result in self.results.items():
            if result:
                summary["successful_tests"] += 1
                summary["total_processing_time"] += result.get("processing_time", 0.0)
                
                if "analysis" in result and "embedder_type" in result["analysis"]:
                    summary["embedding_types"].append(result["analysis"]["embedder_type"])
                
                if "metrics" in result:
                    summary["performance_metrics"][component] = result["metrics"]
        
        summary["success_rate"] = summary["successful_tests"] / summary["total_tests"] if summary["total_tests"] > 0 else 0
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print demo summary"""
        print("\n" + "="*60)
        print("🎯 ADVANCED EMBEDDING PIPELINE DEMO SUMMARY")
        print("="*60)
        print(f"✅ Successful tests: {summary['successful_tests']}/{summary['total_tests']}")
        print(f"📈 Success rate: {summary['success_rate']:.1%}")
        print(f"⏱️  Total processing time: {summary['total_processing_time']:.2f}s")
        print(f"🔧 Embedding types tested: {', '.join(set(summary['embedding_types']))}")
        
        if summary["performance_metrics"]:
            print("\n📊 Performance Metrics:")
            for component, metrics in summary["performance_metrics"].items():
                print(f"   {component}: {metrics}")
        
        print("="*60)


async def main():
    """Main demo function"""
    demo = EmbeddingPipelineDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
