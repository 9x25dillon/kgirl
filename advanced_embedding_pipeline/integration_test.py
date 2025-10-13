#!/usr/bin/env python3
"""
Integration Test for Advanced Embedding Pipeline
Tests the complete system integration with existing services
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from semantic_embedder import SemanticEmbedder, SemanticConfig
from mathematical_embedder import MathematicalEmbedder, MathematicalConfig
from fractal_cascade_embedder import FractalCascadeEmbedder, FractalConfig
from hybrid_pipeline import HybridEmbeddingPipeline, HybridConfig
from optimizer import EmbeddingOptimizer, OptimizationConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("integration_test")


class IntegrationTester:
    """Integration tester for the advanced embedding pipeline"""
    
    def __init__(self):
        self.test_results = {}
        self.services_status = {}
    
    async def run_full_integration_test(self):
        """Run comprehensive integration tests"""
        logger.info("🧪 Starting Advanced Embedding Pipeline Integration Tests")
        
        # Check service availability
        await self.check_services()
        
        # Test individual components
        await self.test_semantic_embedder()
        await self.test_mathematical_embedder()
        await self.test_fractal_embedder()
        await self.test_hybrid_pipeline()
        await self.test_optimization_system()
        
        # Test with your existing data
        await self.test_with_existing_data()
        
        # Generate integration report
        self.generate_integration_report()
        
        logger.info("✅ Integration tests completed!")
    
    async def check_services(self):
        """Check availability of external services"""
        logger.info("🔍 Checking external services...")
        
        import httpx
        
        services = {
            "Eopiez": "http://localhost:8001/health",
            "LIMPS": "http://localhost:8000/health",
            "Knowledge Base API": "http://localhost:8888/health",
            "PostgreSQL": "postgresql://limps:limps@localhost:5432/limps"
        }
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for service_name, endpoint in services.items():
                try:
                    if service_name == "PostgreSQL":
                        # Test database connection
                        import asyncpg
                        conn = await asyncpg.connect(endpoint)
                        await conn.close()
                        self.services_status[service_name] = "available"
                        logger.info(f"✅ {service_name} is available")
                    else:
                        response = await client.get(endpoint)
                        if response.status_code == 200:
                            self.services_status[service_name] = "available"
                            logger.info(f"✅ {service_name} is available")
                        else:
                            self.services_status[service_name] = f"unavailable (status: {response.status_code})"
                            logger.warning(f"⚠️  {service_name} responded with status {response.status_code}")
                except Exception as e:
                    self.services_status[service_name] = f"unavailable: {str(e)}"
                    logger.warning(f"⚠️  {service_name} is not available: {e}")
    
    async def test_semantic_embedder(self):
        """Test semantic embedder integration"""
        logger.info("📝 Testing Semantic Embedder...")
        
        try:
            config = SemanticConfig(
                eopiez_url="http://localhost:8001",
                embedding_dim=768,
                batch_size=8,
                use_cache=True
            )
            
            embedder = SemanticEmbedder(config)
            
            # Test texts
            test_texts = [
                "The quick brown fox jumps over the lazy dog",
                "Artificial intelligence and machine learning",
                "Natural language processing with transformers"
            ]
            
            start_time = time.time()
            
            # Test single embedding
            single_embedding = await embedder.embed_text(test_texts[0])
            
            # Test batch embedding
            batch_embeddings = await embedder.embed_batch(test_texts)
            
            processing_time = time.time() - start_time
            
            # Test similarity search
            similarities = await embedder.query_similar(
                test_texts[0], batch_embeddings, top_k=3
            )
            
            # Test semantic features
            features = await embedder.extract_semantic_features(test_texts[0])
            
            self.test_results["semantic_embedder"] = {
                "status": "success",
                "single_embedding_dim": len(single_embedding),
                "batch_embeddings_count": len(batch_embeddings),
                "processing_time": processing_time,
                "similarities_found": len(similarities),
                "features_extracted": len(features),
                "eopiez_available": self.services_status.get("Eopiez") == "available"
            }
            
            logger.info(f"✅ Semantic embedder test passed")
            logger.info(f"   Single embedding dimension: {len(single_embedding)}")
            logger.info(f"   Batch embeddings: {len(batch_embeddings)}")
            logger.info(f"   Processing time: {processing_time:.3f}s")
            
            await embedder.close()
            
        except Exception as e:
            self.test_results["semantic_embedder"] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Semantic embedder test failed: {e}")
    
    async def test_mathematical_embedder(self):
        """Test mathematical embedder integration"""
        logger.info("🔢 Testing Mathematical Embedder...")
        
        try:
            config = MathematicalConfig(
                limps_url="http://localhost:8000",
                max_dimension=1024,
                polynomial_degree=3,
                use_matrix_optimization=self.services_status.get("LIMPS") == "available"
            )
            
            embedder = MathematicalEmbedder(config)
            
            # Test mathematical expressions
            test_expressions = [
                "x^2 + y^2 = z^2",
                "sin(x) * cos(y) + e^(i*pi)",
                "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
            ]
            
            start_time = time.time()
            
            embeddings = []
            for expr in test_expressions:
                embedding = await embedder.embed_mathematical_expression(expr)
                embeddings.append(embedding)
            
            # Test code AST embedding
            code_embedding = await embedder.embed_code_ast(test_expressions[2])
            
            # Test system of equations
            equations = ["x + y = 5", "2x - y = 1"]
            system_embedding = await embedder.embed_system_of_equations(equations)
            
            processing_time = time.time() - start_time
            
            self.test_results["mathematical_embedder"] = {
                "status": "success",
                "expression_embeddings": len(embeddings),
                "code_embedding_dim": len(code_embedding),
                "system_embedding_dim": len(system_embedding),
                "processing_time": processing_time,
                "limps_available": self.services_status.get("LIMPS") == "available"
            }
            
            logger.info(f"✅ Mathematical embedder test passed")
            logger.info(f"   Expression embeddings: {len(embeddings)}")
            logger.info(f"   Code embedding dimension: {len(code_embedding)}")
            logger.info(f"   Processing time: {processing_time:.3f}s")
            
            await embedder.close()
            
        except Exception as e:
            self.test_results["mathematical_embedder"] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Mathematical embedder test failed: {e}")
    
    async def test_fractal_embedder(self):
        """Test fractal cascade embedder"""
        logger.info("🌀 Testing Fractal Cascade Embedder...")
        
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
            
            # Test texts with different characteristics
            test_texts = [
                "Fractal geometry reveals infinite complexity",
                "The Mandelbrot set exhibits self-similarity",
                "Mathematical beauty in recursive patterns"
            ]
            
            start_time = time.time()
            
            embeddings = embedder.embed_batch(test_texts)
            
            processing_time = time.time() - start_time
            
            self.test_results["fractal_embedder"] = {
                "status": "success",
                "embeddings_count": len(embeddings),
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                "processing_time": processing_time
            }
            
            logger.info(f"✅ Fractal embedder test passed")
            logger.info(f"   Embeddings generated: {len(embeddings)}")
            logger.info(f"   Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
            logger.info(f"   Processing time: {processing_time:.3f}s")
            
        except Exception as e:
            self.test_results["fractal_embedder"] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Fractal embedder test failed: {e}")
    
    async def test_hybrid_pipeline(self):
        """Test hybrid embedding pipeline"""
        logger.info("🔗 Testing Hybrid Embedding Pipeline...")
        
        try:
            # Configure components
            semantic_config = SemanticConfig(
                eopiez_url="http://localhost:8001",
                embedding_dim=512,
                batch_size=4
            )
            
            mathematical_config = MathematicalConfig(
                limps_url="http://localhost:8000",
                max_dimension=512,
                use_matrix_optimization=False  # Disable for faster testing
            )
            
            fractal_config = FractalConfig(
                embedding_dim=512,
                max_depth=4
            )
            
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
            
            # Test texts covering different domains
            test_texts = [
                "The quick brown fox jumps over the lazy dog",
                "x^2 + y^2 = z^2",
                "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "Fractal geometry reveals infinite complexity in finite space",
                "Neural networks learn hierarchical representations"
            ]
            
            start_time = time.time()
            
            # Test single embedding
            single_result = await pipeline.embed(test_texts[0])
            
            # Test batch embedding
            batch_results = await pipeline.embed_batch(test_texts)
            
            processing_time = time.time() - start_time
            
            # Get metrics
            metrics = pipeline.get_metrics()
            
            self.test_results["hybrid_pipeline"] = {
                "status": "success",
                "single_result_components": len(single_result.get("embeddings", {})),
                "batch_results_count": len(batch_results),
                "processing_time": processing_time,
                "metrics": metrics,
                "fusion_method": single_result.get("metadata", {}).get("fusion_method", "unknown")
            }
            
            logger.info(f"✅ Hybrid pipeline test passed")
            logger.info(f"   Single result components: {len(single_result.get('embeddings', {}))}")
            logger.info(f"   Batch results: {len(batch_results)}")
            logger.info(f"   Processing time: {processing_time:.3f}s")
            logger.info(f"   Fusion method: {single_result.get('metadata', {}).get('fusion_method', 'unknown')}")
            
            await pipeline.close()
            
        except Exception as e:
            self.test_results["hybrid_pipeline"] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Hybrid pipeline test failed: {e}")
    
    async def test_optimization_system(self):
        """Test optimization and caching system"""
        logger.info("⚡ Testing Optimization System...")
        
        try:
            config = OptimizationConfig(
                use_disk_cache=True,
                cache_directory="./cache/test_embeddings",
                batch_processing=True,
                max_batch_size=8,
                adaptive_batching=True,
                use_indexing=True,
                index_type="faiss"
            )
            
            optimizer = EmbeddingOptimizer(config)
            
            # Create a simple embedder function for testing
            async def test_embedder(texts_batch):
                # Simple hash-based embedding for testing
                import hashlib
                embeddings = []
                for text in texts_batch:
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
            
            test_texts = [
                "Test text 1",
                "Test text 2", 
                "Test text 3",
                "Test text 4",
                "Test text 5"
            ]
            
            start_time = time.time()
            
            # First run (no cache)
            results1 = await optimizer.optimize_embedding_generation(
                test_embedder, test_texts, "test_config"
            )
            
            # Second run (with cache)
            results2 = await optimizer.optimize_embedding_generation(
                test_embedder, test_texts, "test_config"
            )
            
            processing_time = time.time() - start_time
            
            # Create index
            embeddings = [result["embedding"] for result in results1]
            index_data = optimizer.create_index(embeddings, test_texts)
            
            # Test search
            search_results = []
            if index_data["index"]:
                query_embedding = embeddings[0]
                search_results = optimizer.search_similar(index_data, query_embedding, top_k=3)
            
            # Get performance report
            performance_report = optimizer.get_performance_report()
            
            self.test_results["optimization_system"] = {
                "status": "success",
                "first_run_results": len(results1),
                "second_run_results": len(results2),
                "processing_time": processing_time,
                "cache_stats": optimizer.cache.get_stats(),
                "index_created": index_data["type"] != "none",
                "search_results_count": len(search_results),
                "performance_metrics": performance_report["performance_metrics"]
            }
            
            logger.info(f"✅ Optimization system test passed")
            logger.info(f"   First run results: {len(results1)}")
            logger.info(f"   Second run results: {len(results2)}")
            logger.info(f"   Processing time: {processing_time:.3f}s")
            logger.info(f"   Cache hit rate: {optimizer.cache.get_stats()['hit_rate']:.2%}")
            logger.info(f"   Index created: {index_data['type'] != 'none'}")
            
        except Exception as e:
            self.test_results["optimization_system"] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Optimization system test failed: {e}")
    
    async def test_with_existing_data(self):
        """Test with your existing document data"""
        logger.info("📚 Testing with existing document data...")
        
        try:
            # Try to connect to your existing database
            import asyncpg
            
            DATABASE_URL = "postgresql://limps:limps@localhost:5432/limps"
            
            conn = await asyncpg.connect(DATABASE_URL)
            
            # Get some sample documents
            documents = await conn.fetch("""
                SELECT d.file_name, dc.content 
                FROM documents d
                JOIN document_content dc ON d.id = dc.document_id
                WHERE dc.content IS NOT NULL AND LENGTH(dc.content) > 100
                LIMIT 5
            """)
            
            await conn.close()
            
            if not documents:
                logger.warning("⚠️  No documents found in database")
                self.test_results["existing_data"] = {
                    "status": "skipped",
                    "reason": "No documents found"
                }
                return
            
            # Test with real document data
            hybrid_config = HybridConfig(
                use_semantic=True,
                use_mathematical=True,
                use_fractal=True,
                fusion_method="weighted_average",
                parallel_processing=True
            )
            
            pipeline = HybridEmbeddingPipeline(hybrid_config)
            
            # Extract texts from documents
            texts = [doc['content'][:500] for doc in documents]  # Limit to 500 chars
            
            start_time = time.time()
            results = await pipeline.embed_batch(texts)
            processing_time = time.time() - start_time
            
            # Analyze results
            successful_results = [r for r in results if 'error' not in r]
            
            self.test_results["existing_data"] = {
                "status": "success",
                "documents_processed": len(documents),
                "successful_embeddings": len(successful_results),
                "processing_time": processing_time,
                "average_time_per_doc": processing_time / len(documents)
            }
            
            logger.info(f"✅ Existing data test passed")
            logger.info(f"   Documents processed: {len(documents)}")
            logger.info(f"   Successful embeddings: {len(successful_results)}")
            logger.info(f"   Processing time: {processing_time:.3f}s")
            logger.info(f"   Average time per document: {processing_time/len(documents):.3f}s")
            
            await pipeline.close()
            
        except Exception as e:
            self.test_results["existing_data"] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Existing data test failed: {e}")
    
    def generate_integration_report(self):
        """Generate comprehensive integration test report"""
        logger.info("📊 Generating Integration Test Report")
        
        import json
        
        # Calculate overall success rate
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get("status") == "success")
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        report = {
            "integration_test_info": {
                "timestamp": time.time(),
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate
            },
            "services_status": self.services_status,
            "test_results": self.test_results,
            "summary": {
                "all_services_available": all(status == "available" for status in self.services_status.values()),
                "all_tests_passed": success_rate == 1.0,
                "recommendations": self._generate_recommendations()
            }
        }
        
        # Save report
        report_file = "integration_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Integration test report saved to: {report_file}")
        
        # Print summary
        self._print_integration_summary(report)
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check service availability
        unavailable_services = [name for name, status in self.services_status.items() if status != "available"]
        if unavailable_services:
            recommendations.append(f"Start unavailable services: {', '.join(unavailable_services)}")
        
        # Check test failures
        failed_tests = [name for name, result in self.test_results.items() if result.get("status") == "failed"]
        if failed_tests:
            recommendations.append(f"Investigate failed tests: {', '.join(failed_tests)}")
        
        # Performance recommendations
        if "hybrid_pipeline" in self.test_results and self.test_results["hybrid_pipeline"].get("status") == "success":
            processing_time = self.test_results["hybrid_pipeline"].get("processing_time", 0)
            if processing_time > 10:
                recommendations.append("Consider optimizing hybrid pipeline performance")
        
        if not recommendations:
            recommendations.append("All systems are working optimally!")
        
        return recommendations
    
    def _print_integration_summary(self, report):
        """Print integration test summary"""
        print("\n" + "="*70)
        print("🧪 ADVANCED EMBEDDING PIPELINE INTEGRATION TEST SUMMARY")
        print("="*70)
        
        # Services status
        print("🔍 Services Status:")
        for service, status in self.services_status.items():
            status_icon = "✅" if status == "available" else "❌"
            print(f"   {status_icon} {service}: {status}")
        
        # Test results
        print(f"\n📊 Test Results:")
        print(f"   Total tests: {report['integration_test_info']['total_tests']}")
        print(f"   Successful: {report['integration_test_info']['successful_tests']}")
        print(f"   Success rate: {report['integration_test_info']['success_rate']:.1%}")
        
        # Individual test results
        print(f"\n🔬 Individual Test Results:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result.get("status") == "success" else "❌"
            print(f"   {status_icon} {test_name}: {result.get('status', 'unknown')}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for recommendation in report["summary"]["recommendations"]:
            print(f"   • {recommendation}")
        
        print("="*70)


async def main():
    """Main integration test function"""
    tester = IntegrationTester()
    await tester.run_full_integration_test()


if __name__ == "__main__":
    asyncio.run(main())
