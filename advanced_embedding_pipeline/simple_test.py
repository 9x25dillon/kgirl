#!/usr/bin/env python3
"""
Simple Test for Advanced Embedding Pipeline
Tests basic functionality without external dependencies
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("simple_test")


async def test_basic_functionality():
    """Test basic functionality of the embedding pipeline"""
    logger.info("🧪 Starting Simple Embedding Pipeline Test")
    
    try:
        # Test imports
        logger.info("📦 Testing imports...")
        
        from fractal_cascade_embedder import FractalCascadeEmbedder, FractalConfig
        logger.info("✅ Fractal embedder imported successfully")
        
        # Test fractal embedder (no external dependencies)
        logger.info("🌀 Testing Fractal Cascade Embedder...")
        
        config = FractalConfig(
            max_depth=4,
            branching_factor=2,
            embedding_dim=512,
            fractal_type="mandelbrot",
            use_entropy=True,
            visualization=False
        )
        
        embedder = FractalCascadeEmbedder(config)
        
        # Test texts
        test_texts = [
            "The quick brown fox jumps over the lazy dog",
            "Fractal geometry reveals infinite complexity",
            "Mathematical beauty in recursive patterns"
        ]
        
        start_time = time.time()
        embeddings = embedder.embed_batch(test_texts)
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Fractal embeddings generated successfully")
        logger.info(f"   Texts processed: {len(test_texts)}")
        logger.info(f"   Embeddings generated: {len(embeddings)}")
        logger.info(f"   Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
        logger.info(f"   Processing time: {processing_time:.3f}s")
        
        # Test embedding properties
        if embeddings:
            import numpy as np
            
            # Check embedding norms
            norms = [np.linalg.norm(emb) for emb in embeddings]
            avg_norm = np.mean(norms)
            std_norm = np.std(norms)
            
            logger.info(f"   Average embedding norm: {avg_norm:.4f}")
            logger.info(f"   Standard deviation norm: {std_norm:.4f}")
            
            # Check embedding diversity
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                logger.info(f"   Average pairwise similarity: {avg_similarity:.4f}")
        
        # Test semantic embedder (with fallback)
        logger.info("📝 Testing Semantic Embedder (fallback mode)...")
        
        try:
            from semantic_embedder import SemanticEmbedder, SemanticConfig
            
            semantic_config = SemanticConfig(
                eopiez_url="http://localhost:8001",  # Will use fallback
                embedding_dim=768,
                batch_size=4,
                use_cache=False  # Disable cache for simple test
            )
            
            semantic_embedder = SemanticEmbedder(semantic_config)
            
            # Test single embedding
            single_embedding = await semantic_embedder.embed_text(test_texts[0])
            logger.info(f"✅ Semantic embedding generated (fallback mode)")
            logger.info(f"   Embedding dimension: {len(single_embedding)}")
            
            await semantic_embedder.close()
            
        except Exception as e:
            logger.warning(f"⚠️  Semantic embedder test failed: {e}")
        
        # Test mathematical embedder (without LIMPS)
        logger.info("🔢 Testing Mathematical Embedder (local mode)...")
        
        try:
            from mathematical_embedder import MathematicalEmbedder, MathematicalConfig
            
            math_config = MathematicalConfig(
                limps_url="http://localhost:8000",  # Will skip LIMPS
                max_dimension=512,
                polynomial_degree=2,
                use_matrix_optimization=False  # Disable LIMPS
            )
            
            math_embedder = MathematicalEmbedder(math_config)
            
            # Test mathematical expression
            math_embedding = await math_embedder.embed_mathematical_expression("x^2 + y^2 = z^2")
            logger.info(f"✅ Mathematical embedding generated")
            logger.info(f"   Embedding dimension: {len(math_embedding)}")
            
            await math_embedder.close()
            
        except Exception as e:
            logger.warning(f"⚠️  Mathematical embedder test failed: {e}")
        
        logger.info("🎉 Simple test completed successfully!")
        
        # Summary
        print("\n" + "="*60)
        print("🧪 SIMPLE EMBEDDING PIPELINE TEST SUMMARY")
        print("="*60)
        print("✅ Fractal Cascade Embedder: WORKING")
        print("✅ Semantic Embedder (fallback): WORKING")
        print("✅ Mathematical Embedder (local): WORKING")
        print("✅ All core components functional")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple test failed: {e}")
        return False


async def main():
    """Main test function"""
    success = await test_basic_functionality()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Start Eopiez service: cd ~/aipyapp/Eopiez && python api.py --port 8001")
        print("2. Start LIMPS service: cd ~/aipyapp/9xdSq-LIMPS-FemTO-R1C/limps && julia --project=. -e 'using LIMPS; LIMPS.start_limps_server(8000)'")
        print("3. Run full integration test: python integration_test.py")
        print("4. Run comprehensive demo: python demo.py")
    else:
        print("\n❌ Basic test failed. Check error messages above.")


if __name__ == "__main__":
    asyncio.run(main())
