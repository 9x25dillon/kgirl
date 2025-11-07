#!/usr/bin/env python3
"""
Setup script for Advanced Embedding Pipeline
Installs dependencies and configures the system
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_command(command, description):
    """Run a command and handle errors"""
    logger.info(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    logger.info("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8 or higher is required")
        return False
    
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True


def install_dependencies():
    """Install required dependencies"""
    logger.info("📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install core requirements
    if not run_command("pip install -r requirements.txt", "Installing core dependencies"):
        return False
    
    # Install optional dependencies if available
    optional_deps = [
        "faiss-gpu",  # GPU-accelerated FAISS
        "torch-gpu",  # GPU-accelerated PyTorch
    ]
    
    for dep in optional_deps:
        run_command(f"pip install {dep}", f"Installing optional dependency: {dep}")
    
    return True


def create_directories():
    """Create necessary directories"""
    logger.info("📁 Creating directories...")
    
    directories = [
        "cache/embeddings",
        "cache/optimized_embeddings", 
        "logs",
        "data",
        "models"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")
    
    return True


def setup_configuration():
    """Setup default configuration files"""
    logger.info("⚙️  Setting up configuration...")
    
    # Create default config file
    config_content = """# Advanced Embedding Pipeline Configuration

[services]
eopiez_url = "http://localhost:8001"
limps_url = "http://localhost:8000"

[semantic]
embedding_dim = 768
batch_size = 32
use_cache = true

[mathematical]
max_dimension = 1024
polynomial_degree = 3
use_matrix_optimization = true

[fractal]
max_depth = 6
branching_factor = 3
embedding_dim = 1024
fractal_type = "mandelbrot"
use_entropy = true

[hybrid]
fusion_method = "weighted_average"
semantic_weight = 0.4
mathematical_weight = 0.3
fractal_weight = 0.3
parallel_processing = true

[optimization]
use_disk_cache = true
batch_processing = true
max_batch_size = 64
adaptive_batching = true
use_indexing = true
index_type = "faiss"
"""
    
    config_file = Path("config.ini")
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    logger.info("✅ Created configuration file: config.ini")
    return True


def test_installation():
    """Test the installation"""
    logger.info("🧪 Testing installation...")
    
    try:
        # Test imports
        import numpy as np
        import scipy
        import sklearn
        import torch
        logger.info("✅ Core scientific libraries imported successfully")
        
        # Test our modules
        sys.path.insert(0, str(Path.cwd()))
        
        from semantic_embedder import SemanticEmbedder
        from mathematical_embedder import MathematicalEmbedder
        from fractal_cascade_embedder import FractalCascadeEmbedder
        from hybrid_pipeline import HybridEmbeddingPipeline
        from optimizer import EmbeddingOptimizer
        
        logger.info("✅ All embedding pipeline modules imported successfully")
        
        # Test basic functionality
        import asyncio
        
        async def test_basic_functionality():
            # Test semantic embedder
            semantic_embedder = SemanticEmbedder()
            test_embedding = await semantic_embedder.embed_text("Test text")
            assert len(test_embedding) > 0
            logger.info("✅ Semantic embedder test passed")
            
            await semantic_embedder.close()
            
            # Test fractal embedder
            fractal_embedder = FractalCascadeEmbedder()
            fractal_embedding = fractal_embedder.embed_text_with_fractal("Test fractal")
            assert len(fractal_embedding) > 0
            logger.info("✅ Fractal embedder test passed")
            
            return True
        
        # Run async test
        asyncio.run(test_basic_functionality())
        
        logger.info("✅ All basic functionality tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Installation test failed: {e}")
        return False


def check_external_services():
    """Check if external services are available"""
    logger.info("🔍 Checking external services...")
    
    import httpx
    import asyncio
    
    async def check_services():
        services = [
            ("Eopiez", "http://localhost:8001/health"),
            ("LIMPS", "http://localhost:8000/health")
        ]
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for service_name, url in services:
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        logger.info(f"✅ {service_name} service is available")
                    else:
                        logger.warning(f"⚠️  {service_name} service responded with status {response.status_code}")
                except Exception as e:
                    logger.warning(f"⚠️  {service_name} service is not available: {e}")
    
    try:
        asyncio.run(check_services())
    except Exception as e:
        logger.warning(f"⚠️  Service check failed: {e}")
    
    return True


def create_example_scripts():
    """Create example usage scripts"""
    logger.info("📝 Creating example scripts...")
    
    # Simple usage example
    simple_example = """#!/usr/bin/env python3
'''
Simple usage example for Advanced Embedding Pipeline
'''

import asyncio
from advanced_embedding_pipeline import HybridEmbeddingPipeline, HybridConfig

async def main():
    # Configure pipeline
    config = HybridConfig(
        use_semantic=True,
        use_mathematical=True,
        use_fractal=True,
        fusion_method="weighted_average"
    )
    
    # Create pipeline
    pipeline = HybridEmbeddingPipeline(config)
    
    # Example texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "x^2 + y^2 = z^2",
        "Fractal geometry reveals infinite complexity"
    ]
    
    # Generate embeddings
    print("🚀 Generating embeddings...")
    results = await pipeline.embed_batch(texts)
    
    # Display results
    for i, result in enumerate(results):
        print(f"\\nText {i+1}: {result['text']}")
        print(f"Embedding dimension: {len(result['fused_embedding'])}")
        print(f"Processing time: {result['metadata']['processing_time']:.3f}s")
    
    # Get metrics
    metrics = pipeline.get_metrics()
    print(f"\\n📊 Metrics:")
    print(f"Total embeddings: {metrics['total_embeddings']}")
    print(f"Average time: {metrics['average_time']:.3f}s")
    
    # Cleanup
    await pipeline.close()
    print("\\n✅ Example completed!")

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    with open("example_simple.py", 'w') as f:
        f.write(simple_example)
    
    logger.info("✅ Created example_simple.py")
    
    # Advanced usage example
    advanced_example = """#!/usr/bin/env python3
'''
Advanced usage example with optimization and indexing
'''

import asyncio
import numpy as np
from advanced_embedding_pipeline import (
    HybridEmbeddingPipeline, HybridConfig,
    EmbeddingOptimizer, OptimizationConfig
)

async def main():
    # Configure pipeline with optimization
    hybrid_config = HybridConfig(
        use_semantic=True,
        use_mathematical=True,
        use_fractal=True,
        fusion_method="attention",
        parallel_processing=True
    )
    
    optimization_config = OptimizationConfig(
        use_disk_cache=True,
        batch_processing=True,
        adaptive_batching=True,
        use_indexing=True,
        index_type="faiss"
    )
    
    # Create components
    pipeline = HybridEmbeddingPipeline(hybrid_config)
    optimizer = EmbeddingOptimizer(optimization_config)
    
    # Large corpus of texts
    texts = [
        "Mathematical formula: E = mc²",
        "Code: def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Natural language: The theory of relativity revolutionized physics",
        "Fractal: The Mandelbrot set exhibits self-similarity at all scales",
        "Scientific: Quantum mechanics describes atomic behavior",
        "Programming: Neural networks learn through backpropagation",
        "Physics: Schrödinger equation: iℏ∂ψ/∂t = Ĥψ",
        "Mathematics: Fractal dimension D = log(N)/log(r)",
        "AI: Machine learning algorithms optimize objective functions",
        "Geometry: Sierpinski triangle shows recursive patterns"
    ]
    
    print("🚀 Processing large corpus with optimization...")
    
    # Optimized embedding generation
    async def embedder_func(texts_batch):
        return await pipeline.embed_batch(texts_batch)
    
    results = await optimizer.optimize_embedding_generation(
        embedder_func, texts, "advanced_demo"
    )
    
    # Create search index
    embeddings = [result['fused_embedding'] for result in results]
    index_data = optimizer.create_index(embeddings, texts)
    
    if index_data['index']:
        print(f"✅ Created {index_data['type']} index with {index_data['size']} vectors")
        
        # Test similarity search
        query_embedding = embeddings[0]
        search_results = optimizer.search_similar(index_data, query_embedding, top_k=5)
        
        print("\\n🔍 Similarity search results:")
        for i, (idx, score) in enumerate(search_results):
            print(f"{i+1}. {texts[idx]} (similarity: {score:.4f})")
    
    # Performance report
    performance_report = optimizer.get_performance_report()
    print(f"\\n📊 Performance Report:")
    print(f"Cache hit rate: {performance_report['cache_stats']['hit_rate']:.2%}")
    print(f"Average processing time: {performance_report['performance_metrics']['average_processing_time']:.3f}s")
    print(f"Total embeddings: {performance_report['performance_metrics']['total_embeddings']}")
    
    # Cleanup
    await pipeline.close()
    print("\\n✅ Advanced example completed!")

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    with open("example_advanced.py", 'w') as f:
        f.write(advanced_example)
    
    logger.info("✅ Created example_advanced.py")
    
    return True


def main():
    """Main setup function"""
    logger.info("🚀 Starting Advanced Embedding Pipeline Setup")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        logger.error("❌ Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        logger.error("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup configuration
    if not setup_configuration():
        logger.error("❌ Failed to setup configuration")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        logger.error("❌ Installation test failed")
        sys.exit(1)
    
    # Check external services
    check_external_services()
    
    # Create example scripts
    if not create_example_scripts():
        logger.error("❌ Failed to create example scripts")
        sys.exit(1)
    
    logger.info("🎉 Setup completed successfully!")
    logger.info("")
    logger.info("📋 Next steps:")
    logger.info("1. Run the demo: python demo.py")
    logger.info("2. Try the simple example: python example_simple.py")
    logger.info("3. Try the advanced example: python example_advanced.py")
    logger.info("4. Start your Eopiez service: cd ~/aipyapp/Eopiez && python api.py --port 8001")
    logger.info("5. Start your LIMPS service: cd ~/aipyapp/9xdSq-LIMPS-FemTO-R1C/limps && julia --project=. -e 'using LIMPS; LIMPS.start_limps_server(8000)'")
    logger.info("")
    logger.info("🔧 Configuration file: config.ini")
    logger.info("📚 Documentation: README.md")
    logger.info("🧪 Demo script: demo.py")


if __name__ == "__main__":
    main()
