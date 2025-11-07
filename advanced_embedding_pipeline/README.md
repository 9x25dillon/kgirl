# Advanced Embedding Pipeline - Fractal Cascade Simulation

A sophisticated multi-modal embedding system that integrates semantic vectorization, mathematical optimization, and fractal-based embedding generation for advanced AI applications.

## 🌟 Features

### Core Components

- **Semantic Embedder**: Advanced semantic vectorization using Eopiez integration
- **Mathematical Embedder**: Symbolic and mathematical expression processing with LIMPS optimization
- **Fractal Cascade Embedder**: Fractal-based embedding generation with hierarchical structures
- **Hybrid Pipeline**: Unified orchestration combining all embedding methods
- **Embedding Optimizer**: Performance optimization, caching, and indexing

### Advanced Capabilities

- **Multi-Modal Fusion**: Weighted averaging, concatenation, and attention-based fusion
- **Parallel Processing**: Concurrent embedding generation with adaptive batching
- **Intelligent Caching**: Memory and disk-based caching with compression
- **Vector Indexing**: FAISS, Annoy, and HNSWlib integration for similarity search
- **Performance Monitoring**: Real-time metrics and adaptive optimization
- **Fractal Mathematics**: Mandelbrot, Julia, and Sierpinski fractal structures

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the pipeline directory
cd /home/kill/aipyapp/Fractal_cascade_simulation/advanced_embedding_pipeline

# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo.py
```

### Basic Usage

```python
import asyncio
from advanced_embedding_pipeline import HybridEmbeddingPipeline, HybridConfig

async def main():
    # Configure the pipeline
    config = HybridConfig(
        use_semantic=True,
        use_mathematical=True,
        use_fractal=True,
        fusion_method="weighted_average"
    )
    
    # Create pipeline
    pipeline = HybridEmbeddingPipeline(config)
    
    # Generate embeddings
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "x^2 + y^2 = z^2",
        "Fractal geometry reveals infinite complexity"
    ]
    
    results = await pipeline.embed_batch(texts)
    
    # Process results
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Embedding dimension: {len(result['fused_embedding'])}")
        print(f"Processing time: {result['metadata']['processing_time']:.3f}s")
    
    # Cleanup
    await pipeline.close()

# Run the example
asyncio.run(main())
```

## 🔧 Configuration

### Hybrid Pipeline Configuration

```python
from advanced_embedding_pipeline import HybridConfig, SemanticConfig, MathematicalConfig, FractalConfig

# Component configurations
semantic_config = SemanticConfig(
    eopiez_url="http://localhost:8001",
    embedding_dim=768,
    batch_size=32,
    use_cache=True
)

mathematical_config = MathematicalConfig(
    limps_url="http://localhost:8000",
    max_dimension=1024,
    polynomial_degree=3,
    use_matrix_optimization=True
)

fractal_config = FractalConfig(
    max_depth=6,
    branching_factor=3,
    embedding_dim=1024,
    fractal_type="mandelbrot",
    use_entropy=True
)

# Hybrid pipeline configuration
hybrid_config = HybridConfig(
    semantic_config=semantic_config,
    mathematical_config=mathematical_config,
    fractal_config=fractal_config,
    use_semantic=True,
    use_mathematical=True,
    use_fractal=True,
    fusion_method="weighted_average",  # or "concatenation", "attention"
    semantic_weight=0.4,
    mathematical_weight=0.3,
    fractal_weight=0.3,
    parallel_processing=True
)
```

### Optimization Configuration

```python
from advanced_embedding_pipeline import OptimizationConfig

optimization_config = OptimizationConfig(
    use_disk_cache=True,
    cache_directory="./cache/embeddings",
    batch_processing=True,
    max_batch_size=64,
    adaptive_batching=True,
    use_indexing=True,
    index_type="faiss",  # or "annoy", "hnswlib"
    performance_monitoring=True
)
```

## 📊 Component Details

### Semantic Embedder

- **Purpose**: Generate semantic embeddings using Eopiez vectorization
- **Features**: Batch processing, caching, fallback generation
- **Integration**: HTTP-based communication with Eopiez service
- **Fallback**: Hash-based embedding when service unavailable

### Mathematical Embedder

- **Purpose**: Process mathematical expressions and symbolic content
- **Features**: SymPy integration, LIMPS optimization, code AST analysis
- **Capabilities**: Polynomial processing, system of equations, code analysis
- **Optimization**: Matrix-based optimization via LIMPS service

### Fractal Cascade Embedder

- **Purpose**: Generate embeddings based on fractal structures
- **Fractal Types**: Mandelbrot, Julia, Sierpinski, custom patterns
- **Features**: Entropy-based modifications, hierarchical structures
- **Visualization**: Optional fractal structure visualization

### Hybrid Pipeline

- **Purpose**: Orchestrate multiple embedding methods
- **Fusion Methods**:
  - **Weighted Average**: Combine embeddings with configurable weights
  - **Concatenation**: Concatenate all embedding vectors
  - **Attention**: Attention-based fusion with similarity scoring
- **Features**: Parallel processing, caching, performance monitoring

### Embedding Optimizer

- **Purpose**: Performance optimization and advanced caching
- **Features**:
  - Disk-based caching with compression
  - Adaptive batch sizing
  - Vector indexing (FAISS, Annoy, HNSWlib)
  - Performance monitoring and auto-tuning
- **Indexing**: Similarity search with configurable algorithms

## 🎯 Use Cases

### 1. Multi-Modal Document Processing

```python
# Process documents with mixed content
documents = [
    "Mathematical formula: E = mc²",
    "Code snippet: def fibonacci(n): ...",
    "Natural language: The theory of relativity..."
]

# Generate hybrid embeddings
results = await pipeline.embed_batch(documents)
```

### 2. Scientific Text Analysis

```python
# Analyze scientific papers with mathematical content
scientific_texts = [
    "The Schrödinger equation: iℏ∂ψ/∂t = Ĥψ",
    "Fractal dimension: D = log(N)/log(r)",
    "Neural network: y = σ(Wx + b)"
]
```

### 3. Code Semantic Analysis

```python
# Embed code with mathematical understanding
code_snippets = [
    "def matrix_multiply(A, B): return A @ B",
    "for i in range(len(data)): process(data[i])",
    "if x > threshold: return sigmoid(x)"
]
```

## 🔍 Advanced Features

### Similarity Search

```python
# Create index for similarity search
embeddings = [result['fused_embedding'] for result in results]
index_data = optimizer.create_index(embeddings, texts)

# Search for similar content
query_embedding = results[0]['fused_embedding']
similar_items = optimizer.search_similar(index_data, query_embedding, top_k=5)
```

### Performance Monitoring

```python
# Get performance metrics
metrics = pipeline.get_metrics()
print(f"Total embeddings: {metrics['total_embeddings']}")
print(f"Cache hit rate: {metrics['cache_hits'] / metrics['total_embeddings']:.2%}")

# Get optimization report
optimization_report = optimizer.get_performance_report()
print(f"Average processing time: {optimization_report['performance_metrics']['average_processing_time']:.3f}s")
```

### Custom Fractal Structures

```python
# Generate custom fractal-based embeddings
fractal_config = FractalConfig(
    fractal_type="custom",
    max_depth=8,
    branching_factor=4
)

fractal_embedder = FractalCascadeEmbedder(fractal_config)
embedding = fractal_embedder.embed_text_with_fractal("Custom fractal text")
```

## 🛠️ Integration with Existing Systems

### Eopiez Integration

The pipeline integrates with your existing Eopiez service:

```python
semantic_config = SemanticConfig(
    eopiez_url="http://localhost:8001"  # Your Eopiez service
)
```

### LIMPS Integration

Mathematical optimization via LIMPS:

```python
mathematical_config = MathematicalConfig(
    limps_url="http://localhost:8000"  # Your LIMPS service
)
```

### Database Integration

Store and retrieve embeddings:

```python
# Store embeddings in PostgreSQL with pgvector
import asyncpg

async def store_embeddings(results):
    conn = await asyncpg.connect("postgresql://user:pass@localhost/db")
    
    for result in results:
        await conn.execute(
            "INSERT INTO embeddings (text, embedding_vector, metadata) VALUES ($1, $2, $3)",
            result['text'],
            result['fused_embedding'].tobytes(),
            json.dumps(result['metadata'])
        )
```

## 📈 Performance Optimization

### Batch Processing

```python
# Process large batches efficiently
large_text_corpus = [...]  # Thousands of texts
results = await pipeline.embed_batch(large_text_corpus)
```

### Caching Strategy

```python
# Enable aggressive caching for repeated processing
optimization_config = OptimizationConfig(
    use_disk_cache=True,
    max_cache_size_mb=2000,
    cache_compression=True
)
```

### Memory Management

```python
# Monitor and manage memory usage
import psutil

def check_memory_usage():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
```

## 🧪 Testing and Validation

### Run Comprehensive Demo

```bash
python demo.py
```

The demo will test all components and generate a detailed report.

### Unit Testing

```bash
pytest tests/
```

### Performance Benchmarking

```python
import time

async def benchmark_pipeline(texts):
    start_time = time.time()
    results = await pipeline.embed_batch(texts)
    processing_time = time.time() - start_time
    
    print(f"Processed {len(texts)} texts in {processing_time:.2f}s")
    print(f"Average time per text: {processing_time/len(texts):.3f}s")
```

## 🔧 Troubleshooting

### Common Issues

1. **Eopiez Service Unavailable**
   - The semantic embedder will fall back to hash-based embeddings
   - Check Eopiez service status and URL configuration

2. **LIMPS Service Unavailable**
   - Mathematical embedder will skip optimization
   - Mathematical processing will still work with SymPy

3. **Memory Issues**
   - Reduce batch size in configuration
   - Enable disk caching
   - Use memory mapping for large datasets

4. **Performance Issues**
   - Enable parallel processing
   - Use adaptive batching
   - Monitor cache hit rates

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
pipeline = HybridEmbeddingPipeline(config)
```

## 📚 API Reference

### HybridEmbeddingPipeline

- `embed(text: str) -> Dict[str, Any]`: Generate embedding for single text
- `embed_batch(texts: List[str]) -> List[Dict[str, Any]]`: Generate embeddings for batch
- `get_metrics() -> Dict[str, Any]`: Get performance metrics
- `clear_cache()`: Clear embedding cache
- `close()`: Close all embedders

### EmbeddingOptimizer

- `optimize_embedding_generation(embedder_func, texts, config_hash)`: Optimized embedding generation
- `create_index(embeddings, texts) -> Dict[str, Any]`: Create search index
- `search_similar(index_data, query_embedding, top_k) -> List[Tuple[int, float]]`: Similarity search
- `get_performance_report() -> Dict[str, Any]`: Comprehensive performance report

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is part of the Fractal Cascade Simulation system. See the main project license for details.

## 🙏 Acknowledgments

- Eopiez team for vectorization services
- LIMPS team for mathematical optimization
- Fractal mathematics research community
- Open source embedding and ML libraries

---

**Advanced Embedding Pipeline** - Bringing together semantic understanding, mathematical precision, and fractal beauty in AI embeddings.
