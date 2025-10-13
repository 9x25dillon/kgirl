# LFM2-8B-A1B + Numbskull + Dual LLM Integration

Complete integration guide for the unified workflow combining LFM2-8B-A1B local inference, Numbskull hybrid embeddings, and dual LLM orchestration.

## Overview

This integration creates a sophisticated AI workflow that:

1. **Generates Rich Embeddings** - Uses Numbskull's hybrid pipeline (semantic, mathematical, fractal)
2. **Summarizes Resources** - Remote LLM or local fallback for context summarization
3. **Final Inference** - LFM2-8B-A1B provides the final answer based on enriched context

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Query + Resources                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Numbskull Hybrid Pipeline                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Semantic   │  │ Mathematical │  │   Fractal    │         │
│  │  Embeddings  │  │  Embeddings  │  │  Embeddings  │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         └──────────────────┴──────────────────┘                 │
│                            │                                     │
│                  Fusion (weighted/concat/attention)             │
│                            │                                     │
│                    Hybrid Embedding Vector                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Resource LLM (Optional Remote)                     │
│        Summarizes context with embedding awareness              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LFM2-8B-A1B (Local LLM)                       │
│              Final inference with enriched context              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                       Final Answer
```

## Installation

### 1. Prerequisites

Ensure you have Python 3.8+ and the following services available:

- **LFM2-8B-A1B**: Local LLM server (llama.cpp, text-generation-webui, or compatible)
- **Eopiez** (optional): Semantic embedding service on port 8001
- **LIMPS** (optional): Mathematical optimization service on port 8000
- **Numbskull**: Embedding pipeline at `/home/kill/numbskull`

### 2. Install Dependencies

```bash
cd /home/kill/LiMp

# Install requirements including numbskull
pip install -r requirements.txt

# Or manually install numbskull as editable
pip install -e /home/kill/numbskull
```

### 3. Verify Numbskull Installation

```bash
python -c "from advanced_embedding_pipeline import HybridEmbeddingPipeline; print('✅ Numbskull available')"
```

## Configuration

### LFM2-8B-A1B Server Setup

The integration supports multiple backend modes. Choose one:

#### Option 1: llama.cpp Server (Recommended)

```bash
# Start llama-server with LFM2-8B-A1B model
llama-server \
  --model /path/to/LFM2-8B-A1B.gguf \
  --port 8080 \
  --ctx-size 8192 \
  --n-gpu-layers 35
```

#### Option 2: text-generation-webui

```bash
# Start text-generation-webui
cd /path/to/text-generation-webui
python server.py --model LFM2-8B-A1B --api --port 5000
```

#### Option 3: vLLM (OpenAI-compatible)

```bash
# Start vLLM server
vllm serve /path/to/LFM2-8B-A1B \
  --port 8080 \
  --dtype auto
```

### Configuration File

Edit `config_lfm2.json` to match your setup:

```json
{
  "local_llm": {
    "base_url": "http://127.0.0.1:8080",
    "mode": "llama-cpp",
    "model": "LFM2-8B-A1B",
    "timeout": 120,
    "max_retries": 3
  },
  "orchestrator_settings": {
    "use_numbskull": true,
    "use_semantic": true,
    "use_mathematical": true,
    "use_fractal": true,
    "fusion_method": "weighted_average",
    "embedding_enhancement": "metadata"
  }
}
```

### Optional Services

#### Semantic Embeddings (Eopiez)

```bash
cd ~/aipyapp/Eopiez
python api.py --port 8001
```

#### Mathematical Embeddings (LIMPS)

```bash
cd ~/aipyapp/9xdSq-LIMPS-FemTO-R1C/limps
julia --project=. -e 'using LIMPS; LIMPS.start_limps_server(8000)'
```

**Note**: If these services are unavailable, the system will use local fallbacks.

## Usage

### Quick Start - Demo Suite

Run the integrated demo suite in **Terminal**:

```bash
cd /home/kill/LiMp
python run_integrated_workflow.py --demo
```

This runs three demonstration queries showing different capabilities.

### Single Query

Run a single query in **Terminal**:

```bash
python run_integrated_workflow.py \
  --query "What are the main features of this system?" \
  --resources README.md requirements.txt \
  --inline "Focus on AI capabilities"
```

### Interactive Mode

Launch interactive mode in **Terminal** for testing:

```bash
python run_integrated_workflow.py --interactive
```

Commands in interactive mode:
- Type your query and press Enter
- `stats` - Show embedding statistics
- `clear` - Clear embedding cache
- `quit` or `exit` - Exit interactive mode

### Custom Configuration

Use a custom config file in **Terminal**:

```bash
python run_integrated_workflow.py --config my_config.json --demo
```

## Python API Usage

### Basic Example

```python
import asyncio
from numbskull_dual_orchestrator import create_numbskull_orchestrator

async def main():
    # Configuration
    local_configs = [{
        "base_url": "http://127.0.0.1:8080",
        "mode": "llama-cpp",
        "model": "LFM2-8B-A1B"
    }]
    
    settings = {
        "use_numbskull": True,
        "use_semantic": True,
        "use_mathematical": True,
        "use_fractal": True,
        "fusion_method": "weighted_average",
        "embedding_enhancement": "metadata"
    }
    
    # Create orchestrator
    orchestrator = create_numbskull_orchestrator(
        local_configs=local_configs,
        settings=settings
    )
    
    # Run query
    result = await orchestrator.run_with_embeddings(
        user_prompt="Analyze this system",
        resource_paths=["README.md"],
        inline_resources=["Additional context here"]
    )
    
    # Access results
    print("Summary:", result["summary"])
    print("Final Answer:", result["final"])
    print("Embeddings:", result["embedding_result"]["metadata"])
    
    # Cleanup
    await orchestrator.close()

asyncio.run(main())
```

### Advanced Example with Custom Configuration

```python
from numbskull_dual_orchestrator import (
    create_numbskull_orchestrator,
    NumbskullOrchestratorSettings
)
from advanced_embedding_pipeline import HybridConfig

# Custom numbskull config
numbskull_config = {
    "use_semantic": True,
    "use_mathematical": True,
    "use_fractal": False,  # Disable fractal for speed
    "fusion_method": "attention",  # Use attention-based fusion
    "parallel_processing": True,
    "cache_embeddings": True
}

# Custom orchestrator settings
settings = {
    "temperature": 0.8,
    "max_tokens": 1024,
    "style": "detailed",
    "use_numbskull": True,
    "embedding_enhancement": "full_vectors"  # Include embedding vectors in context
}

orchestrator = create_numbskull_orchestrator(
    local_configs=[{
        "base_url": "http://127.0.0.1:8080",
        "mode": "llama-cpp",
        "model": "LFM2-8B-A1B",
        "timeout": 180
    }],
    remote_config={  # Optional: use remote LLM for summarization
        "base_url": "https://api.openai.com",
        "api_key": "your-key",
        "model": "gpt-4o-mini"
    },
    settings=settings,
    numbskull_config=numbskull_config
)
```

## Features

### Hybrid Embedding Pipeline

The numbskull integration provides three types of embeddings:

1. **Semantic Embeddings**
   - Deep semantic understanding via Eopiez service
   - 768-dimensional vectors
   - Captures contextual meaning

2. **Mathematical Embeddings**
   - Symbolic and numerical analysis
   - LIMPS optimization integration
   - 1024-dimensional vectors
   - Handles equations, expressions, code AST

3. **Fractal Embeddings**
   - Mandelbrot, Julia, Sierpinski patterns
   - Hierarchical structure analysis
   - Entropy-based modifications
   - 1024-dimensional vectors

### Fusion Methods

Configure how embeddings are combined:

- **weighted_average** (default): Weighted fusion with configurable weights
- **concatenation**: Concatenate all embeddings into one vector
- **attention**: Attention-based dynamic weighting

### Embedding Enhancement Modes

Control how embeddings enhance the LLM context:

- **metadata** (default): Include embedding statistics and component info
- **similarity**: Add similarity metrics between embeddings
- **full_vectors**: Include truncated embedding vectors in prompt

### Performance Features

- **Caching**: Automatic embedding cache with configurable size
- **Parallel Processing**: Concurrent embedding generation
- **Async Operations**: Full async/await support
- **Fallback Mechanisms**: Local fallbacks when services unavailable

## Monitoring & Debugging

### View Embedding Statistics

```python
stats = orchestrator.get_embedding_stats()
print(f"Total embeddings: {stats['total_embeddings']}")
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Avg embedding time: {stats['avg_embedding_time']:.3f}s")
```

### Clear Caches

```python
orchestrator.clear_embedding_cache()
```

### Logging

Set logging level for detailed output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Troubleshooting

### LFM2-8B-A1B Not Responding

```bash
# Check if server is running
curl http://127.0.0.1:8080/v1/models

# Check llama.cpp logs
# Ensure model is loaded and endpoint is correct
```

### Numbskull Import Error

```bash
# Verify numbskull is installed
pip list | grep numbskull

# Reinstall if needed
pip install -e /home/kill/numbskull --force-reinstall
```

### Service Unavailable (Eopiez/LIMPS)

The system automatically falls back to local implementations when services are unavailable. Check logs for warnings:

```
WARNING - Semantic embedding failed: Connection refused
INFO - Using local fallback for semantic embeddings
```

This is expected behavior and the system will continue to work.

### Memory Issues

If embeddings consume too much memory:

```python
# Reduce cache size
settings = {
    "max_embedding_cache_size": 100,  # Default is 1000
    "use_fractal": False  # Disable resource-intensive components
}
```

## Performance Tuning

### For Speed

```json
{
  "orchestrator_settings": {
    "use_semantic": true,
    "use_mathematical": false,
    "use_fractal": false,
    "fusion_method": "weighted_average",
    "max_embedding_cache_size": 1000
  }
}
```

### For Quality

```json
{
  "orchestrator_settings": {
    "use_semantic": true,
    "use_mathematical": true,
    "use_fractal": true,
    "fusion_method": "attention",
    "embedding_enhancement": "full_vectors"
  }
}
```

### For Resource Efficiency

```json
{
  "orchestrator_settings": {
    "use_semantic": true,
    "use_mathematical": true,
    "use_fractal": true,
    "fusion_method": "weighted_average",
    "max_embedding_cache_size": 500,
    "embed_resources": true,
    "embed_user_prompt": false
  },
  "local_llm": {
    "timeout": 60,
    "max_retries": 2
  }
}
```

## Examples

### Example 1: Technical Documentation Analysis

```bash
python run_integrated_workflow.py \
  --query "Summarize the key technical concepts" \
  --resources SYSTEM_OVERVIEW.md README.md \
  --inline "Focus on architecture and design patterns"
```

### Example 2: Mathematical Problem Solving

```bash
python run_integrated_workflow.py \
  --query "Solve and explain the optimization problem" \
  --inline "minimize f(x) = x^2 + 2x + 1 subject to x >= 0"
```

### Example 3: Code Analysis

```python
result = await orchestrator.run_with_embeddings(
    user_prompt="Analyze the code complexity and suggest improvements",
    resource_paths=["dual_llm_orchestrator.py"],
    inline_resources=["Focus on: performance, maintainability, scalability"]
)
```

## Integration with Other Components

### With Holographic Memory System

```python
from holographic_memory_system import HolographicMemorySystem

memory = HolographicMemorySystem()
orchestrator = create_numbskull_orchestrator(...)

# Store results in holographic memory
result = await orchestrator.run_with_embeddings(...)
await memory.store(
    content=result["final"],
    metadata=result["embedding_result"]["metadata"]
)
```

### With Emergent Cognitive Network

```python
from emergent_cognitive_network import EmergentCognitiveNetwork

network = EmergentCognitiveNetwork()
orchestrator = create_numbskull_orchestrator(...)

# Use orchestrator in cognitive network
result = await orchestrator.run_with_embeddings(...)
await network.process_with_context(
    result["final"],
    embeddings=result["embedding_result"]["fused_embedding"]
)
```

## Files Reference

- **`numbskull_dual_orchestrator.py`** - Main orchestrator implementation
- **`config_lfm2.json`** - Configuration file
- **`run_integrated_workflow.py`** - Demo and testing script
- **`requirements.txt`** - Dependencies including numbskull
- **`dual_llm_orchestrator.py`** - Base orchestrator (inherited)

## API Reference

### NumbskullDualOrchestrator

Main orchestrator class with embedding integration.

#### Methods

- `run_with_embeddings(user_prompt, resource_paths, inline_resources)` - Run with full embedding support
- `get_embedding_stats()` - Get embedding performance statistics
- `clear_embedding_cache()` - Clear the embedding cache
- `close()` - Cleanup resources

### create_numbskull_orchestrator

Factory function to create orchestrator instances.

#### Parameters

- `local_configs` - List of local LLM configurations
- `remote_config` - Optional remote LLM configuration
- `settings` - Orchestrator settings dictionary
- `numbskull_config` - Numbskull pipeline configuration

## License

MIT License - See LICENSE file for details.

## Support

For issues or questions:

1. Check logs for detailed error messages
2. Verify all services are running correctly
3. Test with demo suite: `python run_integrated_workflow.py --demo`
4. Review this documentation for configuration options

## Next Steps

1. **Start Services** - Launch LFM2-8B-A1B and optional services
2. **Run Demo** - Execute `python run_integrated_workflow.py --demo`
3. **Configure** - Adjust `config_lfm2.json` for your setup
4. **Integrate** - Use the orchestrator in your own applications

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Status**: Production Ready

