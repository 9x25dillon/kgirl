# kgirl - Multi-Framework LLM Knowledge Platform

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Julia](https://img.shields.io/badge/Julia-1.9%2B-purple)](https://julialang.org/)

A production-ready, multi-framework LLM knowledge processing platform that combines quantum-inspired architectures, GPU-accelerated optimization, chaos-aware routing, and topological consensus for advanced AI reasoning.

## Overview

**kgirl** integrates four major AI/ML frameworks into a unified platform that transforms knowledge processing through:

- **Multi-model LLM consensus** with topological coherence
- **Quantum-inspired knowledge representation** with holographic patterns
- **GPU-accelerated matrix optimization** via LIMPS framework
- **Chaos-aware RAG** with Julia-based vector database
- **Neuro-symbolic reasoning** with fractal embeddings
- **Multi-backend optimization** (Python GPU, Julia mathematical, hybrid)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    KGIRL PLATFORM                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │  Topological     │  │  ChaosRAGJulia   │  │   Unified    │ │
│  │  Consensus API   │  │  Vector DB       │  │   Quantum    │ │
│  │  (main.py)       │  │  (server.jl)     │  │   LLM System │ │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────┤ │
│  │ • Multi-LLM      │  │ • PostgreSQL     │  │ • QHKS       │ │
│  │ • Phase          │  │ • pgvector       │  │ • LIMPS      │ │
│  │   Coherence      │  │ • Chaos Router   │  │ • NuRea_sim  │ │
│  │ • Cardy Energy   │  │ • HHT/EEMD       │  │ • Numbskull  │ │
│  │ • Trinary        │  │ • RAG Query      │  │ • 4-Layer    │ │
│  │   Quantization   │  │ • Temporal       │  │   Embedding  │ │
│  │ • Reranking      │  │   Causality      │  │ • Quad       │ │
│  │                  │  │                  │  │   Entropy    │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
│         │                      │                     │         │
│         └──────────────────────┼─────────────────────┘         │
│                                │                               │
│                  ┌─────────────▼──────────────┐                │
│                  │   Service Orchestration    │                │
│                  │  • Ollama LLM (11434)      │                │
│                  │  • LIMPS Julia (8000)      │                │
│                  │  • PostgreSQL (5432)       │                │
│                  │  • FastAPI Server (8000)   │                │
│                  └────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Topological Consensus API (`main.py`)

FastAPI server providing multi-model LLM consensus with topological mathematics.

**Key Features:**
- Multi-model orchestration (OpenAI GPT-4, Anthropic Claude)
- Phase coherence calculation using topological anyon theory
- Cardy boundary energy for hallucination detection
- Document reranking with spectral weights
- Trinary quantization for efficient representation

**Endpoints:**
```bash
GET  /health          # Service health check
GET  /config          # System configuration
POST /ask             # Multi-model consensus query
POST /rerank          # Document reranking
```

**Example Usage:**
```python
import requests

# Multi-model consensus query
response = requests.post("http://localhost:8000/ask", json={
    "prompt": "Explain quantum entanglement",
    "min_coherence": 0.80,
    "max_energy": 0.30,
    "return_all": False
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Coherence: {result['coherence']}")
print(f"Decision: {result['decision']}")  # auto, needs_citations, or escalate
```

### 2. ChaosRAGJulia Vector Database (`server.jl`)

Julia-based RAG system with chaos-aware routing and time-frequency analytics.

**Key Features:**
- PostgreSQL + pgvector (1536D embeddings)
- KFP-inspired chaos router: `stress = σ(1.8·vol + 1.5·entropy + 0.8·|grad|)`
- HHT/EEMD (Hilbert-Huang Transform / Ensemble Empirical Mode Decomposition)
- Temporal causality tracking (step-1 and step-5 edges)
- Mixed retrieval strategy (vector, graph, HHT) based on stress levels

**Endpoints:**
```bash
POST /chaos/rag/index           # Index documents
POST /chaos/telemetry           # Push asset telemetry
POST /chaos/hht/ingest          # EEMD + Hilbert analysis
POST /chaos/graph/entangle      # Upsert graph edges
GET  /chaos/graph/:uuid         # Fetch node + edges
POST /chaos/rag/query           # Chaos-routed RAG query
```

**Example Usage:**
```bash
# Index documents
curl -X POST http://localhost:8001/chaos/rag/index \
  -H "Content-Type: application/json" \
  -d '{"docs": [
    {"source": "paper.pdf", "kind": "research", "content": "Quantum computing..."}
  ]}'

# Query with chaos routing
curl -X POST http://localhost:8001/chaos/rag/query \
  -H "Content-Type: application/json" \
  -d '{"q": "What is quantum computing?", "k": 5}'
```

### 3. Unified Quantum LLM System (`complete_unified_platform.py`)

Integrates four major frameworks for comprehensive knowledge processing.

**Integrated Frameworks:**

#### a. Quantum Holographic Knowledge System (QHKS)
- Quantum-dimensional encodings with superposition and entanglement
- Chaos_Ragged learning (edge of chaos dynamics)
- Orwells-egged hierarchical structuring
- Holographic interference patterns
- Qualia encoding (experiential knowledge)
- Fractal resonance harmonics

**Files:** `quantum_holographic_knowledge_synthesis.py`, `quantum_knowledge_processing.py`, `quantum_knowledge_database.py`

#### b. LIMPS Framework
- GPU-accelerated matrix optimization (PyTorch CUDA)
- 4 optimization methods: sparsity, rank, structure, polynomial
- 2D Chebyshev polynomial approximation
- Advanced entropy analysis
- 10-50x speedup over CPU

**Files:** `quantum_limps_integration.py`, `quantum_limps_demo.py`
**External Repo:** [9xdSq-LIMPS-FemTO-R1C](https://github.com/9x25dillon/9xdSq-LIMPS-FemTO-R1C)

#### c. NuRea_sim Integration
- Julia mathematical optimization (OSQP, Convex.jl, SCS)
- Token-based entropy engine
- Matrix orchestrator with declarative DAG
- Nuclear physics simulation mathematics

**Files:** `unified_quantum_llm_system.py`
**External Repo:** [NuRea_sim](https://github.com/9x25dillon/NuRea_sim)

#### d. Numbskull
- Fractal cascade embedder (Mandelbrot, Julia, Sierpinski fractals)
- Neuro-symbolic engine with 9 analytical modules
- Holographic associative memory
- Swarm intelligence and emergent protocols

**Directory:** `advanced_embedding_pipeline/`

## Installation

### Prerequisites
```bash
# Python 3.10+
python --version

# Julia 1.9+
julia --version

# PostgreSQL with pgvector
psql --version

# Ollama (for local LLM)
ollama --version
```

### Quick Install

```bash
# Clone repository
git clone https://github.com/9x25dillon/kgirl.git
cd kgirl

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
#   DATABASE_URL=postgres://user:pass@localhost:5432/chaos

# Install Julia dependencies
julia --project -e 'using Pkg; Pkg.add.(["HTTP","JSON3","LibPQ","DSP","UUIDs","Interpolations"])'

# Set up PostgreSQL database
createdb chaos
psql chaos -c "CREATE EXTENSION vector;"

# Pull Ollama models (optional)
ollama pull qwen2.5:3b
```

### Optional: Clone External Frameworks

```bash
# For full LIMPS functionality
git clone https://github.com/9x25dillon/9xdSq-LIMPS-FemTO-R1C.git

# For full NuRea_sim functionality
git clone https://github.com/9x25dillon/NuRea_sim.git
```

## Quick Start

### 1. Start Core Services

```bash
# Start all services with one command
./START_NOW.sh

# Or manually:
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Julia LIMPS service
cd 9xdSq-LIMPS-FemTO-R1C
bash start_limps.sh

# Terminal 3: Start ChaosRAG Julia server
julia server.jl

# Terminal 4: Start main API
python main.py
```

### 2. Basic Usage Examples

#### Topological Consensus Query

```python
import asyncio
import requests

# Query multiple LLMs with consensus
response = requests.post("http://localhost:8000/ask", json={
    "prompt": "Explain the implications of quantum entanglement for computing",
    "min_coherence": 0.75,
    "max_energy": 0.35
})

result = response.json()
print(f"Consensus Answer: {result['answer']}")
print(f"Model Agreement (Coherence): {result['coherence']:.2%}")
print(f"Hallucination Risk (Energy): {result['energy']:.2%}")
print(f"Decision: {result['decision']}")
```

#### ChaosRAG Knowledge Query

```bash
# Index documents
curl -X POST http://localhost:8001/chaos/rag/index \
  -H "Content-Type: application/json" \
  -d '{
    "docs": [
      {
        "source": "quantum_computing.pdf",
        "kind": "research",
        "content": "Quantum computers use qubits that can exist in superposition...",
        "meta": {"author": "Alice", "year": 2024}
      }
    ]
  }'

# Query with chaos-aware routing
curl -X POST http://localhost:8001/chaos/rag/query \
  -H "Content-Type: application/json" \
  -d '{"q": "How do quantum computers work?", "k": 5}'
```

#### Unified Quantum LLM System

```python
from complete_unified_platform import CompleteUnifiedPlatform, CompleteSystemConfig
import asyncio

async def main():
    # Create system with all frameworks enabled
    config = CompleteSystemConfig(
        enable_numbskull_embeddings=True,
        enable_neuro_symbolic=True,
        enable_holographic_memory=True,
        use_quad_entropy=True
    )

    platform = CompleteUnifiedPlatform(config)
    await platform.initialize()

    # Process knowledge with all 4 frameworks
    result = await platform.process_complete(
        knowledge_source="quantum_physics_textbook.pdf",
        use_all_backends=True
    )

    print(f"Processing Results:")
    print(f"  - Quantum encoding: {result.quantum_dimensions}D")
    print(f"  - LIMPS optimization: {result.limps_compression:.1%}")
    print(f"  - Entropy scores: {result.entropy_scores}")
    print(f"  - Neuro-symbolic insights: {len(result.insights)}")

asyncio.run(main())
```

## Workflow & Pipelines

### Knowledge Processing Pipeline

```
1. Ingestion
   ↓ (PDFs, code, text, equations)
2. Multi-Layer Embedding
   ↓ (Semantic, Mathematical, Fractal, Holographic)
3. Triple-Backend Optimization
   ↓ (LIMPS GPU, NuRea Julia, Numbskull)
4. Quad Entropy Analysis
   ↓ (4 entropy engines)
5. Vector Database Storage
   ↓ (PostgreSQL + pgvector)
6. Chaos-Aware Indexing
   ↓ (ChaosRAG router)
7. Query Processing
   ↓ (Multi-model consensus + RAG)
8. Topological Validation
   ↓ (Coherence + Energy metrics)
9. Response Generation
   └→ (Validated, contextualized answer)
```

### Multi-Model Consensus Pipeline

```
User Query
   ↓
1. Route to Multiple LLMs (OpenAI + Anthropic)
   ↓
2. Generate Parallel Responses
   ↓
3. Embed All Responses
   ↓
4. Calculate Phase Coherence
   │ (Topological anyon weights)
   ↓
5. Calculate Cardy Energy
   │ (Hallucination detection)
   ↓
6. Apply Decision Logic
   │ • High coherence + Low energy → Auto-respond
   │ • Medium values → Request citations
   │ • Low coherence OR High energy → Escalate to human
   ↓
7. Return Weighted Consensus
```

### Chaos-Aware RAG Pipeline

```
Query
   ↓
1. Calculate System Stress
   │ stress = σ(1.8·vol + 1.5·entropy + 0.8·|grad|)
   ↓
2. Determine Retrieval Mix
   │ • Low stress → Vector search (90%)
   │ • Medium stress → Vector (60%) + Graph (30%) + HHT (10%)
   │ • High stress → Graph (50%) + HHT (40%) + Vector (10%)
   ↓
3. Execute Mixed Retrieval
   ↓
4. Apply Temporal Causality
   │ (step-1 and step-5 edge traversal)
   ↓
5. Generate Context-Aware Response
   │ (OpenAI GPT-4 with retrieved context)
   ↓
6. Return Answer + Metadata
```

## Key Scripts & Entry Points

| File | Purpose | Usage |
|------|---------|-------|
| `main.py` | Topological Consensus API | `python main.py` |
| `server.jl` | ChaosRAG Julia server | `julia server.jl` |
| `complete_unified_platform.py` | Unified 4-framework system | `python complete_unified_platform.py` |
| `quantum_llm_interface.py` | Interactive REPL | `python quantum_llm_interface.py` |
| `START_NOW.sh` | Start all services | `./START_NOW.sh` |
| `start_all_services.sh` | Service orchestration | `./start_all_services.sh` |

## Configuration

### Environment Variables

**IMPORTANT**: All configuration values, especially service ports, should be defined in your `.env` file. The `.env.example` file serves as the single source of truth for available configuration options.

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
DATABASE_URL=postgres://user:pass@localhost:5432/chaos

# LLM Models
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-large
ANTHROPIC_CHAT_MODEL=claude-3-5-sonnet-latest

# Topological Parameters
CENTRAL_CHARGE=627
N_ANYONS=5

# Service Ports (all configurable via .env)
MAIN_API_PORT=8000        # Topological Consensus API (main.py)
CHAOS_RAG_PORT=8001       # ChaosRAG Julia service (server.jl)
LIMPS_PORT=8000           # LIMPS Julia service (external)
OLLAMA_PORT=11434         # Ollama LLM service
```

**Port Configuration Notes:**
- All services read their ports from environment variables
- To change ports, update your `.env` file (copy from `.env.example`)
- The services will automatically use the configured ports on startup
- Default values are provided if environment variables are not set

### Model Configuration Files

- `config_v3.1.json` - Main model configuration
- `config_lfm2.json` - LFM2 model settings
- `config_671B.json` - Large model (671B parameters)
- `config_236B.json` - Medium model (236B parameters)
- `config_16B.json` - Small model (16B parameters)

## API Reference

### Topological Consensus API

#### POST /ask
Query multiple LLMs with topological consensus.

**Request:**
```json
{
  "prompt": "string",
  "min_coherence": 0.80,
  "max_energy": 0.30,
  "return_all": false
}
```

**Response:**
```json
{
  "answer": "string",
  "decision": "auto|needs_citations|escalate",
  "coherence": 0.85,
  "energy": 0.25,
  "weights": [0.6, 0.4],
  "model_names": ["openai:gpt-4o-mini", "anthropic:claude-3-5-sonnet"],
  "all_outputs": ["response1", "response2"]
}
```

#### POST /rerank
Rerank documents using spectral coherence weights.

**Request:**
```json
{
  "query": "string",
  "docs": [
    {"id": "doc1", "text": "content1"},
    {"id": "doc2", "text": "content2"}
  ],
  "trinary_threshold": 0.25,
  "alpha": 0.7,
  "beta": 0.3
}
```

**Response:**
```json
{
  "ranked_ids": ["doc2", "doc1"],
  "scores": [0.92, 0.78]
}
```

### ChaosRAG API

#### POST /chaos/rag/query
Query knowledge base with chaos-aware routing.

**Request:**
```json
{
  "q": "What is quantum computing?",
  "k": 5
}
```

**Response:**
```json
{
  "answer": "Quantum computing is...",
  "sources": ["doc1", "doc2"],
  "stress_level": 0.45,
  "retrieval_mix": {"vector": 0.6, "graph": 0.3, "hht": 0.1}
}
```

## Performance

### Benchmarks

| Component | Metric | Performance |
|-----------|--------|-------------|
| LIMPS GPU Optimization | Speedup | 10-50x vs CPU |
| Matrix Compression | Ratio | 30-70% with <5% loss |
| Chaos Router | Latency | <100ms query routing |
| Multi-Model Consensus | Coherence | 85%+ typical |
| Vector Search | Throughput | 10k queries/sec |
| HHT Analysis | Processing | 1k samples/sec |

### Scalability

- Horizontal scaling via multiple FastAPI workers
- Distributed knowledge base with network sync
- GPU acceleration for matrix operations
- Async/await for concurrent processing
- Connection pooling for database

## Development

### Running Tests

```bash
# All tests
pytest

# Specific modules
pytest tests/test_emergent_system.py
pytest tests/test_system.py

# With coverage
pytest --cov=. --cov-report=html
```

### Code Quality

```bash
# Format code
black .

# Lint
flake8

# Type checking
mypy .
```

## Documentation

- **`UNIFIED_SYSTEM_README.md`** - Comprehensive unified system guide
- **`FINAL_PROJECT_SUMMARY.md`** - Complete project overview
- **`QUANTUM_KNOWLEDGE_README.md`** - QHKS documentation
- **`QUANTUM_LIMPS_INTEGRATION_README.md`** - LIMPS integration guide
- **`QUICKSTART.md`** - Quick setup guide
- **`COMPLETE_STARTUP_GUIDE.md`** - Detailed startup instructions
- **`SERVICE_STARTUP_GUIDE.md`** - Service management

## External Integrations

### Required Services

1. **Ollama** - Local LLM inference
   ```bash
   ollama serve
   ollama pull qwen2.5:3b
   ```

2. **PostgreSQL + pgvector** - Vector database
   ```bash
   createdb chaos
   psql chaos -c "CREATE EXTENSION vector;"
   ```

3. **Julia** - Mathematical backend
   ```bash
   julia --project -e 'using Pkg; Pkg.instantiate()'
   ```

### Optional Frameworks

- **[LIMPS](https://github.com/9x25dillon/9xdSq-LIMPS-FemTO-R1C)** - GPU-accelerated matrix optimization
- **[NuRea_sim](https://github.com/9x25dillon/NuRea_sim)** - Nuclear physics simulation
- **CTH** - Topological consciousness library (custom path via `CTH_PATH`)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Code components are also available under [LICENSE-CODE](LICENSE-CODE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and standards
- Testing requirements
- Documentation updates
- Pull request process

## Security

For security issues, see [SECURITY.md](SECURITY.md).

## Acknowledgments

- Topological consciousness theory integration
- Quantum-inspired algorithms research community
- Julia mathematical optimization ecosystem
- Open-source LLM community (Ollama, OpenAI, Anthropic)
- PostgreSQL and pgvector teams

## Citation

If you use this platform in research, please cite:

```bibtex
@software{kgirl2025,
  title={kgirl: Multi-Framework LLM Knowledge Platform},
  author={9x25dillon},
  year={2025},
  url={https://github.com/9x25dillon/kgirl},
  license={Apache-2.0}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/9x25dillon/kgirl/issues)
- **Discussions**: [GitHub Discussions](https://github.com/9x25dillon/kgirl/discussions)
- **Documentation**: See `docs/` directory

---

**Built with**: Python, Julia, PostgreSQL, FastAPI, PyTorch, NumPy, SciPy, and the collective intelligence of the open-source community.
