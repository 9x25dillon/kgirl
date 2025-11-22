# Complete Unified Platform - Dependency Map & Integration Plan

## Executive Summary

This document maps all dependencies required for the complete end-to-end LLM pipeline to function without deviations or substitutions.

**Status Date:** 2025-11-08
**Branch:** claude/kgirl-rllm-status-011CUvLXaTN1ZJKCVUNDuUCv

---

## Component Status Matrix

| Component | Status | Location | Repository |
|-----------|--------|----------|------------|
| **Core Python Modules** | ✅ PRESENT | `/home/user/kgirl/` | kgirl |
| **LIMPS Framework** | ⚠️ NEEDS CLONE | Not present | 9xdSq-LIMPS-FemTO-R1C |
| **NuRea_sim** | ⚠️ NEEDS CLONE | Not present | NuRea_sim |
| **Numbskull (external)** | ⚠️ NEEDS CLONE | Not present | numbskull |
| **ChaosRAG Julia** | ✅ PRESENT | `/home/user/kgirl/server.jl` | kgirl |
| **Topological API** | ✅ PRESENT | `/home/user/kgirl/main.py` | kgirl |

---

## Dependency Tree

###  1. complete_unified_platform.py

**Direct Imports:**
```python
from unified_quantum_llm_system import (
    UnifiedQuantumLLMSystem,              # ✅ Present
    UnifiedSystemConfig,                   # ✅ Present
    OptimizationBackend,                   # ✅ Present
    UnifiedOptimizationResult              # ✅ Present
)

# Numbskull components (EXTERNAL - needs clone)
from neuro_symbolic_engine import (
    EntropyAnalyzer,                       # ⚠️ EXISTS LOCALLY in kgirl
    DianneReflector,                       # ⚠️ Needs verification
    MatrixTransformer,                     # ⚠️ Needs verification
    JuliaSymbolEngine,                     # ⚠️ Needs verification
    FractalResonator,                      # ⚠️ Needs verification
    NeuroSymbolicEngine                    # ✅ EXISTS LOCALLY in kgirl
)
from fractal_cascade_embedder import (
    FractalCascadeEmbedder,                # ✅ EXISTS LOCALLY
    FractalConfig                          # ✅ EXISTS LOCALLY
)
from holographic_similarity_engine import HolographicSimilarityEngine  # ✅ EXISTS LOCALLY
from emergent_cognitive_network import execute_emergent_protocol       # ✅ EXISTS LOCALLY
```

**Expected Path Structure:**
```
NUMBSKULL_PATH = Path(__file__).parent / "numbskull"
- /home/user/kgirl/numbskull/
  - neuro_symbolic_engine.py
  - fractal_cascade_embedder.py
  - holographic_similarity_engine.py
  - emergent_cognitive_network.py
  - advanced_embedding_pipeline/
```

---

### 2. unified_quantum_llm_system.py

**Direct Imports:**
```python
from quantum_limps_integration import (
    QuantumLIMPSIntegration,               # ✅ Present
    QuantumLIMPSConfig,                    # ✅ Present
    OptimizedQuantumState                  # ✅ Present
)

from quantum_holographic_knowledge_synthesis import (
    KnowledgeQuantum,                      # ✅ Present
    DataSourceType                         # ✅ Present
)

from quantum_knowledge_database import (
    QuantumHolographicKnowledgeDatabase    # ✅ Present
)

# NuRea_sim components (EXTERNAL - needs clone)
from matrix_orchestrator import (
    JuliaBackend,                          # ⚠️ Needs clone
    MockBackend,                           # ⚠️ Needs clone
    OptimizeRequest,                       # ⚠️ Needs clone
    OptimizeResponse,                      # ⚠️ Needs clone
    MatrixChunk,                           # ⚠️ Needs clone
    RunPlan,                               # ⚠️ Needs clone
    EntropyReport                          # ⚠️ Needs clone
)

from ent.entropy_engine import (
    Token,                                 # ⚠️ Needs clone
    EntropyNode,                           # ⚠️ Needs clone
    EntropyEngine                          # ⚠️ Needs clone
)
```

**Expected Path Structure:**
```
NUREA_PATH = Path(__file__).parent / "NuRea_sim"
- /home/user/kgirl/NuRea_sim/
  - matrix_orchestrator.py
  - entropy engine/
    - ent/
      - entropy_engine.py
```

---

### 3. quantum_limps_integration.py

**External Dependencies:**
```python
# Requires LIMPS framework to be cloned
# Path: /home/user/kgirl/9xdSq-LIMPS-FemTO-R1C/
```

**Key Files Needed from LIMPS:**
- Matrix processor implementations
- GPU optimization routines
- Julia service configurations

---

## Required Repositories

### Repository 1: 9xdSq-LIMPS-FemTO-R1C
**URL:** https://github.com/9x25dillon/9xdSq-LIMPS-FemTO-R1C
**Clone Command:**
```bash
cd /home/user/kgirl
git clone https://github.com/9x25dillon/9xdSq-LIMPS-FemTO-R1C.git
```

**Purpose:** GPU-accelerated matrix optimization
**Integration Point:** `quantum_limps_integration.py`

---

### Repository 2: NuRea_sim
**URL:** https://github.com/9x25dillon/NuRea_sim
**Clone Command:**
```bash
cd /home/user/kgirl
git clone https://github.com/9x25dillon/NuRea_sim.git
```

**Purpose:** Julia backend, matrix orchestrator, entropy engine
**Integration Point:** `unified_quantum_llm_system.py`

**Key Components:**
- `matrix_orchestrator.py` - Julia backend coordination
- `entropy engine/ent/entropy_engine.py` - Token-based entropy

---

### Repository 3: numbskull
**URL:** https://github.com/9x25dillon/numbskull
**Clone Command:**
```bash
cd /home/user/kgirl
git clone https://github.com/9x25dillon/numbskull.git
```

**Purpose:** Fractal embeddings, neuro-symbolic engine, holographic memory
**Integration Point:** `complete_unified_platform.py`

**Key Components:**
- `neuro_symbolic_engine.py`
- `fractal_cascade_embedder.py`
- `holographic_similarity_engine.py`
- `emergent_cognitive_network.py`
- `advanced_embedding_pipeline/`

---

## Local Components (Already Present in kgirl)

### Core Quantum System ✅
- `quantum_holographic_knowledge_synthesis.py`
- `quantum_knowledge_processing.py`
- `quantum_knowledge_database.py`
- `quantum_llm_interface.py`
- `quantum_limps_integration.py`

### Standalone Python Modules ✅
- `neuro_symbolic_engine.py` (local version)
- `entropy_engine.py` (local version)
- `emergent_cognitive_network.py` (local version)
- `holographic_memory_system.py` (local version)
- `matrix_processor.py` (local adapter)
- `advanced_embedding_pipeline/fractal_cascade_embedder.py`

### Julia/Database Services ✅
- `server.jl` - ChaosRAG Julia server
- `main.py` - Topological Consensus API

---

## Integration Strategy

### Phase 1: Clone External Repositories ⚠️
```bash
cd /home/user/kgirl
git clone https://github.com/9x25dillon/9xdSq-LIMPS-FemTO-R1C.git
git clone https://github.com/9x25dillon/NuRea_sim.git
git clone https://github.com/9x25dillon/numbskull.git
```

### Phase 2: Verify Directory Structure ✅
```
/home/user/kgirl/
├── complete_unified_platform.py
├── unified_quantum_llm_system.py
├── quantum_limps_integration.py
├── main.py
├── server.jl
├── 9xdSq-LIMPS-FemTO-R1C/          # ← NEEDS CLONE
├── NuRea_sim/                       # ← NEEDS CLONE
│   ├── matrix_orchestrator.py
│   └── entropy engine/
│       └── ent/
│           └── entropy_engine.py
└── numbskull/                       # ← NEEDS CLONE
    ├── neuro_symbolic_engine.py
    ├── fractal_cascade_embedder.py
    ├── holographic_similarity_engine.py
    ├── emergent_cognitive_network.py
    └── advanced_embedding_pipeline/
```

### Phase 3: Create Integration Glue (New Files Only)
1. **`complete_integration_runner.py`** - End-to-end test script
2. **`integration_health_check.py`** - Verify all components loaded
3. **`service_orchestrator.py`** - Unified service startup

### Phase 4: Dependency Resolution
- Install Python dependencies from all repos
- Configure Julia packages
- Set up PostgreSQL + pgvector
- Configure API keys

---

## Service Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Complete Unified Platform                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Port 8000: FastAPI Main Server (main.py)                      │
│  ├─ Topological Consensus API                                  │
│  ├─ Multi-model orchestration (OpenAI + Anthropic)             │
│  └─ Phase coherence + Cardy energy                             │
│                                                                 │
│  Port 8081: Julia ChaosRAG Server (server.jl)                  │
│  ├─ PostgreSQL + pgvector (1536D)                              │
│  ├─ EEMD/Hilbert-Huang Transform                               │
│  ├─ Chaos router (stress-based mixing)                         │
│  └─ RAG query endpoint                                         │
│                                                                 │
│  Port 8000: LIMPS Julia Service (9xdSq-LIMPS-FemTO-R1C)       │
│  ├─ GPU-accelerated matrix optimization                        │
│  └─ 4 optimization methods                                     │
│                                                                 │
│  Port 9000: NuRea Julia Backend (NuRea_sim)                    │
│  ├─ Matrix orchestrator (OSQP, Convex.jl)                      │
│  └─ Entropy engine                                             │
│                                                                 │
│  Port 11434: Ollama LLM Service (external)                     │
│  └─ Local model inference                                      │
│                                                                 │
│  Port 5432: PostgreSQL + pgvector (external)                   │
│  └─ Vector database backend                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## API Keys & Environment Variables

```bash
# Required for production
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DATABASE_URL="postgres://user:pass@localhost:5432/chaos"

# Optional CTH integration
export CTH_PATH="/path/to/CTH"

# Service endpoints
export LIMPS_JULIA_URL="http://localhost:8000"
export NUREA_JULIA_URL="http://localhost:9000"
export CHAOS_RAG_URL="http://localhost:8081"
export OLLAMA_URL="http://localhost:11434"

# Model configuration
export OPENAI_CHAT_MODEL="gpt-4o-mini"
export OPENAI_EMBED_MODEL="text-embedding-3-large"
export ANTHROPIC_CHAT_MODEL="claude-3-5-sonnet-latest"
export CENTRAL_CHARGE="627"
export N_ANYONS="5"
```

---

## Execution Plan

### Step 1: Clone Repositories
```bash
cd /home/user/kgirl
git clone https://github.com/9x25dillon/9xdSq-LIMPS-FemTO-R1C.git
git clone https://github.com/9x25dillon/NuRea_sim.git
git clone https://github.com/9x25dillon/numbskull.git
```

### Step 2: Install Dependencies
```bash
# Python dependencies
pip install -r requirements.txt
pip install -r 9xdSq-LIMPS-FemTO-R1C/requirements.txt
pip install -r NuRea_sim/requirements.txt
pip install -r numbskull/requirements.txt

# Julia dependencies
julia --project -e 'using Pkg; Pkg.add.(["HTTP","JSON3","LibPQ","DSP","UUIDs","Interpolations","OSQP","Convex","SCS"])'
```

### Step 3: Database Setup
```bash
createdb chaos
psql chaos -c "CREATE EXTENSION vector;"
```

### Step 4: Start Services (in order)
```bash
# Terminal 1: PostgreSQL (already running)
# Terminal 2: Ollama
ollama serve

# Terminal 3: Julia ChaosRAG
julia server.jl

# Terminal 4: LIMPS Julia Service
cd 9xdSq-LIMPS-FemTO-R1C
bash start_limps.sh

# Terminal 5: NuRea Julia Backend (if separate service)
cd NuRea_sim
# [start command from repo]

# Terminal 6: Main FastAPI Server
python main.py
```

### Step 5: Test End-to-End
```bash
python complete_integration_runner.py
```

---

## Health Check Endpoints

Once all services are running:

| Service | Health Check |
|---------|-------------|
| Main API | `curl http://localhost:8000/health` |
| ChaosRAG | `curl http://localhost:8081/health` |
| Complete Platform | `python integration_health_check.py` |

---

## Known Gaps & Resolutions

### Gap 1: Missing External Repos
**Status:** Identified
**Resolution:** Clone 3 repos (LIMPS, NuRea_sim, numbskull)

### Gap 2: Local vs External Numbskull Components
**Status:** kgirl has local versions of some Numbskull modules
**Resolution:** External numbskull repo is canonical; local versions are adapters/bridges

### Gap 3: Julia Service Coordination
**Status:** Multiple Julia services on same port
**Resolution:** Configure distinct ports (8000 → LIMPS, 9000 → NuRea, 8081 → ChaosRAG)

### Gap 4: API Key Dependencies
**Status:** Some components fall back to stubs without keys
**Resolution:** Set all required API keys for production use

---

## Success Criteria

✅ All 3 external repos cloned
✅ All Python imports resolve without errors
✅ All Julia services start successfully
✅ PostgreSQL + pgvector operational
✅ End-to-end pipeline processes test document
✅ All 4 optimization backends return results
✅ Quad entropy analysis completes
✅ RAG queries return contextualized answers

---

**Next Action:** Clone the 3 required repositories

```bash
cd /home/user/kgirl && \
  git clone https://github.com/9x25dillon/9xdSq-LIMPS-FemTO-R1C.git && \
  git clone https://github.com/9x25dillon/NuRea_sim.git && \
  git clone https://github.com/9x25dillon/numbskull.git
```
