# kgirl API Documentation

This document provides comprehensive API reference documentation for the kgirl platform's two main services:

1. **Topological Consensus API** - Multi-model LLM consensus with topological coherence (`main.py`)
2. **ChaosRAG API** - Chaos-aware retrieval-augmented generation (`server.jl`)

## 🚀 Local LLM Support

**NEW:** kgirl now supports local LLMs via [Ollama](https://ollama.ai) - **no API keys required!**

### Quick Configuration

**Local only (default - no API keys):**
```bash
# .env
MODELS=ollama:chat=qwen2.5:3b,embed=nomic-embed-text
```

**Cloud APIs:**
```bash
# .env
MODELS=openai:chat=gpt-4o-mini,embed=text-embedding-3-large
OPENAI_API_KEY=sk-...
```

**Hybrid (local + cloud consensus):**
```bash
# .env
MODELS=ollama:chat=qwen2.5:3b,embed=nomic-embed-text|openai:chat=gpt-4o-mini,embed=text-embedding-3-large
OPENAI_API_KEY=sk-...
```

See [LOCAL_LLM_SETUP.md](LOCAL_LLM_SETUP.md) for complete setup guide.

---

## Table of Contents

- [Topological Consensus API](#topological-consensus-api)
  - [GET /health](#get-health)
  - [GET /config](#get-config)
  - [POST /ask](#post-ask)
  - [POST /rerank](#post-rerank)
- [ChaosRAG API](#chaosrag-api)
  - [POST /chaos/rag/index](#post-chaosragindex)
  - [POST /chaos/telemetry](#post-chaostelemetry)
  - [POST /chaos/hht/ingest](#post-chaoshhtingest)
  - [POST /chaos/graph/entangle](#post-chaosgraphentangle)
  - [GET /chaos/graph/:uuid](#get-chaosgraphuuid)
  - [POST /chaos/rag/query](#post-chaosragquery)

---

## Topological Consensus API

**Base URL**: `http://localhost:8000` (configurable via `MAIN_API_PORT`)

The Topological Consensus API provides multi-model LLM consensus using topological mathematics to measure coherence and detect hallucinations.

**🚀 Now supports local LLMs via Ollama - no API keys required!**

### GET /health

Health check endpoint to verify service status.

**Response (Local LLM mode)**

```json
{
  "ok": true,
  "models": ["ollama:qwen2.5:3b"],
  "cth": false
}
```

**Response (Cloud API mode)**

```json
{
  "ok": true,
  "models": ["openai:gpt-4o-mini", "anthropic:claude-3-5-sonnet-latest"],
  "cth": false
}
```

**Response (Hybrid mode - local + cloud)**

```json
{
  "ok": true,
  "models": ["ollama:qwen2.5:3b", "openai:gpt-4o-mini"],
  "cth": false
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `ok` | boolean | Service health status |
| `models` | array[string] | List of configured model names |
| `cth` | boolean | Whether CTH (topological consciousness) library is available |

---

### GET /config

Get current system configuration and parameters.

**Response (Local LLM)**

```json
{
  "central_charge": 627,
  "n_anyons": 5,
  "models": ["ollama:qwen2.5:3b"]
}
```

**Response (Cloud API)**

```json
{
  "central_charge": 627,
  "n_anyons": 5,
  "models": ["openai:gpt-4o-mini", "anthropic:claude-3-5-sonnet-latest"]
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `central_charge` | integer | Central charge for topological field theory |
| `n_anyons` | integer | Number of anyons for phase coherence |
| `models` | array[string] | Configured model pool (ollama:*, openai:*, anthropic:*) |

---

### POST /ask

Query multiple LLMs with topological consensus and return a validated answer.

**Request Body**

```json
{
  "prompt": "Explain quantum entanglement for a software engineer.",
  "min_coherence": 0.80,
  "max_energy": 0.30,
  "return_all": true
}
```

**Request Fields**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | The question or prompt to send to all models |
| `min_coherence` | float | No | 0.80 | Minimum coherence threshold (0-1) for auto-response |
| `max_energy` | float | No | 0.30 | Maximum energy threshold (0-1) for hallucination detection |
| `return_all` | boolean | No | false | If true, return all model outputs in response |

**Response**

```json
{
  "answer": "Quantum entanglement is a phenomenon where...",
  "decision": "auto",
  "coherence": 0.87,
  "energy": 0.23,
  "weights": [0.52, 0.48],
  "model_names": ["openai:gpt-4o-mini", "anthropic:claude-3-5-sonnet-latest"],
  "all_outputs": [
    "Quantum entanglement is a phenomenon where...",
    "Quantum entanglement occurs when..."
  ]
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string \| null | The consensus answer (null if decision is "escalate") |
| `decision` | string | Decision type: `"auto"`, `"needs_citations"`, or `"escalate"` |
| `coherence` | float | Phase coherence score (0-1); higher = more agreement |
| `energy` | float | Cardy boundary energy (0-1); higher = more hallucination risk |
| `weights` | array[float] | Topological weights for each model's output |
| `model_names` | array[string] | Names of models that generated outputs |
| `all_outputs` | array[string] \| null | All model outputs (only if `return_all=true`) |

**Decision Logic**

- `"auto"`: High coherence (≥ min_coherence) AND low energy (≤ max_energy) → Return answer
- `"needs_citations"`: Medium coherence (≥ 0.5) AND medium energy (≤ 0.5) → Return answer with warning
- `"escalate"`: Low coherence OR high energy → Human review needed, answer is null

**Example**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the speed of light?",
    "min_coherence": 0.85,
    "max_energy": 0.25
  }'
```

---

### POST /rerank

Rerank documents using spectral coherence weights and query similarity.

**Request Body**

```json
{
  "query": "How do qubits differ from classical bits?",
  "docs": [
    {
      "id": "doc1",
      "text": "Quantum computing uses qubits that can exist in superposition...",
      "embedding": null
    },
    {
      "id": "doc2",
      "text": "Classical computing uses bits that are either 0 or 1...",
      "embedding": [0.1, 0.2, ...]
    }
  ],
  "trinary_threshold": 0.25,
  "alpha": 0.7,
  "beta": 0.3
}
```

**Request Fields**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | The search query |
| `docs` | array[Doc] | Yes | - | Documents to rerank |
| `docs[].id` | string | Yes | - | Unique document identifier |
| `docs[].text` | string | Yes | - | Document text content |
| `docs[].embedding` | array[float] \| null | No | null | Pre-computed embedding (computed if null) |
| `trinary_threshold` | float | No | 0.25 | Threshold for trinary quantization |
| `alpha` | float | No | 0.7 | Weight for query-document similarity |
| `beta` | float | No | 0.3 | Weight for document coherence |

**Response**

```json
{
  "ranked_ids": ["doc1", "doc2"],
  "scores": [0.92, 0.78]
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `ranked_ids` | array[string] | Document IDs in ranked order (best first) |
| `scores` | array[float] | Combined scores for each document (alpha*similarity + beta*coherence) |

**Example**

```bash
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quantum computing basics",
    "docs": [
      {"id": "1", "text": "Quantum computers use qubits..."},
      {"id": "2", "text": "Classical computers use transistors..."}
    ],
    "alpha": 0.8,
    "beta": 0.2
  }'
```

---

## ChaosRAG API

**Base URL**: `http://localhost:8001` (configurable via `CHAOS_RAG_PORT`)

The ChaosRAG API provides chaos-aware retrieval-augmented generation with PostgreSQL + pgvector, graph-based traversal, and Hilbert-Huang Transform (HHT) analysis.

### POST /chaos/rag/index

Index documents into the vector database and knowledge graph.

**Request Body**

```json
{
  "docs": [
    {
      "source": "quantum_computing.pdf",
      "kind": "research",
      "content": "Quantum computers use qubits that can exist in superposition...",
      "meta": {
        "author": "Alice Johnson",
        "year": 2024,
        "tags": ["quantum", "computing"]
      }
    }
  ]
}
```

**Request Fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `docs` | array[Doc] | Yes | Array of documents to index |
| `docs[].source` | string | No | Source identifier (e.g., filename, URL) |
| `docs[].kind` | string | No | Document type (e.g., "research", "code", "note") |
| `docs[].content` | string | Yes | Full text content of the document |
| `docs[].meta` | object | No | Arbitrary metadata (JSON object) |

**Response**

```json
{
  "inserted": 1
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `inserted` | integer | Number of documents successfully indexed |

**Example**

```bash
curl -X POST http://localhost:8001/chaos/rag/index \
  -H "Content-Type: application/json" \
  -d '{
    "docs": [{
      "source": "paper.pdf",
      "kind": "research",
      "content": "Quantum entanglement enables...",
      "meta": {"author": "Bob"}
    }]
  }'
```

---

### POST /chaos/telemetry

Push asset telemetry data for chaos routing decisions.

**Request Body**

```json
{
  "asset": "BTC",
  "realized_vol": 0.45,
  "entropy": 0.62,
  "mod_intensity_grad": 0.12,
  "router_noise": 0.05
}
```

**Request Fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset` | string | Yes | Asset identifier (e.g., "BTC", "ETH") |
| `realized_vol` | float | Yes | Realized volatility (0-1 range typical) |
| `entropy` | float | Yes | System entropy measure (0-1 range typical) |
| `mod_intensity_grad` | float | No | Modular intensity gradient |
| `router_noise` | float | No | Router noise level |

**Response**

```json
{
  "ok": true
}
```

**Example**

```bash
curl -X POST http://localhost:8001/chaos/telemetry \
  -H "Content-Type: application/json" \
  -d '{
    "asset": "ETH",
    "realized_vol": 0.38,
    "entropy": 0.55
  }'
```

---

### POST /chaos/hht/ingest

Ingest time series data and perform EEMD + Hilbert-Huang Transform analysis.

**Request Body**

```json
{
  "asset": "BTC",
  "x": [100.5, 101.2, 99.8, 102.1, ...],
  "ts": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", ...],
  "fs": 1.0,
  "max_imfs": 4,
  "ensemble": 30,
  "noise_std": 0.2,
  "amp_threshold_pct": 0.8
}
```

**Request Fields**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `asset` | string | Yes | - | Asset identifier |
| `x` | array[float] | Yes | - | Time series values |
| `ts` | array[string] | Yes | - | Timestamps (ISO 8601 format) |
| `fs` | float | No | 1.0 | Sampling frequency |
| `max_imfs` | integer | No | 4 | Maximum number of intrinsic mode functions |
| `ensemble` | integer | No | 30 | Ensemble size for EEMD |
| `noise_std` | float | No | 0.2 | Noise standard deviation for EEMD |
| `amp_threshold_pct` | float | No | 0.8 | Amplitude threshold percentile for burst detection |

**Response**

```json
{
  "ok": true,
  "imfs": 4
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `ok` | boolean | Success status |
| `imfs` | integer | Number of IMFs extracted |

---

### POST /chaos/graph/entangle

Create or update edges in the knowledge graph (temporal causality, relationships).

**Request Body**

```json
{
  "pairs": [
    ["uuid1", "uuid2"],
    ["uuid2", "uuid3"]
  ],
  "nesting_level": 1,
  "weight": 0.85,
  "attrs": {
    "edge_type": "step-1",
    "created_by": "system"
  }
}
```

**Request Fields**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `pairs` | array[array[string]] | Yes | - | Array of [source_uuid, destination_uuid] pairs |
| `nesting_level` | integer | No | 0 | Graph nesting level (for hierarchical structures) |
| `weight` | float | No | 1.0 | Edge weight (relevance, strength) |
| `attrs` | object | No | {} | Arbitrary edge attributes (JSON object) |

**Response**

```json
{
  "ok": true
}
```

**Error Response**

```json
{
  "error": "constraint violation: ..."
}
```

---

### GET /chaos/graph/:uuid

Retrieve a node and its connected edges from the knowledge graph.

**URL Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `uuid` | string | UUID of the node to retrieve |

**Response**

```json
{
  "node": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "label": "doc",
    "payload": {
      "doc_id": "550e8400-e29b-41d4-a716-446655440000",
      "snippet": "Quantum computing uses..."
    },
    "coords": [0.0, 0.0, 0.0],
    "unitary_tag": "identity",
    "created_at": "2024-01-15T10:30:00Z"
  },
  "edges": [
    {
      "src": "550e8400-e29b-41d4-a716-446655440000",
      "dst": "660e8400-e29b-41d4-a716-446655440001",
      "weight": 0.85,
      "nesting_level": 1,
      "attrs": {"edge_type": "step-1"}
    }
  ]
}
```

**Error Response** (404 Not Found)

```json
{
  "error": "not found"
}
```

**Example**

```bash
curl http://localhost:8001/chaos/graph/550e8400-e29b-41d4-a716-446655440000
```

---

### POST /chaos/rag/query

Query the knowledge base with chaos-aware routing (vector + graph + HHT retrieval).

**Request Body**

```json
{
  "q": "How does quantum entanglement work in quantum computing?",
  "k": 12
}
```

**Request Fields**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `q` | string | Yes | - | Query string |
| `k` | integer | No | 12 | Base number of results to retrieve |

**Response**

```json
{
  "router": {
    "stress": 0.42,
    "mix": {
      "vector": 0.65,
      "graph": 0.25,
      "hht": 0.10
    },
    "top_k": 10
  },
  "answer": "Quantum entanglement in quantum computing refers to...",
  "hits": [
    "{\"doc_id\": \"...\", \"snippet\": \"...\"}",
    "HHT {\"asset\": \"BTC\", \"features\": {...}}"
  ]
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `router.stress` | float | System stress level (0-1); drives retrieval strategy |
| `router.mix.vector` | float | Vector search weight in mixed retrieval |
| `router.mix.graph` | float | Graph traversal weight in mixed retrieval |
| `router.mix.hht` | float | HHT temporal analysis weight in mixed retrieval |
| `router.top_k` | integer | Adjusted top-k based on stress |
| `answer` | string | Generated answer from OpenAI chat model |
| `hits` | array[string] | Retrieved context snippets (serialized JSON) |

**Chaos Router Logic**

The system calculates stress based on telemetry:

```
stress = σ(1.8 × volatility + 1.5 × entropy + 0.8 × |gradient|)
```

- **Low stress** (< 0.3): Mostly vector search (90%), minimal graph/HHT
- **Medium stress** (0.3-0.6): Balanced mix (60% vector, 30% graph, 10% HHT)
- **High stress** (> 0.6): Heavy graph/HHT (10% vector, 50% graph, 40% HHT)

**Example**

```bash
curl -X POST http://localhost:8001/chaos/rag/query \
  -H "Content-Type: application/json" \
  -d '{"q": "Explain quantum superposition", "k": 8}'
```

---

## Error Responses

Both APIs use standard HTTP status codes:

| Status Code | Meaning |
|-------------|---------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 404 | Not Found (resource doesn't exist) |
| 500 | Internal Server Error |

**Error Response Format**

```json
{
  "error": "Description of what went wrong"
}
```

---

## Authentication

Currently, both APIs do not require authentication for local development. For production deployments, implement:

- API key authentication (via headers)
- OAuth 2.0 / JWT tokens
- Rate limiting

---

## Rate Limits

No rate limits are enforced in the current version. For production:

- Implement per-IP rate limiting
- Consider per-user quotas
- Use Redis for distributed rate limiting

---

## Client Libraries

### Python Client Examples

See the `examples/` directory for:

- `ask_client.py` - Topological consensus query client
- `rerank_client.py` - Document reranking client
- `chaos_rag_client.py` - ChaosRAG indexing and querying client

---

## Monitoring & Observability

Both services log to stdout. For production:

- Integrate with logging aggregators (e.g., ELK stack, Datadog)
- Add Prometheus metrics endpoints
- Implement distributed tracing (OpenTelemetry)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01 | Initial API documentation |

---

**For more information**, see:
- [README.md](README.md) - Platform overview
- [.env.example](.env.example) - Configuration reference
- [examples/](examples/) - Client code examples
