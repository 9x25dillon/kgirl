# kgirl API Client Examples

This directory contains Python client scripts demonstrating how to interact with the kgirl platform APIs.

## Prerequisites

```bash
# Install dependencies
pip install requests

# Ensure services are running
python ../main.py          # Topological Consensus API (port 8000)
julia ../server.jl         # ChaosRAG API (port 8001)
```

## Available Clients

### 1. `ask_client.py` - Multi-Model Consensus Queries

Query multiple LLMs with topological consensus and hallucination detection.

**Basic Usage:**

```bash
# Simple query
python ask_client.py "What is quantum entanglement?"

# Strict thresholds
python ask_client.py "Explain black holes" --min-coherence 0.9 --max-energy 0.2

# Show all model outputs
python ask_client.py "How does photosynthesis work?" --verbose
```

**Parameters:**

- `--min-coherence`: Minimum coherence threshold (default: 0.80)
- `--max-energy`: Maximum energy threshold (default: 0.30)
- `-v, --verbose`: Show all model outputs
- `--url`: Custom API URL

**Exit Codes:**

- `0`: Auto (high confidence)
- `1`: Needs citations (medium confidence)
- `2`: Escalate (low confidence, human review needed)

---

### 2. `rerank_client.py` - Document Reranking

Rerank documents using spectral coherence weights and query similarity.

**Basic Usage:**

```bash
# Demo mode with sample documents
python rerank_client.py "quantum computing basics" --demo

# Rerank text files
python rerank_client.py "machine learning" doc1.txt doc2.txt doc3.txt

# Adjust similarity vs coherence weights
python rerank_client.py "AI research" *.txt --alpha 0.8 --beta 0.2
```

**Parameters:**

- `--demo`: Use built-in sample documents
- `--alpha`: Weight for query-document similarity (default: 0.7)
- `--beta`: Weight for document coherence (default: 0.3)
- `--trinary-threshold`: Trinary quantization threshold (default: 0.25)
- `--url`: Custom API URL

**Use Cases:**

- Semantic search result reranking
- Document clustering and organization
- Context selection for RAG systems

---

### 3. `chaos_rag_client.py` - ChaosRAG Operations

Interact with the chaos-aware RAG system for indexing, querying, and telemetry.

**Commands:**

#### Index Documents

```bash
# Index a single document
python chaos_rag_client.py index document.txt

# Index with custom metadata
python chaos_rag_client.py index paper.pdf --source "research paper" --kind research
```

#### Query Knowledge Base

```bash
# Basic query
python chaos_rag_client.py query "What is quantum entanglement?"

# Retrieve more results
python chaos_rag_client.py query "Explain AI" --k 20
```

#### Push Telemetry

```bash
# Update system telemetry (affects routing decisions)
python chaos_rag_client.py telemetry --asset BTC --vol 0.45 --entropy 0.62

# With additional parameters
python chaos_rag_client.py telemetry --asset ETH --vol 0.38 --entropy 0.55 --grad 0.12 --noise 0.05
```

**Parameters:**

- `--k`: Number of results to retrieve (default: 12)
- `--asset`: Asset identifier (BTC, ETH, etc.)
- `--vol`: Realized volatility (0-1 range)
- `--entropy`: System entropy (0-1 range)
- `--grad`: Modular intensity gradient
- `--noise`: Router noise level

**Chaos Router Logic:**

The system adjusts retrieval strategy based on telemetry:

- **Low stress** (< 0.3): Mostly vector search (90%)
- **Medium stress** (0.3-0.6): Balanced mix (60% vector, 30% graph, 10% HHT)
- **High stress** (> 0.6): Heavy graph/HHT (10% vector, 50% graph, 40% HHT)

---

## Making the Scripts Executable

```bash
chmod +x ask_client.py rerank_client.py chaos_rag_client.py

# Then run directly
./ask_client.py "Your question"
./rerank_client.py "search query" --demo
./chaos_rag_client.py query "What is quantum computing?"
```

---

## Environment Variables

All clients respect the following environment variables:

```bash
# Topological Consensus API port
export MAIN_API_PORT=8000

# ChaosRAG API port
export CHAOS_RAG_PORT=8001
```

Or override with `--url` flag:

```bash
python ask_client.py "question" --url http://api.example.com:8000
```

---

## Example Workflows

### Workflow 1: Index and Query Documents

```bash
# 1. Index documents
python chaos_rag_client.py index quantum_paper.txt --kind research
python chaos_rag_client.py index ml_notes.txt --kind notes

# 2. Query the indexed knowledge
python chaos_rag_client.py query "Compare quantum computing and machine learning"
```

### Workflow 2: Multi-Model Consensus with Context

```bash
# 1. Get consensus answer
python ask_client.py "Explain the uncertainty principle" --verbose

# 2. Rerank related documents for additional context
python rerank_client.py "uncertainty principle quantum" doc1.txt doc2.txt doc3.txt
```

### Workflow 3: Stress-Aware RAG

```bash
# 1. Push high-stress telemetry
python chaos_rag_client.py telemetry --asset BTC --vol 0.85 --entropy 0.92

# 2. Query (will use heavy graph/HHT retrieval)
python chaos_rag_client.py query "Analyze current market regime"

# 3. Push low-stress telemetry
python chaos_rag_client.py telemetry --asset BTC --vol 0.15 --entropy 0.22

# 4. Query again (will use mostly vector search)
python chaos_rag_client.py query "Analyze current market regime"
```

---

## Troubleshooting

### Connection Refused

```
Error: Connection refused
```

**Solution:** Ensure services are running:

```bash
# Terminal 1
python ../main.py

# Terminal 2
julia ../server.jl
```

### Port Already in Use

```
Error: Address already in use
```

**Solution:** Change ports in `.env` or kill existing processes:

```bash
# Find processes
lsof -i :8000
lsof -i :8001

# Kill and restart
kill <PID>
```

### Missing Dependencies

```
ModuleNotFoundError: No module named 'requests'
```

**Solution:** Install dependencies:

```bash
pip install requests
```

---

## Further Reading

- [API.md](../API.md) - Complete API reference
- [README.md](../README.md) - Platform overview
- [.env.example](../.env.example) - Configuration options

---

**Happy querying!** 🚀
