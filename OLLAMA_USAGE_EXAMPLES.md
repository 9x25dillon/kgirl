# Ollama Usage Examples for kgirl

Complete examples showing how to use kgirl with local Ollama LLMs.

## Prerequisites

```bash
# 1. Ollama installed and running
ollama serve

# 2. Models pulled
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# 3. kgirl configured for local
# .env already defaults to: MODELS=ollama:chat=qwen2.5:3b,embed=nomic-embed-text
```

---

## Example 1: Basic Query (Python)

```python
import requests

# Ask a question using local LLM
response = requests.post("http://localhost:8000/ask", json={
    "prompt": "Explain recursion in programming in simple terms",
    "min_coherence": 0.5,
    "max_energy": 0.5
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Model: {result['model_names']}")
print(f"Decision: {result['decision']}")
```

---

## Example 2: Batch Processing

```python
import requests
from concurrent.futures import ThreadPoolExecutor

queries = [
    "What is machine learning?",
    "Explain neural networks",
    "What is deep learning?",
    "How does backpropagation work?",
    "What are transformers in AI?"
]

def ask_question(prompt):
    response = requests.post("http://localhost:8000/ask", json={
        "prompt": prompt,
        "return_all": False
    })
    return response.json()

# Process 5 queries in parallel with local LLM
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(ask_question, queries))

for query, result in zip(queries, results):
    print(f"\nQ: {query}")
    print(f"A: {result.get('answer', 'N/A')[:100]}...")
```

---

## Example 3: Document Reranking

```python
import requests

# Rerank documents based on semantic relevance (using local embeddings!)
response = requests.post("http://localhost:8000/rerank", json={
    "query": "quantum computing applications",
    "docs": [
        {
            "id": "doc1",
            "text": "Quantum computers can factor large numbers efficiently using Shor's algorithm"
        },
        {
            "id": "doc2",
            "text": "Classical computers use bits that are either 0 or 1"
        },
        {
            "id": "doc3",
            "text": "Python is a popular programming language for data science"
        },
        {
            "id": "doc4",
            "text": "Quantum algorithms like Grover's search can speed up database queries"
        }
    ]
})

result = response.json()
print("Ranked by relevance:")
for rank, (doc_id, score) in enumerate(zip(result['ranked_ids'], result['scores']), 1):
    print(f"{rank}. {doc_id} (score: {score:.3f})")
```

---

## Example 4: Multi-Model Consensus (Local + Cloud)

```python
import requests
import os

# Configure hybrid mode in .env:
# MODELS=ollama:chat=qwen2.5:3b,embed=nomic-embed-text|openai:chat=gpt-4o-mini,embed=text-embedding-3-large
# OPENAI_API_KEY=sk-...

# Restart main.py, then query
response = requests.post("http://localhost:8000/ask", json={
    "prompt": "What are the key differences between quantum and classical computing?",
    "min_coherence": 0.80,
    "max_energy": 0.30,
    "return_all": True
})

result = response.json()
print(f"Models used: {result['model_names']}")
print(f"Coherence: {result['coherence']:.3f}")
print(f"Energy (hallucination risk): {result['energy']:.3f}")
print(f"\nConsensus answer: {result['answer']}")
print(f"\nAll responses:")
for i, output in enumerate(result['all_outputs'], 1):
    print(f"{i}. {output[:200]}...")
```

---

## Example 5: Using llm_adapters.py (Advanced)

```python
import asyncio
from llm_adapters import OllamaAdapter

async def main():
    # Create Ollama adapter
    ollama = OllamaAdapter(
        model="qwen2.5:3b",
        host="http://localhost:11434",
        timeout=30.0
    )

    # Generate text
    prompt = "Write a Python function to calculate fibonacci numbers"
    response = await ollama.generate(prompt, max_tokens=512, temperature=0.7)

    print("Generated code:")
    print(response)

    # Try streaming
    print("\n\nStreaming response:")
    stream = await ollama.generate(
        "Explain what this code does",
        max_tokens=256,
        temperature=0.7,
        stream=True
    )

    async for chunk in stream:
        print(chunk, end='', flush=True)

    print("\n\nDone!")

# Run
asyncio.run(main())
```

---

## Example 6: Dual LLM Orchestrator with Ollama

```python
import asyncio
from dual_llm_orchestrator import LocalLLM, HTTPConfig

async def main():
    # Configure Ollama as local LLM
    config = HTTPConfig(
        base_url="http://localhost:11434",
        model="qwen2.5:3b",
        timeout=60,
        mode="ollama"  # Use Ollama mode
    )

    # Create orchestrator
    llm = LocalLLM([config])

    # Generate
    prompt = "Explain the concept of recursion in 3 sentences"
    response = llm.generate(prompt, temperature=0.7, max_tokens=256)

    print(f"Response: {response}")

asyncio.run(main())
```

---

## Example 7: Curl Commands (Shell)

```bash
# Health check
curl http://localhost:8000/health

# Simple query
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the speed of light?",
    "min_coherence": 0.5,
    "max_energy": 0.5
  }'

# Document reranking
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "docs": [
      {"id": "1", "text": "Neural networks are used for deep learning"},
      {"id": "2", "text": "Python is a programming language"}
    ]
  }'

# Get configuration
curl http://localhost:8000/config
```

---

## Example 8: Continuous Load Testing

```python
import requests
import time
from threading import Thread

def continuous_queries():
    """Send queries continuously to test local LLM performance"""
    query_count = 0
    start_time = time.time()

    while True:
        try:
            response = requests.post("http://localhost:8000/ask", json={
                "prompt": f"Query {query_count}: Explain quantum computing",
                "return_all": False
            }, timeout=60)

            if response.status_code == 200:
                query_count += 1
                elapsed = time.time() - start_time
                qps = query_count / elapsed
                print(f"Queries: {query_count}, QPS: {qps:.2f}, Response time: {response.elapsed.total_seconds():.2f}s")
            else:
                print(f"Error: {response.status_code}")

        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(1)

# Start 3 threads to hammer the local LLM
threads = []
for i in range(3):
    t = Thread(target=continuous_queries, daemon=True)
    t.start()
    threads.append(t)

# Run until Ctrl+C
try:
    for t in threads:
        t.join()
except KeyboardInterrupt:
    print("\nStopped")
```

---

## Example 9: Compare Model Quality

```python
import requests

# Test same prompt with different models
models_configs = [
    "ollama:chat=qwen2.5:3b,embed=nomic-embed-text",
    "ollama:chat=llama3.2:3b,embed=nomic-embed-text",
    "ollama:chat=mistral:7b,embed=nomic-embed-text"
]

prompt = "Explain the difference between supervised and unsupervised learning"

for config in models_configs:
    # Update .env and restart main.py for each test
    print(f"\nTesting: {config}")
    print("-" * 60)

    response = requests.post("http://localhost:8000/ask", json={
        "prompt": prompt,
        "return_all": True
    })

    result = response.json()
    print(f"Model: {result['model_names'][0]}")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Coherence: {result['coherence']:.3f}")
```

---

## Example 10: Integration Test Script

```python
#!/usr/bin/env python3
"""Complete integration test for Ollama with kgirl"""

import requests
import sys

def test_ollama_integration():
    """Test all API endpoints with Ollama"""

    print("Testing kgirl + Ollama integration...")
    print("=" * 60)

    # 1. Health check
    print("\n1. Health Check")
    try:
        r = requests.get("http://localhost:8000/health")
        if r.status_code == 200:
            data = r.json()
            print(f"✅ Health check passed")
            print(f"   Models: {data['models']}")
            assert "ollama:" in str(data['models']), "Expected Ollama model"
        else:
            print(f"❌ Health check failed: {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

    # 2. Config check
    print("\n2. Config Check")
    try:
        r = requests.get("http://localhost:8000/config")
        data = r.json()
        print(f"✅ Config retrieved")
        print(f"   Central charge: {data['central_charge']}")
        print(f"   N anyons: {data['n_anyons']}")
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

    # 3. Ask endpoint
    print("\n3. Ask Endpoint")
    try:
        r = requests.post("http://localhost:8000/ask", json={
            "prompt": "What is 2+2?",
            "min_coherence": 0.5,
            "max_energy": 0.5
        })
        data = r.json()
        print(f"✅ Ask query successful")
        print(f"   Answer: {data['answer'][:100]}...")
        print(f"   Decision: {data['decision']}")
    except Exception as e:
        print(f"❌ Ask error: {e}")
        return False

    # 4. Rerank endpoint
    print("\n4. Rerank Endpoint")
    try:
        r = requests.post("http://localhost:8000/rerank", json={
            "query": "AI",
            "docs": [
                {"id": "1", "text": "Artificial intelligence systems"},
                {"id": "2", "text": "Python programming"}
            ]
        })
        data = r.json()
        print(f"✅ Rerank successful")
        print(f"   Ranked: {data['ranked_ids']}")
    except Exception as e:
        print(f"❌ Rerank error: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_ollama_integration()
    sys.exit(0 if success else 1)
```

---

## Performance Tips

1. **Model Selection:**
   - `qwen2.5:3b` - Best balance (2GB RAM, fast)
   - `llama3.2:1b` - Fastest (1GB RAM, good quality)
   - `mistral:7b` - Best quality (4GB RAM, slower)

2. **Concurrent Requests:**
   - Local LLM can handle 3-5 concurrent requests well
   - Use ThreadPoolExecutor for batch processing
   - Monitor with: `curl http://localhost:11434/api/ps`

3. **GPU Acceleration:**
   - Ollama automatically uses GPU if available
   - Check with: `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD)
   - 5-10x faster with GPU

---

## Troubleshooting

**"Cannot connect to Ollama"**
```bash
# Start Ollama
ollama serve

# Check it's running
curl http://localhost:11434/api/version
```

**"Model not found"**
```bash
# List models
ollama list

# Pull if missing
ollama pull qwen2.5:3b
```

**Slow responses**
```bash
# Use smaller model
ollama pull llama3.2:1b

# Update .env
OLLAMA_CHAT_MODEL=llama3.2:1b
```

---

## Next Steps

- Try the [full test suite](test_local_llm.py): `python test_local_llm.py`
- Read the [complete guide](LOCAL_LLM_SETUP.md)
- Explore [advanced features](DEEP_REPOSITORY_REVIEW.md)

**You now have a complete AI system running locally - no API keys, no costs, complete privacy!** 🎉
