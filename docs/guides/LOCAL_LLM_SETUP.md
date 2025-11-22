# 🚀 Local LLM Setup Guide - No API Keys Required!

**kgirl** now runs completely locally using **Ollama** - no cloud API keys needed!

This guide will help you set up kgirl to work with local LLMs in under 5 minutes.

---

## Why Local LLMs?

✅ **No API Keys** - Completely free, no credit card required
✅ **Privacy** - Your data never leaves your machine
✅ **No Rate Limits** - Use as much as you want
✅ **Offline** - Works without internet connection
✅ **Fast** - Local inference can be faster than API calls
✅ **Cost-Effective** - Zero ongoing costs

---

## Quick Start (5 Minutes)

### Step 1: Install Ollama

**macOS / Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from: https://ollama.ai/download

**Verify Installation:**
```bash
ollama --version
```

### Step 2: Start Ollama Service

```bash
ollama serve
```

Keep this terminal open. Ollama will run in the background on `http://localhost:11434`.

### Step 3: Pull Required Models

Open a **new terminal** and run:

```bash
# Chat model (3B parameters, ~2GB download)
ollama pull qwen2.5:3b

# Embedding model (~270MB download)
ollama pull nomic-embed-text
```

**Alternative Models:**
```bash
# Larger chat models (better quality, more RAM needed):
ollama pull llama3.2:3b        # Meta's Llama 3.2 (3B)
ollama pull mistral:7b         # Mistral 7B
ollama pull llama3.2:1b        # Smaller/faster (1B)

# Check available models:
ollama list
```

### Step 4: Configure kgirl

Create a `.env` file in the kgirl directory:

```bash
cp .env.example .env
```

The default configuration already uses Ollama - no changes needed! But you can customize:

```bash
# .env
USE_LOCAL_LLM=true
MODELS=ollama:chat=qwen2.5:3b,embed=nomic-embed-text

# Optional: Use different models
# OLLAMA_CHAT_MODEL=llama3.2:3b
# OLLAMA_EMBED_MODEL=nomic-embed-text
```

### Step 5: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Start kgirl!

```bash
python main.py
```

You should see:
```
✓ Loaded sentence-transformers for embedding fallback
✓ Loaded Ollama adapter: qwen2.5:3b (local, no API key required)

✓ Successfully loaded 1 model adapter(s)
  Models: ['ollama:qwen2.5:3b']

INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Testing Your Setup

### Quick Test

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in one sentence",
    "min_coherence": 0.5,
    "max_energy": 0.5
  }'
```

You should get a response with an answer!

### Health Check

```bash
curl http://localhost:8000/health
```

Expected output:
```json
{
  "ok": true,
  "models": ["ollama:qwen2.5:3b"],
  "cth": false
}
```

### Python Test

```python
import requests

response = requests.post("http://localhost:8000/ask", json={
    "prompt": "What is the meaning of life?",
    "return_all": True
})

print(response.json())
```

---

## Advanced Configuration

### Multi-Model Local Consensus

Run multiple local models for consensus (improves quality):

```bash
# Pull additional models
ollama pull llama3.2:3b
ollama pull mistral:7b

# Update .env
MODELS=ollama:chat=qwen2.5:3b,embed=nomic-embed-text|ollama:chat=llama3.2:3b,embed=nomic-embed-text
```

### Hybrid: Local + Cloud

Use local models as primary with cloud fallback:

```bash
# .env
MODELS=ollama:chat=qwen2.5:3b,embed=nomic-embed-text|openai:chat=gpt-4o-mini,embed=text-embedding-3-large
OPENAI_API_KEY=sk-...
```

This gives you:
- Local model for speed and privacy
- Cloud model for quality comparison and consensus
- Topological coherence across both

### Custom Ollama Host

If running Ollama on a different machine:

```bash
OLLAMA_HOST=http://192.168.1.100:11434
```

---

## Recommended Models

### Chat Models

| Model | Size | RAM Needed | Speed | Quality | Use Case |
|-------|------|------------|-------|---------|----------|
| `qwen2.5:3b` | 2GB | 4GB | ⚡⚡⚡ | ⭐⭐⭐ | **Default, balanced** |
| `llama3.2:1b` | 1GB | 2GB | ⚡⚡⚡⚡ | ⭐⭐ | Ultra-fast, low RAM |
| `llama3.2:3b` | 2GB | 4GB | ⚡⚡⚡ | ⭐⭐⭐ | Meta's latest |
| `mistral:7b` | 4GB | 8GB | ⚡⚡ | ⭐⭐⭐⭐ | High quality |
| `llama3.2:8b` | 5GB | 8GB | ⚡⚡ | ⭐⭐⭐⭐ | Best quality |

### Embedding Models

| Model | Size | Dimensions | Use Case |
|-------|------|------------|----------|
| `nomic-embed-text` | 270MB | 768D | **Default, fast** |
| `mxbai-embed-large` | 669MB | 1024D | Higher quality |
| `all-minilm` | 45MB | 384D | Ultra-light |

---

## Troubleshooting

### "Failed to initialize Ollama adapter"

**Problem:** Ollama not running
**Solution:**
```bash
# Start Ollama in a separate terminal
ollama serve
```

### "model 'qwen2.5:3b' not found"

**Problem:** Model not downloaded
**Solution:**
```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### "No models configured!"

**Problem:** Configuration issue
**Solution:**
1. Check `.env` file exists: `ls .env`
2. Verify MODELS setting: `grep MODELS .env`
3. Make sure it's set to: `MODELS=ollama:chat=qwen2.5:3b,embed=nomic-embed-text`

### Ollama uses too much RAM

**Solution:** Use smaller models
```bash
ollama pull llama3.2:1b  # Only 1B parameters, uses ~2GB RAM
```

Update `.env`:
```bash
OLLAMA_CHAT_MODEL=llama3.2:1b
```

### Slow response times

**Solutions:**
1. Use smaller/faster model: `llama3.2:1b`
2. Enable GPU acceleration (if you have NVIDIA GPU):
   ```bash
   # Ollama automatically uses GPU if available
   nvidia-smi  # Check if GPU detected
   ```
3. Increase Ollama's context window:
   ```bash
   ollama run qwen2.5:3b --context 2048
   ```

---

## Performance Comparison

### Local vs Cloud

| Metric | Ollama (Local) | OpenAI GPT-4 | Anthropic Claude |
|--------|----------------|--------------|------------------|
| **Cost** | Free | $0.01-0.03/1K tokens | $0.015-0.075/1K tokens |
| **Privacy** | ✅ Complete | ❌ Cloud | ❌ Cloud |
| **Speed** | ~50-200 tokens/sec | ~20-50 tokens/sec | ~20-40 tokens/sec |
| **Offline** | ✅ Yes | ❌ No | ❌ No |
| **Quality** | ⭐⭐⭐ (3B model) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Rate Limits** | None | Yes | Yes |

**Best Practice:** Use local for development/testing, cloud for production when quality is critical.

---

## Next Steps

Once your local LLM is working:

1. **Try the full system:** `./START_NOW.sh`
2. **Explore the playground:** `python master_playground.py`
3. **Build knowledge bases:** Use the quantum knowledge processing features
4. **Scale up:** Add more local models for multi-model consensus

---

## FAQ

**Q: Do I need a GPU?**
A: No, but it helps. Ollama works on CPU, but GPU is 5-10x faster.

**Q: How much RAM do I need?**
A: Minimum 4GB for `qwen2.5:3b`, 8GB recommended for larger models.

**Q: Can I use multiple models at once?**
A: Yes! Use the MODELS setting with `|` separator for multi-model consensus.

**Q: Are local models as good as GPT-4?**
A: Not quite, but they're surprisingly capable for most tasks. For critical applications, use hybrid mode with cloud fallback.

**Q: Can I fine-tune the models?**
A: Yes, Ollama supports custom model creation. See: https://ollama.ai/library

**Q: Does this work on Raspberry Pi?**
A: Yes! Use `llama3.2:1b` for best performance on low-power devices.

---

## Resources

- **Ollama Documentation:** https://ollama.ai/docs
- **Available Models:** https://ollama.ai/library
- **Ollama GitHub:** https://github.com/ollama/ollama
- **kgirl Documentation:** See `README.md`

---

**Built with ❤️ for the open-source community. No API keys, no tracking, no limits.**
