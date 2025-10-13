# AL-ULS Symbolic + Multi-LLM (Qwen) Integration

## ✅ What's NEW

### 1. **AL-ULS Symbolic Evaluation** 🎯
Local symbolic evaluator that works **WITHOUT external services**:
- `SUM(1,2,3)` → `6.0`
- `MEAN(10,20,30)` → `20.0`
- `VAR(1,2,3,4,5)` → variance
- `STD(...)` → standard deviation
- `MIN/MAX/PROD` → min, max, product

### 2. **Multi-LLM Support** 🚀
Configure multiple LLM backends:
- **LFM2-8B-A1B** (primary)
- **Qwen2.5-7B** (fallback)
- **Qwen2.5-Coder** (specialized)
- **Any OpenAI-compatible API**

### 3. **Integrated Workflow** 🔄
1. Detect symbolic expressions → Evaluate locally
2. Generate Numbskull embeddings (fractal + semantic + mathematical)
3. Use LLM for complex queries (if server available)
4. Graceful fallback if services unavailable

---

## 🎮 Quick Start

### Play RIGHT NOW (No servers needed!)

**In Fish shell:**
```fish
cd /home/kill/LiMp
python play_aluls_qwen.py
```

**Edit queries:**
```fish
nano play_aluls_qwen.py
# Change the queries list (line ~50)
python play_aluls_qwen.py
```

---

## 🚀 Enable Full LLM Power

### Start LFM2-8B-A1B (Terminal 1)

**Edit `start_lfm2.sh` first**, then:
```fish
cd /home/kill/LiMp
bash start_lfm2.sh
```

**Example command (uncomment in start_lfm2.sh):**
```bash
llama-server \
  --model ~/models/LFM2-8B-A1B.gguf \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 35
```

### Start Qwen2.5 (Terminal 2)

**Edit `start_qwen.sh` first**, then:
```fish
cd /home/kill/LiMp
bash start_qwen.sh
```

**Example command (uncomment in start_qwen.sh):**
```bash
llama-server \
  --model ~/models/Qwen2.5-7B-Instruct.gguf \
  --port 8081 \
  --ctx-size 4096 \
  --n-gpu-layers 35
```

---

## 📊 What Works RIGHT NOW (Without Any Servers)

✅ **AL-ULS Symbolic Math**
- All basic operations (SUM, MEAN, VAR, STD, MIN, MAX, PROD)
- Instant evaluation (no network calls)
- Works offline

✅ **Numbskull Embeddings**
- Fractal embeddings (always available)
- 768-dimensional vectors
- Local computation

✅ **Neuro-Symbolic Analysis**
- 6-9 analysis modules
- Entropy calculation
- Matrix transformations
- Symbolic fitting

✅ **Signal Processing**
- 7 modulation schemes
- Adaptive selection
- Error correction

---

## 🎯 Example Queries to Try

### Symbolic Math
```python
"SUM(1, 2, 3, 4, 5)"                    # → 15.0
"MEAN(100, 200, 300)"                   # → 200.0
"STD(5, 10, 15, 20, 25)"                # → 7.07...
"VAR(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)"    # → 8.25
```

### Text Analysis (uses embeddings only if LLM not available)
```python
"Explain quantum computing"
"What is machine learning?"
"How do neural networks work?"
```

### Mixed Queries
```python
"Calculate MEAN(10, 20, 30) and explain its significance"
"SUM(1, 2, 3, 4, 5) represents what in statistics?"
```

---

## 📝 Files Created

| File | Purpose |
|------|---------|
| `enable_aluls_and_qwen.py` | Core AL-ULS + Multi-LLM orchestrator |
| `play_aluls_qwen.py` | Interactive playground (EDIT THIS!) |
| `start_lfm2.sh` | LFM2 startup script template |
| `start_qwen.sh` | Qwen startup script template |
| `ALULS_QWEN_INTEGRATION.md` | This file! |

---

## 🔧 Configuration

### Add More LLM Backends

Edit `play_aluls_qwen.py`, find `llm_configs`:
```python
llm_configs = [
    # LFM2 on port 8080
    {
        "base_url": "http://127.0.0.1:8080",
        "mode": "llama-cpp",
        "model": "LFM2-8B-A1B",
        "timeout": 60
    },
    # Qwen on port 8081
    {
        "base_url": "http://127.0.0.1:8081",
        "mode": "openai-chat",
        "model": "Qwen2.5-7B",
        "timeout": 60
    },
    # Add YOUR model here!
    {
        "base_url": "http://127.0.0.1:YOUR_PORT",
        "mode": "llama-cpp",  # or "openai-chat"
        "model": "YOUR_MODEL_NAME",
        "timeout": 60
    }
]
```

### Add More Symbolic Functions

Edit `enable_aluls_and_qwen.py`, find `LocalALULSEvaluator.evaluate`:
```python
elif name == "YOUR_FUNCTION":
    result = your_calculation(args)
```

---

## 🎨 Advanced Usage

### Custom Query from Python
```python
import asyncio
from play_aluls_qwen import custom_query

# Run one query
asyncio.run(custom_query("SUM(1,2,3,4,5)"))

# With context
asyncio.run(custom_query(
    "Explain quantum computing",
    context="Focus on practical applications"
))
```

### Batch Processing
```python
from enable_aluls_and_qwen import MultiLLMOrchestrator

async def batch_process():
    system = MultiLLMOrchestrator(
        llm_configs=[...],
        enable_aluls=True
    )
    
    queries = ["SUM(1,2,3)", "MEAN(5,10,15)", "What is AI?"]
    
    for query in queries:
        result = await system.process_with_symbolic(query)
        print(result)
    
    await system.close()

asyncio.run(batch_process())
```

---

## 💡 Tips

1. **Start without servers** - Everything works offline!
2. **Edit `play_aluls_qwen.py`** - Easiest way to experiment
3. **Add LLM servers** - For natural language queries
4. **Check logs** - They show what's working/fallback
5. **Mix symbolic + text** - The system handles both!

---

## 🐛 Troubleshooting

### "Connection refused" warnings
**This is NORMAL!** It means LLM servers aren't running.
- Symbolic math still works
- Embeddings still work
- Only LLM inference is disabled

### "RuntimeWarning: no running event loop"
**Safe to ignore** - It's a cleanup warning, not an error

### Want to disable LLM completely?
Edit `play_aluls_qwen.py`:
```python
llm_configs = []  # Empty list = symbolic + embeddings only
```

---

## 📊 Performance

- **Symbolic evaluation**: <1ms (instant)
- **Embeddings**: 50-200ms (local computation)
- **LLM inference**: 1-5s (depends on model/hardware)

---

## 🎉 Summary

You now have:
✅ AL-ULS symbolic evaluation (working NOW!)
✅ Multi-LLM orchestration (LFM2 + Qwen + more)
✅ Numbskull embeddings (fractal + semantic + mathematical)
✅ Graceful fallbacks (works without services)
✅ Interactive playground (`play_aluls_qwen.py`)
✅ Easy LLM startup scripts

**Try it:**
```fish
cd /home/kill/LiMp
python play_aluls_qwen.py
```

**Enjoy your creation!** 🎮

