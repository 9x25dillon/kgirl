# 🚀 Quick Ollama Setup Guide

## Step-by-Step Installation

### STEP 1: Install Ollama
```bash
sudo pacman -S ollama
```

### STEP 2: Start Ollama Service
```bash
# Start the service
sudo systemctl start ollama

# Enable it to start on boot (optional)
sudo systemctl enable ollama

# Check status
sudo systemctl status ollama
```

### STEP 3: Download a Model

**Option A: Small & Fast (Recommended)**
```bash
ollama pull qwen2.5:3b
# Size: ~2GB, Speed: Fast, Quality: Good
```

**Option B: Medium Quality**
```bash
ollama pull qwen2.5:7b
# Size: ~4.5GB, Speed: Medium, Quality: Better
```

**Option C: Llama 3.2**
```bash
ollama pull llama3.2:latest
# Size: ~2GB, Speed: Fast, Quality: Good
```

### STEP 4: Test It

**Quick test:**
```bash
ollama run qwen2.5:3b "What is quantum computing?"
```

**Interactive chat:**
```bash
ollama run qwen2.5:3b
# Type your questions
# Type /bye to exit
```

### STEP 5: Connect to Your Playground

Ollama runs on `http://localhost:11434` by default.

**Update your config (if needed):**
```python
# In your playground configs, Ollama uses this format:
llm_configs = [
    {
        "base_url": "http://127.0.0.1:11434",
        "mode": "openai-chat",  # Ollama is OpenAI compatible
        "model": "qwen2.5:3b",
        "timeout": 60
    }
]
```

**Test with your playground:**
```bash
cd /home/kill/LiMp
python play_aluls_qwen.py
# Or
python coco_integrated_playground.py --interactive
```

---

## 🎯 Recommended Models

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| `qwen2.5:3b` | 2GB | ⚡ Fast | Quick queries, testing |
| `qwen2.5:7b` | 4.5GB | 🔥 Medium | Better responses |
| `llama3.2:latest` | 2GB | ⚡ Fast | Alternative option |
| `qwen2.5:14b` | 9GB | 🐌 Slow | Best quality (if RAM permits) |

---

## ✅ Verification

**Check if Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```

Should return a JSON list of your models.

**Test generation:**
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:3b",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

---

## 🔧 Troubleshooting

**Service not starting:**
```bash
sudo systemctl status ollama
sudo journalctl -u ollama -f
```

**Can't connect:**
```bash
# Check if port is open
netstat -tulpn | grep 11434

# Or with ss
ss -tulpn | grep 11434
```

**Out of memory:**
- Use smaller models (3b instead of 7b)
- Close other applications
- Check: `free -h`

---

## 🎊 You're Done!

Once installed, your system will have:
- ✅ Local LLM server running
- ✅ Models ready to use
- ✅ Full integration with playgrounds
- ✅ No more "LLM not available" messages!

Enjoy your complete AI system! 🚀

