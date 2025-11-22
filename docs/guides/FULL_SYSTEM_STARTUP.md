# 🚀 Full System Startup Guide - All Services Running

## 🎯 **Goal: Get ALL 5 Services Running**

This guide will help you start ALL optional services so you have **100% system power**.

---

## 📋 **Current Status Check**

Run this first to see what's running:
```bash
cd /home/kill/LiMp
bash start_all_services.sh
```

---

## 🚀 **Service Startup - Step by Step**

### **Service 1: Ollama (LLM) - Priority 1** ⭐

**This is the most important - gives you LLM inference!**

**Terminal 1:**
```bash
# Install Ollama (if not installed)
sudo pacman -S ollama

# Start the service
sudo systemctl start ollama

# Enable on boot (optional)
sudo systemctl enable ollama

# Download a model (choose ONE)
ollama pull qwen2.5:3b      # Fast, 2GB
# OR
ollama pull qwen2.5:7b      # Better quality, 4.5GB
# OR  
ollama pull llama3.2:latest # Alternative, 2GB

# Test it works
ollama run qwen2.5:3b "Hello, world!"
```

**Verification:**
```bash
curl http://localhost:11434/api/tags
# Should return JSON with your models
```

---

### **Service 2: LIMPS (Mathematical) - Priority 2**

**Enhances mathematical embeddings**

**Check if you have LIMPS:**
```bash
ls ~/aipyapp/9xdSq-LIMPS-FemTO-R1C/limps
```

**If directory exists - Terminal 2:**
```bash
cd ~/aipyapp/9xdSq-LIMPS-FemTO-R1C/limps

# Check Julia is installed
julia --version

# Start LIMPS server
julia --project=. -e 'using LIMPS; LIMPS.start_limps_server(8000)'
```

**If directory doesn't exist:**
```bash
# Skip for now - system works without it
echo "LIMPS not available, skipping"
```

**Verification:**
```bash
curl http://localhost:8000/health
# Should return health status
```

---

### **Service 3: Eopiez (Semantic) - Priority 3**

**Enhances semantic embeddings**

**Check if you have Eopiez:**
```bash
ls ~/aipyapp/Eopiez/api.py
```

**If file exists - Terminal 3:**
```bash
cd ~/aipyapp/Eopiez

# Activate venv if it exists
source venv/bin/activate

# Start Eopiez server
python api.py --port 8001
```

**If file doesn't exist:**
```bash
# Skip for now - system works without it
echo "Eopiez not available, skipping"
```

**Verification:**
```bash
curl http://localhost:8001/health
# Should return health status
```

---

## ✅ **Verify All Services**

Run the status checker:
```bash
cd /home/kill/LiMp
bash start_all_services.sh
```

**Should see:**
```
✅ AL-ULS Symbolic       (local, always available)
✅ Fractal Embeddings     (local, always available)
✅ Semantic Embeddings    (Eopiez on port 8001)      ← If you started it
✅ Mathematical Embeddings (LIMPS on port 8000)      ← If you started it
✅ LLM Inference          (Ollama on port 11434)     ← Most important!

Active: 5/5 services  ← This means EVERYTHING is running!
```

---

## 🎮 **Run Your Complete System**

Once services are running:

```bash
cd /home/kill/LiMp

# Ultra-clean demo
./play

# Interactive mode (RECOMMENDED!)
./play --interactive
```

**In interactive mode, try:**
```
🎮 Query: SUM(100, 200, 300)
# ✅ Symbolic: 600.0000
# ✅ Embeddings: ['semantic', 'mathematical', 'fractal'] (768D)

🎮 Query: What is quantum computing?
# ✅ Embeddings: ['semantic', 'mathematical', 'fractal'] (768D)
# 🤖 LLM: Quantum computing is a revolutionary computing paradigm...

🎮 Query: status
# Shows current service status

🎮 Query: exit
```

---

## 🎯 **Quick Start (Minimum for LLM)**

If you only want LLM working (skip Eopiez/LIMPS for now):

**Terminal 1:**
```bash
sudo pacman -S ollama
sudo systemctl start ollama
ollama pull qwen2.5:3b
```

**Your terminal:**
```bash
cd /home/kill/LiMp
./play --interactive
```

**Done!** You'll have:
- ✅ AL-ULS symbolic (2/5)
- ✅ Fractal embeddings (2/5)
- ✅ LLM inference (3/5)

That's 60% power and the most important features!

---

## 📊 **Service Priority**

| Priority | Service | Impact | Setup Time |
|----------|---------|--------|------------|
| 🔥 Critical | Ollama (LLM) | Huge | 5 min |
| ⚡ High | LIMPS (Math) | Medium | 2 min |
| 💡 Medium | Eopiez (Semantic) | Small | 2 min |
| ✅ Always | AL-ULS | - | Built-in |
| ✅ Always | Fractal | - | Built-in |

**Recommendation:** Start with Ollama first!

---

## 🔧 **Troubleshooting**

### Ollama Not Starting
```bash
# Check service status
sudo systemctl status ollama

# View logs
sudo journalctl -u ollama -f

# Try manual start
ollama serve
```

### Model Download Slow
```bash
# Use smaller model
ollama pull qwen2.5:3b  # Only 2GB

# Check disk space
df -h
```

### Port Already in Use
```bash
# Check what's using the port
sudo lsof -i :11434  # Ollama
sudo lsof -i :8000   # LIMPS
sudo lsof -i :8001   # Eopiez

# Kill if needed
kill -9 <PID>
```

### Service Won't Connect
```bash
# Test connectivity
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8000/health      # LIMPS
curl http://localhost:8001/health      # Eopiez

# Check firewall
sudo iptables -L
```

---

## 💡 **Pro Tips**

### 1. Use tmux for Persistence
```bash
# Start services in tmux sessions
tmux new -s ollama
ollama serve
# Ctrl+B, D to detach

tmux new -s limps
cd ~/aipyapp/9xdSq-LIMPS-FemTO-R1C/limps && julia --project=. -e 'using LIMPS; LIMPS.start_limps_server(8000)'
# Ctrl+B, D to detach

# List sessions
tmux ls

# Reattach
tmux attach -t ollama
```

### 2. Auto-Start Ollama on Boot
```bash
sudo systemctl enable ollama
sudo systemctl start ollama

# Verify it's enabled
systemctl is-enabled ollama
```

### 3. Quick Service Restart
```bash
# Stop all services
# Ctrl+C in each terminal

# Or kill
pkill -f "ollama serve"
pkill -f "api.py"
pkill -f "julia.*LIMPS"

# Restart
bash start_all_services.sh  # Shows startup commands
```

---

## 🎉 **Complete Setup Summary**

### What You Need to Do:

**Minimum (60% power):**
1. Install Ollama: `sudo pacman -S ollama`
2. Start Ollama: `sudo systemctl start ollama`
3. Download model: `ollama pull qwen2.5:3b`
4. Run: `./play --interactive`

**Full Power (100%):**
1. Do minimum setup above
2. Start LIMPS (if available): See Terminal 2 commands
3. Start Eopiez (if available): See Terminal 3 commands
4. Run: `./play --interactive`
5. Type `status` to verify all 5/5 services active!

---

## 🚀 **Ready to Start!**

**Let's get Ollama running first:**

```bash
# Install
sudo pacman -S ollama

# Start
sudo systemctl start ollama

# Download model
ollama pull qwen2.5:3b

# Test
ollama run qwen2.5:3b "Hello!"

# Run your system
cd /home/kill/LiMp
./play --interactive
```

**That's it!** Your cohesive, integrated system will be fully operational! 🎉

