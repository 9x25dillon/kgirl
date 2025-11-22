# Complete Startup Guide - All Optional Components

This guide shows you **step-by-step** how to enable ALL optional components.

---

## 📋 **What We'll Enable**

1. **PyTorch** - For CoCo full features (TA-ULS, Holographic Memory, Quantum)
2. **Eopiez** - For semantic embeddings (better text understanding)
3. **LIMPS** - For mathematical embeddings (better math processing)
4. **LFM2-8B-A1B** - Primary LLM for inference
5. **Qwen2.5-7B** - Fallback/alternative LLM

---

## 🎯 **Option 1: Quick Start (Just PyTorch)**

If you only want to enable CoCo full features:

```fish
# Install PyTorch
pip install torch

# Run the system
cd /home/kill/LiMp
python coco_integrated_playground.py --interactive
```

**Done!** This enables:
- ✅ Full CoCo Cognitive Organism
- ✅ TA-ULS Transformer
- ✅ Holographic Memory
- ✅ Quantum Processor

---

## 🚀 **Option 2: Full Power (All Services)**

Follow these steps to enable EVERYTHING:

---

### **STEP 1: Install PyTorch**

Open your main terminal:

```fish
cd /home/kill/LiMp

# Install PyTorch
pip install torch

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed!')"
```

**Expected output:**
```
PyTorch 2.x.x installed!
```

---

### **STEP 2: Start Eopiez (Semantic Embeddings)**

Open a **NEW terminal** (Terminal 1):

```fish
# Navigate to Eopiez directory
cd ~/aipyapp/Eopiez

# Start Eopiez server on port 8001
python api.py --port 8001
```

**Expected output:**
```
✅ Eopiez semantic embedding server started on port 8001
```

**Keep this terminal open!**

---

### **STEP 3: Start LIMPS (Mathematical Embeddings)**

Open a **NEW terminal** (Terminal 2):

```fish
# Navigate to LIMPS directory
cd ~/aipyapp/9xdSq-LIMPS-FemTO-R1C/limps

# Start LIMPS server on port 8000
julia --project=. -e 'using LIMPS; LIMPS.start_limps_server(8000)'
```

**Expected output:**
```
✅ LIMPS mathematical server started on port 8000
```

**Keep this terminal open!**

---

### **STEP 4: Start LFM2-8B-A1B (Primary LLM)**

Open a **NEW terminal** (Terminal 3):

#### Option A: Using llama.cpp

```fish
# Navigate to your models directory
cd ~/models  # Or wherever your models are

# Start llama-server with LFM2
llama-server \
  --model LFM2-8B-A1B.gguf \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 35 \
  --threads 8
```

#### Option B: Using text-generation-webui

```fish
cd ~/text-generation-webui

python server.py \
  --model LFM2-8B-A1B \
  --api \
  --listen-port 8080 \
  --auto-devices
```

#### Option C: Using Ollama

```fish
# Start Ollama service
ollama serve &

# Run LFM2 model
ollama run LFM2-8B-A1B
```

**Expected output:**
```
✅ LLM server running on http://127.0.0.1:8080
```

**Keep this terminal open!**

---

### **STEP 5: Start Qwen2.5-7B (Fallback LLM) [OPTIONAL]**

Open a **NEW terminal** (Terminal 4):

#### Option A: Using llama.cpp

```fish
cd ~/models

llama-server \
  --model Qwen2.5-7B-Instruct.gguf \
  --port 8081 \
  --ctx-size 4096 \
  --n-gpu-layers 35 \
  --threads 8
```

#### Option B: Using Ollama

```fish
ollama run qwen2.5:7b --port 8081
```

**Expected output:**
```
✅ Qwen LLM server running on http://127.0.0.1:8081
```

**Keep this terminal open!**

---

### **STEP 6: Test the Complete System**

Open your **MAIN terminal** (or a new Terminal 5):

```fish
cd /home/kill/LiMp

# Run the interactive playground
python coco_integrated_playground.py --interactive
```

**You should see:**
```
✅ CoCo organism ready (3-level cognitive architecture)
✅ AL-ULS symbolic evaluator initialized
✅ Multi-LLM orchestrator with 2 backends
✅ Numbskull pipeline initialized
Active components: 4/4  ← All components active!
```

---

### **STEP 7: Try These Queries**

In the interactive mode, try:

```
Query: SUM(100, 200, 300, 400, 500)
# ✅ Symbolic: 1500.00
# ✅ Embeddings: ['semantic', 'mathematical', 'fractal']

Query: What is quantum computing?
# ✅ Embeddings: ['semantic', 'mathematical', 'fractal'] (768D)
# 🤖 LLM: Quantum computing uses quantum mechanics to process...

Query: Explain neural networks in simple terms
# 🤖 LLM: Neural networks are computational models inspired by...

Query: MEAN(10, 20, 30, 40, 50)
# ✅ Symbolic: 30.00

Query: demo
# Runs full demonstration

Query: exit
# Exits interactive mode
```

---

## 📊 **Verify All Services Are Running**

Run this check script:

```fish
cd /home/kill/LiMp

# Create quick check script
cat << 'EOF' > check_services.sh
#!/usr/bin/env bash
echo "Checking all services..."
echo ""

echo "1. Eopiez (port 8001):"
curl -s http://127.0.0.1:8001/health && echo "✅ Running" || echo "❌ Not running"

echo "2. LIMPS (port 8000):"
curl -s http://127.0.0.1:8000/health && echo "✅ Running" || echo "❌ Not running"

echo "3. LFM2 (port 8080):"
curl -s http://127.0.0.1:8080/health && echo "✅ Running" || echo "❌ Not running"

echo "4. Qwen (port 8081):"
curl -s http://127.0.0.1:8081/health && echo "✅ Running" || echo "❌ Not running"

echo "5. PyTorch:"
python -c "import torch; print('✅ Installed')" 2>/dev/null || echo "❌ Not installed"
EOF

chmod +x check_services.sh
bash check_services.sh
```

**Expected output when all services are running:**
```
1. Eopiez (port 8001): ✅ Running
2. LIMPS (port 8000): ✅ Running
3. LFM2 (port 8080): ✅ Running
4. Qwen (port 8081): ✅ Running
5. PyTorch: ✅ Installed
```

---

## 🎯 **Summary of Terminal Setup**

When fully running, you'll have these terminals open:

```
Terminal 1: Eopiez          (port 8001) - Semantic embeddings
Terminal 2: LIMPS           (port 8000) - Mathematical embeddings
Terminal 3: LFM2-8B-A1B     (port 8080) - Primary LLM
Terminal 4: Qwen2.5-7B      (port 8081) - Fallback LLM [optional]
Terminal 5: Your playground             - Interactive mode
```

---

## 🔧 **Troubleshooting**

### Port Already in Use
```fish
# Find what's using the port
lsof -i :8000
lsof -i :8001
lsof -i :8080
lsof -i :8081

# Kill the process if needed
kill -9 <PID>
```

### Model Not Found
If llama-server can't find your model:
```fish
# Find your models
find ~ -name "*.gguf" -type f

# Use the full path in the command
llama-server --model /full/path/to/LFM2-8B-A1B.gguf --port 8080
```

### Julia/LIMPS Not Found
```fish
# Check if Julia is installed
julia --version

# If not, install:
# Visit https://julialang.org/downloads/

# Install LIMPS dependencies
cd ~/aipyapp/9xdSq-LIMPS-FemTO-R1C/limps
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Eopiez Not Found
```fish
# Check if Eopiez directory exists
ls ~/aipyapp/Eopiez

# If not, you may need to clone/install it
# Check your project documentation
```

### Out of Memory
If LLM servers fail due to memory:
```fish
# Reduce GPU layers
llama-server \
  --model your-model.gguf \
  --port 8080 \
  --n-gpu-layers 20  # Reduce from 35
  --ctx-size 2048    # Reduce from 4096
```

---

## 💡 **Quick Reference Commands**

### Start Everything (All Terminals)

**Terminal 1:**
```fish
cd ~/aipyapp/Eopiez && python api.py --port 8001
```

**Terminal 2:**
```fish
cd ~/aipyapp/9xdSq-LIMPS-FemTO-R1C/limps && julia --project=. -e 'using LIMPS; LIMPS.start_limps_server(8000)'
```

**Terminal 3:**
```fish
llama-server --model ~/models/LFM2-8B-A1B.gguf --port 8080 --ctx-size 4096 --n-gpu-layers 35
```

**Terminal 4 (optional):**
```fish
llama-server --model ~/models/Qwen2.5-7B-Instruct.gguf --port 8081 --ctx-size 4096 --n-gpu-layers 35
```

**Terminal 5 (Your playground):**
```fish
cd /home/kill/LiMp && python coco_integrated_playground.py --interactive
```

### Stop Everything

Press `Ctrl+C` in each terminal to stop the services gracefully.

---

## 🎉 **You're Done!**

With all services running, you have the **COMPLETE UNIFIED SYSTEM**:

- ✅ AL-ULS symbolic evaluation
- ✅ Semantic embeddings (Eopiez)
- ✅ Mathematical embeddings (LIMPS)
- ✅ Fractal embeddings (local)
- ✅ LFM2-8B-A1B inference
- ✅ Qwen2.5-7B fallback
- ✅ Full CoCo organism (PyTorch)
- ✅ All 40+ components active!

**Enjoy your creation!** 🚀

