#!/usr/bin/env bash
# Start LFM2-8B-A1B LLM Server
# Run this in Terminal 1

echo "🚀 Starting LFM2-8B-A1B on port 8080..."
echo ""
echo "Make sure you have the model file (LFM2-8B-A1B.gguf) in your models directory"
echo "Adjust the path below if needed:"
echo ""

# Option 1: If you have llama.cpp llama-server
# llama-server \
#   --model ~/models/LFM2-8B-A1B.gguf \
#   --port 8080 \
#   --ctx-size 4096 \
#   --n-gpu-layers 35

# Option 2: If you use text-generation-webui
# cd ~/text-generation-webui
# python server.py \
#   --model LFM2-8B-A1B \
#   --api \
#   --listen-port 8080

# Option 3: If you use ollama
# ollama serve &
# ollama run LFM2-8B-A1B

echo "📝 CONFIGURE YOUR COMMAND ABOVE and uncomment it"
echo ""
echo "After starting, test with:"
echo "  curl http://127.0.0.1:8080/health"

