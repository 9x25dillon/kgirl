#!/usr/bin/env bash
# Start Qwen2.5-7B LLM Server
# Run this in Terminal 2

echo "🚀 Starting Qwen2.5-7B on port 8081..."
echo ""
echo "Make sure you have the model file (Qwen2.5-7B.gguf) in your models directory"
echo "Adjust the path below if needed:"
echo ""

# Option 1: If you have llama.cpp llama-server
# llama-server \
#   --model ~/models/Qwen2.5-7B-Instruct.gguf \
#   --port 8081 \
#   --ctx-size 4096 \
#   --n-gpu-layers 35

# Option 2: If you use text-generation-webui
# cd ~/text-generation-webui
# python server.py \
#   --model Qwen2.5-7B-Instruct \
#   --api \
#   --listen-port 8081

# Option 3: If you use ollama
# ollama serve &
# ollama run qwen2.5:7b --port 8081

echo "📝 CONFIGURE YOUR COMMAND ABOVE and uncomment it"
echo ""
echo "After starting, test with:"
echo "  curl http://127.0.0.1:8081/health"

