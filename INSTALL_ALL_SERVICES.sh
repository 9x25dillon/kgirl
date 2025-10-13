#!/usr/bin/env bash
# Complete Service Installation and Startup Guide

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║        🚀 COMPLETE SERVICE INSTALLATION                              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

echo "STEP 1: Ollama (LLM Service)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Ollama is already installed at: /usr/bin/ollama"
echo ""
echo "Run these commands in your terminal:"
echo "  sudo systemctl start ollama"
echo "  ollama pull qwen2.5:3b"
echo ""
echo "Press Enter after Ollama is running..."
read

echo ""
echo "STEP 2: LIMPS (Mathematical Embeddings)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting LIMPS service on port 8000..."
echo ""

# Start LIMPS in background
bash start_limps.sh

echo ""
echo "STEP 3: Verify All Services"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sleep 3
bash start_all_services.sh

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║        ✅ SERVICES READY!                                            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Run your recursive cognitive system:"
echo "  cd /home/kill/LiMp"
echo "  python recursive_playground.py"
echo ""

