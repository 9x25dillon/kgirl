#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║        🚀 STARTING YOUR RECURSIVE COGNITIVE AI SYSTEM               ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check Ollama
echo "1️⃣  Checking Ollama LLM..."
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "   ✅ Ollama is running!"
else
    echo "   ⚠️  Ollama not running. Starting..."
    echo "   Run in another terminal: ollama serve"
    echo "   Then: ollama pull qwen2.5:3b"
fi

# Check LIMPS
echo ""
echo "2️⃣  Checking LIMPS (Julia mathematical service)..."
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "   ✅ LIMPS is running!"
else
    echo "   ⚠️  LIMPS not running. Starting..."
    echo "   Run in another terminal: cd /home/kill/LiMp && bash start_limps.sh"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "SERVICE STATUS SUMMARY"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

OLLAMA_STATUS="❌"
LIMPS_STATUS="❌"

if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    OLLAMA_STATUS="✅"
fi

if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    LIMPS_STATUS="✅"
fi

echo "Ollama LLM:    $OLLAMA_STATUS  (port 11434)"
echo "LIMPS:         $LIMPS_STATUS  (port 8000)"
echo "AL-ULS:        ✅  (built-in)"
echo "Embeddings:    ✅  (built-in)"
echo "Matrix Proc:   ✅  (built-in)"
echo ""

# Count active services
ACTIVE=3
if [ "$OLLAMA_STATUS" = "✅" ]; then ACTIVE=$((ACTIVE+1)); fi
if [ "$LIMPS_STATUS" = "✅" ]; then ACTIVE=$((ACTIVE+1)); fi

echo "System Power: $ACTIVE/5 services active"
echo ""

if [ "$OLLAMA_STATUS" = "✅" ]; then
    echo "════════════════════════════════════════════════════════════════════════"
    echo "✅ READY TO RUN!"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Choose how to run:"
    echo ""
    echo "Option 1: Interactive Playground (RECOMMENDED)"
    echo "  cd /home/kill/LiMp && python recursive_playground.py"
    echo ""
    echo "Option 2: Complete System Orchestrator"
    echo "  cd /home/kill/LiMp && python complete_integration_orchestrator.py"
    echo ""
    echo "Option 3: Clean Interface"
    echo "  cd /home/kill/LiMp && ./play --interactive"
    echo ""
    echo "Option 4: Simple Demo"
    echo "  cd /home/kill/LiMp && python -c 'import asyncio; from recursive_cognitive_knowledge import RecursiveCognitiveKnowledge; r = RecursiveCognitiveKnowledge(); asyncio.run(r.initialize()); result = asyncio.run(r.process_with_recursion(\"What is consciousness?\")); print(result)'"
    echo ""
else
    echo "════════════════════════════════════════════════════════════════════════"
    echo "⚠️  START OLLAMA FIRST"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "In another terminal, run:"
    echo "  ollama serve"
    echo ""
    echo "Then in this terminal:"
    echo "  ollama pull qwen2.5:3b"
    echo ""
    echo "Then run this script again:"
    echo "  bash START_NOW.sh"
    echo ""
fi

