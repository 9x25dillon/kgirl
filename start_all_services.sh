#!/usr/bin/env bash
# Complete Service Startup Script
# Starts ALL optional services for full LiMp integration

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           🚀 STARTING ALL SERVICES                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Service check function
check_service() {
    local name=$1
    local port=$2
    local url=$3
    
    if curl -s --max-time 2 "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name${NC} (port $port)"
        return 0
    else
        echo -e "${YELLOW}⚠️  $name${NC} (port $port) - Not running"
        return 1
    fi
}

echo "Checking current service status..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

check_service "Eopiez (Semantic)" 8001 "http://localhost:8001/health" || EOPIEZ_DOWN=1
check_service "LIMPS (Mathematical)" 8000 "http://localhost:8000/health" || LIMPS_DOWN=1
check_service "Ollama (LLM)" 11434 "http://localhost:11434/api/tags" || OLLAMA_DOWN=1

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           📋 SERVICE STARTUP INSTRUCTIONS                            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

if [ -n "$EOPIEZ_DOWN" ]; then
    echo "🔴 Eopiez (Semantic Embeddings) - Port 8001"
    echo "   Terminal 1:"
    echo "   cd ~/aipyapp/Eopiez"
    echo "   python api.py --port 8001"
    echo ""
fi

if [ -n "$LIMPS_DOWN" ]; then
    echo "🔴 LIMPS (Mathematical Embeddings) - Port 8000"
    echo "   Terminal 2:"
    echo "   cd ~/aipyapp/9xdSq-LIMPS-FemTO-R1C/limps"
    echo "   julia --project=. -e 'using LIMPS; LIMPS.start_limps_server(8000)'"
    echo ""
fi

if [ -n "$OLLAMA_DOWN" ]; then
    echo "🔴 Ollama (LLM Server) - Port 11434"
    echo "   Terminal 3:"
    echo "   sudo systemctl start ollama"
    echo "   ollama serve"
    echo ""
    echo "   Then download a model:"
    echo "   ollama pull qwen2.5:3b"
    echo ""
fi

if [ -z "$EOPIEZ_DOWN" ] && [ -z "$LIMPS_DOWN" ] && [ -z "$OLLAMA_DOWN" ]; then
    echo -e "${GREEN}✅ ALL SERVICES RUNNING!${NC}"
    echo ""
    echo "You can now run your playground:"
    echo "  python master_playground.py"
    echo ""
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "After starting the services above, run this script again to verify."
    echo "Or run: python master_playground.py"
    echo ""
fi

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           📚 QUICK REFERENCE                                         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Service Ports:"
echo "  • Eopiez:  http://localhost:8001"
echo "  • LIMPS:   http://localhost:8000"
echo "  • Ollama:  http://localhost:11434"
echo ""
echo "Check status anytime: bash start_all_services.sh"
echo "Run playground: python master_playground.py"
echo ""

