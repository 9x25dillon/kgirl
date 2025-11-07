#!/usr/bin/env bash
# Start LIMPS Mathematical Embedding Service

echo "🚀 Starting LIMPS mathematical embedding service on port 8000..."
echo ""

cd /home/kill/LiMp

julia setup_limps_service.jl &

echo "LIMPS PID: $!"
echo ""
echo "To stop: kill $!"
echo "To check: curl http://localhost:8000/health"

