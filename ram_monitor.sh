#!/bin/bash
# RAM Usage Monitor Script

echo "=== RAM Usage Monitor ==="
echo "Timestamp: $(date)"
echo ""

# Basic memory info
echo "Memory Usage:"
free -h
echo ""

# Memory details
echo "Detailed Memory Info:"
cat /proc/meminfo | grep -E "(MemTotal|MemAvailable|MemFree|Buffers|Cached|SwapTotal|SwapFree|SwapCached)"
echo ""

# Top memory consumers
echo "Top 10 Memory Consumers:"
ps aux --sort=-%mem | head -11
echo ""

# Memory usage percentage
TOTAL=$(free | grep Mem | awk '{print $2}')
USED=$(free | grep Mem | awk '{print $3}')
PERCENT=$((USED * 100 / TOTAL))
echo "RAM Usage: ${PERCENT}% (${USED}KB / ${TOTAL}KB)"
echo ""

# Available memory in GB
AVAIL=$(free -g | grep Mem | awk '{print $7}')
echo "Available RAM: ${AVAIL}GB"
echo ""

# System load
echo "System Load:"
uptime
echo ""

# Disk usage
echo "Disk Usage:"
df -h | grep -E "(Filesystem|/dev/)"
