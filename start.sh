#!/usr/bin/env bash
# LocalAI Platform — Linux/Mac launcher

set -e
cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Please install Python 3.10+."
    exit 1
fi

# Run in background
nohup python3 start.py > /tmp/localai.log 2>&1 &
PID=$!
echo "LocalAI Platform started (PID $PID)"
echo "Logs: /tmp/localai.log"

# Wait for server then open browser
sleep 6
if command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:3000/api/localai/model-manager" &
elif command -v open &>/dev/null; then
    open "http://localhost:3000/api/localai/model-manager" &
fi
