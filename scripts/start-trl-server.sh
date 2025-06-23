#!/bin/bash

# Start TRL Remote Server (Option B Architecture)
# Standalone HTTP server that launches 8-GPU workers on demand

set -e

echo "Starting TRL Remote Server..."

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
else
    echo "ERROR: Virtual environment not found. Run scripts/build.sh first"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo "Stopping TRL Remote Server..."
    pkill -f "http_server.py" 2>/dev/null || true
    pkill -f "training_worker.py" 2>/dev/null || true
    pkill -f "torchrun" 2>/dev/null || true
    echo "TRL Remote Server stopped"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Detect GPU configuration
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
    echo "Detected $GPU_COUNT GPU(s)"
else
    GPU_COUNT=0
    echo "No CUDA GPUs detected, using CPU"
fi

# Start standalone HTTP server
echo "Starting standalone HTTP server on port 8080..."
echo "Ready to launch $GPU_COUNT-GPU training workers on demand"
cd llama_stack_provider_trl_remote

python http_server.py 