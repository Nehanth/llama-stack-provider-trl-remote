#!/bin/bash

# TRL Remote Provider - Run Script
# This script starts both the remote TRL service and Llama Stack client

set -e

echo "Starting TRL Remote Provider..."

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
else
    echo "ERROR: Virtual environment not found. Run scripts/build.sh first"
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up..."
    pkill -f "python.*service.py" 2>/dev/null || true
    pkill -f "llama_stack.distribution.server" 2>/dev/null || true
    wait
    echo "Cleanup complete"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Start TRL Remote Service in background
echo "Starting TRL Remote Service on port 8080..."
cd llama_stack_provider_trl_remote
python service.py &
TRL_PID=$!
cd ..

# Wait for TRL service to start
echo "Waiting for TRL service to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8080/health >/dev/null 2>&1; then
        echo "TRL Remote Service is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: TRL service failed to start"
        exit 1
    fi
    sleep 1
done

# Start Llama Stack Client
echo "Starting Llama Stack Client on port 8321..."
python -m llama_stack.distribution.server.server --yaml-config run.yaml &
LLS_PID=$!

# Wait for Llama Stack to start
echo "Waiting for Llama Stack client to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8321/v1/providers >/dev/null 2>&1; then
        echo "Llama Stack Client is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Llama Stack client failed to start"
        exit 1
    fi
    sleep 1
done

echo ""
echo "TRL Remote Provider is RUNNING!"
echo "TRL Remote Service: http://localhost:8080"
echo "Llama Stack Client: http://localhost:8321"
echo "API Documentation: http://localhost:8080/docs"
echo ""
echo "Press Ctrl+C to stop all services..."

# Wait for background processes
wait $TRL_PID $LLS_PID 