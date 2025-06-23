#!/bin/bash

# Start Llama Stack Client Only  
# This script starts only the Llama Stack client in one terminal
# Make sure the TRL Remote Server is already running on port 8080

set -e

echo "🦙 Starting Llama Stack Client..."

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
    echo "Stopping Llama Stack Client..."
    pkill -f "llama_stack.distribution.server" 2>/dev/null || true
    echo "Llama Stack Client stopped"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Wait for TRL service to be ready
echo "Checking if TRL Remote Service is available on port 8080..."
for i in {1..30}; do
    if curl -s http://localhost:8080/health >/dev/null 2>&1; then
        echo "✅ TRL Remote Service is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ ERROR: TRL Remote Service not found on port 8080"
        echo "Please start the TRL server first with: ./scripts/start-trl-server.sh"
        exit 1
    fi
    echo "Waiting for TRL service... ($i/30)"
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
        echo "✅ Llama Stack Client is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ ERROR: Llama Stack client failed to start"
        exit 1
    fi
    sleep 1
done

echo ""
echo "🦙 Llama Stack Client is RUNNING!"
echo "Llama Stack Client: http://localhost:8321"
echo "API Documentation: http://localhost:8321/docs"
echo "Connected to TRL Remote Service: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop the client..."

# Wait for the client process
wait $LLS_PID 