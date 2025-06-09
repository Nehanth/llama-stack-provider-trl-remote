#!/bin/bash
set -e

echo "🚀 Starting TRL Provider server (direct mode)..."
echo "📍 Server will be available at: http://localhost:8321"
echo ""

# First prepare the environment with all dependencies
echo "🔧 Preparing environment with dependencies..."
./scripts/prepare-env.sh

# Activate the .venv and run the server
echo "🚀 Starting server with .venv..."
. .venv/bin/activate
python -m llama_stack.distribution.server.server --config run.yaml --port 8321