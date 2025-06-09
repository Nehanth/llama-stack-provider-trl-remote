#!/bin/bash
set -e

echo "🔧 Preparing TRL Provider environment..."

# This script is used to create a virtual environment for the Llama Stack TRL Provider

rm -r .venv || true
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip

# Should probably be in the default dependencies for llama-stack but is not -
# the reference provider is loaded when no telemetry section is configured
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

# Install some dependencies not pulled by llama-stack for some reason
pip install aiosqlite fastapi uvicorn

pip install -e .

echo "✅ Environment ready! All dependencies installed."
echo ""
echo "🚀 You can now run: ./scripts/run-direct.sh" 