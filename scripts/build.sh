#!/bin/sh

# This script is used to create a virtual environment for the Llama Stack TRL Provider

echo "Building TRL Remote Provider..."

echo "Creating virtual environment..."
rm -r .venv || true
python3 -m venv .venv
. .venv/bin/activate
echo "Virtual environment activated"

echo "Installing dependencies..."
pip install --upgrade pip

# Should probably be in the default dependencies for llama-stack but is not -
# the reference provider is loaded when no telemetry section is configured
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

# Install some dependencies not pulled by llama-stack for some reason
pip install aiosqlite fastapi uvicorn

pip install -e ./llama_stack_provider_trl_remote

# Build the distribution
echo "Building distribution..."
llama stack build --config build.yaml

echo "TRL Remote Provider build complete!"
echo "Run './scripts/run.sh' to start the services" 