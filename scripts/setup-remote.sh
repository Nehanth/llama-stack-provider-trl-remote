#!/bin/sh

# Setup script for TRL Remote Provider
echo "Setting up TRL Remote Provider..."

# Install remote provider package
pip install -e ./llama_stack_provider_trl_remote

# Build the remote distribution
llama stack build --config build_remote.yaml

echo "Remote TRL Provider setup complete!"
echo ""
echo "To use:"
echo "1. Start the remote service: python llama_stack_provider_trl_remote/scripts/start_service.py"
echo "2. Start the client: llama stack run --image-type venv --image-name trl-remote-client run_remote.yaml" 