# TRL Remote Provider for Llama Stack

A TRL (Transformer Reinforcement Learning) remote provider that integrates DPO (Direct Preference Optimization) training capabilities into the Llama Stack ecosystem as a remote external provider.

## Overview

- **Provider Type**: `remote::trl_remote` (remote external provider)
- **API**: Post-training with DPO capabilities  
- **Implementation**: Production-ready multi-GPU DPO training using TRL library
- **Integration**: Full compatibility with Llama Stack protocol
- **Multi-GPU**: Automatic scaling across available GPUs with FSDP support

## What is this?

This remote provider wraps the external TRL (Transformer Reinforcement Learning) library to provide scalable DPO (Direct Preference Optimization) training through Llama Stack's unified API. It runs as a separate HTTP service and automatically scales across available GPUs using FSDP (Fully Sharded Data Parallel) for efficient multi-card training. This allows you to train language models using human preference data to improve their alignment and response quality.

## Getting Started

### Build and Run

1. **Build the distribution:**
   ```bash
   ./scripts/build.sh
   ```

2. **Start the TRL remote service:**
   ```bash
   ./scripts/start-trl-server.sh
   ```

3. **Start the Llama Stack client:**
   ```bash
   ./scripts/start-llama-stack.sh
   ```

The TRL service will be available at `http://localhost:8080` and the Llama Stack client at `http://localhost:8321`

### Documentation

For complete setup and usage instructions, see the documentation in the `examples/` directory.

The documentation covers:

- Dataset registration and training data format
- Running remote DPO training jobs
- Multi-GPU training configuration
- Monitoring training progress and job status
- Configuration options
- Troubleshooting common issues

### GPU Monitoring

For real-time GPU usage monitoring during multi-GPU training, use `nvitop`:

```bash
pip install nvitop
nvitop
```

This provides a real-time view of GPU utilization, memory usage, and running processes across all available GPUs during DPO training.

## Project Structure

```
llama-stack-provider-trl/
├── llama_stack_provider_trl_remote/    # Remote provider package
│   ├── __init__.py                     # Provider entry point
│   ├── adapter.py                      # HTTP client adapter
│   ├── config.py                       # Configuration classes  
│   ├── http_server.py                  # Standalone HTTP server
│   ├── training_worker.py              # Multi-GPU training worker
│   └── recipes/
│       └── dpo_training_multicard.py  # Multi-GPU DPO training logic
├── examples/                           # Documentation and examples
│   └── examples.ipynb                 # Interactive usage examples
├── scripts/                           # Build and run scripts
│   ├── build.sh                       # Build distribution
│   ├── start-trl-server.sh           # Start TRL service
│   └── start-llama-stack.sh          # Start Llama Stack client
├── providers.d/                       # Provider registration
│   └── remote/post_training/trl.yaml  # Provider specification
├── pyproject.toml                     # Package configuration
├── build.yaml                        # Build configuration
├── run.yaml                          # Runtime configuration
└── README.md                         # This file
```

## Dependencies

- `trl==0.18.1` - Transformer Reinforcement Learning library
- `transformers==4.52.4` - Hugging Face Transformers
- `llama-stack>=0.2.3` - Llama Stack framework
- `torch` - PyTorch framework with CUDA support
- `datasets` - Dataset loading and processing
- `fastapi` - HTTP service framework
- `uvicorn` - ASGI server
- `aiohttp` - Async HTTP client

---

For detailed instructions, troubleshooting, and examples, see the documentation in the `examples/` directory. 
