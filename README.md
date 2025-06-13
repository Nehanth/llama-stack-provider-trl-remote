# TRL Provider for Llama Stack

A TRL (Transformer Reinforcement Learning) provider that integrates DPO (Direct Preference Optimization) training capabilities into the Llama Stack ecosystem as an inline external provider.

## Overview

- **Provider Type**: `inline::trl` (inline external provider)
- **API**: Post-training with DPO capabilities  
- **Implementation**: Production-ready DPO training using TRL library
- **Integration**: Full compatibility with Llama Stack protocol

## What is this?

This provider wraps the external TRL (Transformer Reinforcement Learning) library to provide DPO (Direct Preference Optimization) training through Llama Stack's unified API. It allows you to train language models using human preference data to improve their alignment and response quality.

## Getting Started

### Build and Run

1. **Setup environment:**
   ```bash
   ./scripts/prepare-env.sh
   ```

2. **Start the server:**
   ```bash
   ./scripts/run-direct.sh
   ```

The server will be available at `http://localhost:8321`

### Documentation

For complete setup and usage instructions, see the documentation in the `how_to_use/` directory.

The documentation covers:

- Dataset registration and training data format
- Running DPO training jobs
- Monitoring training progress
- Configuration options
- Troubleshooting common issues

## Project Structure

```
llama-stack-provider-trl/
├── llama_stack_provider_trl/           # Main package
│   ├── __init__.py                     # Provider entry point
│   ├── config.py                       # Configuration classes  
│   ├── post_training.py                # Main provider implementation
│   └── recipes/
│       └── dpo_training_single_device.py  # Core DPO training logic
├── how_to_use/                         # Documentation and examples
│   └──how_to_use.ipynb               # Interactive usage examples
├── providers.d/                        # Provider registration
│   └── inline/post_training/trl.yaml   # Provider specification
├── pyproject.toml                      # Package configuration
├── run.yaml                           # Runtime configuration
└── README.md                          # This file
```

## Dependencies

- `trl==0.18.1` - Transformer Reinforcement Learning library
- `transformers>=4.52.0` - Hugging Face Transformers
- `llama-stack>=0.2.3` - Llama Stack framework
- `torch` - PyTorch framework
- `datasets` - Dataset loading and processing

---

For detailed instructions, troubleshooting, and examples, see the documentation in the `examples/` directory. 
