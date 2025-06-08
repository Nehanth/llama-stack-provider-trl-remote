# TRL Provider for Llama Stack

A TRL (Transformer Reinforcement Learning) provider that integrates DPO (Direct Preference Optimization) training capabilities into the Llama Stack ecosystem as an inline external provider.

## Overview

- **Provider Type**: `inline::trl` (inline external provider)
- **API**: Post-training with DPO capabilities  
- **Implementation**: Production-ready DPO training using TRL library
- **Integration**: Full compatibility with Llama Stack protocol

## What is this?

This provider wraps the external TRL (Transformer Reinforcement Learning) library to provide DPO (Direct Preference Optimization) training through Llama Stack's unified API. It allows you to train language models using human preference data to improve their alignment and response quality.

**Key Features:**
- ✅ DPO training with TRL library
- ✅ Async job management with real-time status monitoring  
- ✅ Checkpoint saving and artifact tracking
- ✅ Resource usage monitoring
- ✅ Single-node training optimized for Llama Stack

**Note on SFT (Supervised Fine-Tuning):**
This provider implements a `supervised_fine_tune()` method that raises `NotImplementedError`. This exists only to satisfy Llama Stack's PostTraining protocol requirements - all providers must implement both methods even if they don't support both training types. This TRL provider specializes in DPO training only.

## Getting Started

📖 **For complete setup and usage instructions, see [how_to_run.md](how_to_run.md)**

The guide covers:
- Installation and environment setup
- Starting the Llama Stack server  
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
├── providers.d/                        # Provider registration
│   └── inline/post_training/trl.yaml   # Provider specification
├── pyproject.toml                      # Package configuration
├── run.yaml                           # Runtime configuration
├── how_to_run.md                      # Complete setup guide
└── README.md                          # This file
```

## Dependencies

- `trl==0.18.1` - Transformer Reinforcement Learning library
- `transformers>=4.52.0` - Hugging Face Transformers
- `llama-stack>=0.2.3` - Llama Stack framework
- `torch` - PyTorch framework
- `datasets` - Dataset loading and processing

---

**For detailed instructions, troubleshooting, and examples, see [how_to_run.md](how_to_run.md)** 
