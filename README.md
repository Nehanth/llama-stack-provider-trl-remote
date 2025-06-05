# TRL Provider for Llama Stack

**Status: Fully Operational - DPO Training Provider Successfully Integrated**

A TRL (Transformer Reinforcement Learning) provider that integrates DPO (Direct Preference Optimization) training capabilities into the Llama Stack ecosystem as an inline external provider.

## Overview

- **Provider Type**: `inline::trl` (inline external provider)
- **API**: Post-training with DPO capabilities  
- **Implementation**: 1,850+ lines of production-ready DPO training code
- **Integration**: Full compatibility with Llama Stack protocol

## Quick Start

### Prerequisites
- Python 3.10+
- Llama Stack installed
- CUDA (optional, for GPU training)

### Installation

1. **Build the environment:**
```bash
llama stack build --template experimental-post-training --image-type venv --image-name trl-post-training
```

2. **Install dependencies:**
```bash
source trl-post-training/bin/activate
pip uninstall torchao -y
rm -rf ./trl-post-training/lib/python3.10/site-packages/torchao*
pip install trl==0.18.1 transformers==4.52.4
uv pip install -e . --python ./trl-post-training/bin/python --force-reinstall --no-cache
```

3. **Start the server:**
```bash
llama stack run --image-type venv --image-name trl-post-training run.yaml
```

### Verification

The server should start successfully with these log messages:
- `INFO Loaded inline provider spec for inline::trl`
- `Uvicorn running on http://['::', '0.0.0.0']:8321`

Check provider registration:
```bash
curl -s http://localhost:8321/v1/providers | jq '.data[] | select(.api=="post_training")'
```

## Configuration

The provider configuration is defined in `run.yaml`:

```yaml
providers:
  post_training:
  - provider_id: trl
    provider_type: inline::trl
    config:
      device: "cpu"                    # Device: "cpu" or "cuda"
      dpo_beta: 0.1                   # DPO beta parameter
      use_reference_model: true       # Enable reference model
      max_seq_length: 2048           # Maximum sequence length
      gradient_checkpointing: false   # Memory optimization
      logging_steps: 10              # Logging frequency
      warmup_ratio: 0.1              # Learning rate warmup ratio
      weight_decay: 0.01             # Weight decay coefficient
```

## API Usage

### Start DPO Training

```bash
curl -X POST http://localhost:8321/v1/post-training/preference-optimize \
  -H "Content-Type: application/json" \
  -d '{
    "job_uuid": "dpo-training-001",
    "finetuned_model": "/path/to/your/model",
    "algorithm_config": {
      "type": "dpo",
      "reward_scale": 1.0,
      "reward_clip": 5.0
    },
    "training_config": {
      "n_epochs": 1,
      "max_steps_per_epoch": 100,
      "batch_size": 4,
      "learning_rate": 1e-6,
      "dataset": "your-dataset-id"
    }
  }'
```

### Check Training Status

```bash
curl "http://localhost:8321/v1/post-training/job/status?job_uuid=dpo-training-001"
```

### Available Endpoints

- `POST /v1/post-training/preference-optimize` - Start DPO training job
- `GET /v1/post-training/job/status` - Check training job status  
- `GET /v1/post-training/job/artifacts` - Retrieve training artifacts
- `POST /v1/post-training/job/cancel` - Cancel running job
- `GET /v1/post-training/jobs` - List all training jobs
- `GET /v1/providers` - List registered providers

## Project Structure

```
llama-stack-provider-trl/
├── llama_stack_provider_trl/           # Main package
│   ├── __init__.py                     # Provider entry point (82 lines)
│   ├── config.py                       # Configuration classes (179 lines)
│   ├── post_training.py                # Main implementation (456 lines)
│   └── recipes/
│       └── dpo_training_single_device.py  # Core DPO training (1,134 lines)
├── providers.d/                        # Provider registration
│   └── inline/post_training/trl.yaml   # Provider specification
├── pyproject.toml                      # Package configuration
├── run.yaml                           # Runtime configuration
├── test_dpo_data.json                 # Sample training data
└── README.md                          # Documentation
```

## Data Format

Training data should follow this preference format (see `test_dpo_data.json`):

```json
{
  "data": [
    {
      "prompt": "What is machine learning?",
      "chosen": "Machine learning is a branch of artificial intelligence...",
      "rejected": "Machine learning is just computers doing math stuff."
    }
  ]
}
```

## Troubleshooting

### Import Errors
If you encounter `ModuleNotFoundError` or import issues, re-run the dependency installation:

```bash
source trl-post-training/bin/activate
pip uninstall torchao -y
rm -rf ./trl-post-training/lib/python3.10/site-packages/torchao*
pip install trl==0.18.1 transformers==4.52.4
uv pip install -e . --python ./trl-post-training/bin/python --force-reinstall --no-cache
```

### Port Conflicts
If port 8321 is in use, modify the `server.port` setting in `run.yaml` or kill existing processes:

```bash
lsof -ti:8321 | xargs kill -9
```

### Memory Issues
For large models, enable gradient checkpointing in the provider configuration:

```yaml
config:
  gradient_checkpointing: true
  device: "cuda"  # Use GPU if available
```

## Development

### Testing Locally

```bash
# Activate environment
source trl-post-training/bin/activate

# Install in development mode
pip install -e .

# Run tests (if available)
python -m pytest tests/
```

### Code Structure

- **`post_training.py`**: Main provider implementation with async job management
- **`config.py`**: Pydantic configuration classes for type safety
- **`recipes/dpo_training_single_device.py`**: Core DPO training logic using TRL
- **`providers.d/`**: Llama Stack provider registration

## Dependencies

Key dependencies managed in `pyproject.toml`:
- `trl==0.18.1` - Transformer Reinforcement Learning library
- `transformers>=4.52.0` - Hugging Face Transformers
- `torch` - PyTorch framework
- `datasets` - Dataset loading and processing

## License

This project follows the same licensing as Llama Stack.

---

**Server Endpoint**: http://localhost:8321  
**API Documentation**: http://localhost:8321/docs  
**Provider Type**: `inline::trl` 