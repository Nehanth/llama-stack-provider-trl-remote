# TRL Post-Training Provider for Llama Stack

This is an inline provider for **Direct Preference Optimization (DPO)** training using HuggingFace's **TRL (Transformer Reinforcement Learning)** library. It extends the Llama Stack training framework to support preference-based optimization methods like DPO.

## 🎯 Purpose

While the existing HuggingFace provider focuses on **Supervised Fine-Tuning (SFT)**, this TRL provider specializes in **preference optimization** - training models to prefer certain responses over others based on human feedback or preference data.

## 📁 File Structure

```
llama_stack/providers/inline/post_training/trl/
├── README.md                           # This documentation file
├── __init__.py                         # Provider entry point and registration
├── config.py                          # Configuration classes and settings
├── post_training.py                   # Main provider implementation
└── recipes/
    └── dpo_training_single_device.py  # Core DPO training logic
```

## 📄 File Descriptions

### `__init__.py` - Provider Entry Point
- **Purpose**: Entry point for the TRL provider when loaded by Llama Stack
- **Key Function**: `get_provider_impl()` - Creates and returns the provider instance
- **Dependencies**: Receives `datasetio` and `datasets` APIs from Llama Stack
- **Returns**: Configured `TrlPostTrainingImpl` instance ready for use

### `config.py` - Configuration Management
- **Purpose**: Defines all configuration options for DPO training
- **Key Class**: `TrlPostTrainingConfig` - Contains all hyperparameters and settings
- **DPO-Specific Settings**:
  - `dpo_beta`: Controls how strongly to prefer chosen over rejected responses
  - `use_reference_model`: Whether to use a separate reference model for DPO
  - `dpo_loss_type`: Type of loss function (sigmoid, hinge, ipo)
- **Inherited Settings**: Device, memory, tokenization, and training parameters

### `post_training.py` - Main Provider Implementation
- **Purpose**: Implements the Llama Stack `PostTraining` protocol for TRL
- **Key Class**: `TrlPostTrainingImpl` - Main provider class
- **Key Methods**:
  - `preference_optimize()`: Starts DPO training jobs (main entry point)
  - `supervised_fine_tune()`: Not implemented (raises error - use HuggingFace provider instead)
  - Job management methods: status, cancel, artifacts
- **Architecture**: Uses async job scheduling with separate training processes

### `recipes/dpo_training_single_device.py` - Core Training Logic
- **Purpose**: Contains the actual DPO training implementation using TRL
- **Key Class**: `DPOTrainingSingleDevice` - Handles all aspects of DPO training
- **Key Responsibilities**:
  - Dataset loading and validation (prompt/chosen/rejected format)
  - Model and tokenizer initialization
  - DPO trainer setup with reference model
  - Training execution in isolated process
  - Checkpoint and artifact management

## 🔧 How It Works

### 1. **Provider Registration**
The provider is registered in `llama_stack/providers/registry/post_training.py` as `inline::trl`

### 2. **Job Scheduling**
When `preference_optimize()` is called:
1. Creates an async job handler
2. Schedules job with Llama Stack's `Scheduler`
3. Returns `PostTrainingJob` immediately (non-blocking)

### 3. **Training Process**
The actual training runs in a separate process:
1. **Dataset Processing**: Validates and formats preference data
2. **Model Loading**: Loads base model and optional reference model
3. **DPO Training**: Uses TRL's `DPOTrainer` for preference optimization
4. **Checkpoint Saving**: Saves trained model and tokenizer
5. **Cleanup**: Manages memory and resources

### 4. **Monitoring**
Clients can monitor training progress through:
- `get_training_job_status()`: Check if training is running/completed/failed
- `get_training_job_artifacts()`: Get saved checkpoints
- Real-time logging through job callbacks

## 📊 Dataset Format

DPO training requires preference datasets with three fields per example:

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "The capital of France is Lyon."
}
```

### Required Fields:
- **`prompt`**: The input question or instruction
- **`chosen`**: The preferred/better response
- **`rejected`**: The less preferred/worse response

## 🚀 Usage Example

```python
from llama_stack.providers.inline.post_training.trl import TrlPostTrainingConfig
from llama_stack.providers.inline.post_training.trl.post_training import TrlPostTrainingImpl
from llama_stack.apis.post_training import DPOAlignmentConfig, TrainingConfig

# Configure the TRL provider
trl_config = TrlPostTrainingConfig(
    device="cuda",
    dpo_beta=0.1,           # DPO strength parameter
    use_reference_model=True,
    max_seq_length=1024
)

# Configure DPO training
dpo_config = DPOAlignmentConfig(
    reward_scale=1.0,
    reward_clip=5.0,
    epsilon=0.2,
    gamma=1.0
)

training_config = TrainingConfig(
    n_epochs=3,
    data_config=DataConfig(
        dataset_id="my-preference-dataset",
        batch_size=4,
        data_format=DatasetFormat.instruct
    )
)

# Initialize provider
provider = TrlPostTrainingImpl(trl_config, datasetio_api, datasets_api)

# Start DPO training
job = await provider.preference_optimize(
    job_uuid="my-dpo-job",
    finetuned_model="my-sft-model",
    algorithm_config=dpo_config,
    training_config=training_config,
    hyperparam_search_config={},
    logger_config={}
)

# Monitor progress
status = await provider.get_training_job_status(job.job_uuid)
print(f"Training status: {status.status}")
```

## 🔄 Workflow: SFT → DPO

Typical workflow for training a model with preferences:

1. **Supervised Fine-Tuning**: Use HuggingFace provider to train on instruction data
2. **Preference Optimization**: Use TRL provider to align model with human preferences
3. **Evaluation**: Test the DPO-trained model on your specific tasks

```
Raw Model → [SFT with HuggingFace] → SFT Model → [DPO with TRL] → Aligned Model
```

## 🎛️ Key Configuration Options

### DPO-Specific Parameters:
- **`dpo_beta`** (0.1): Controls preference strength - higher values = stronger preference learning
- **`use_reference_model`** (True): Use separate reference model for DPO loss calculation
- **`dpo_loss_type`** ("sigmoid"): Type of DPO loss function
- **`normalize_rewards`** (True): Whether to normalize reward differences

### Training Parameters:
- **`device`**: Where to run training ("cuda", "cpu", "mps")
- **`max_seq_length`**: Maximum input sequence length
- **`gradient_checkpointing`**: Trade compute for memory
- **`mixed_precision`**: Use fp16/bf16 for faster training

## 🔍 Debugging and Troubleshooting

### Common Issues:

1. **Dataset Format Errors**:
   - Ensure dataset has `prompt`, `chosen`, `rejected` fields
   - Check that all fields are strings (not empty/null)

2. **Memory Issues**:
   - Reduce `batch_size` in training config
   - Enable `gradient_checkpointing`
   - Use `mixed_precision="fp16"`

3. **Model Loading Errors**:
   - Verify model path/ID is correct
   - Check that model supports the tokenizer

4. **Training Failures**:
   - Check logs in job status response
   - Verify dataset is properly registered with Llama Stack
   - Ensure sufficient disk space for checkpoints

### Logging:
All training logs are captured and available through:
```python
status = await provider.get_training_job_status(job_uuid)
# Check status.logs for detailed information
```

## 🚧 Limitations

- **Single Device Only**: Currently supports single-GPU/CPU training
- **No Multi-Node**: No distributed training across multiple machines
- **Memory Requirements**: DPO requires loading both model and reference model
- **Dataset Size**: Large preference datasets may require batch processing

## 🔮 Future Enhancements

- **PPO Support**: Add Proximal Policy Optimization training
- **Multi-GPU Training**: Support for distributed DPO training
- **Custom Loss Functions**: Additional preference learning algorithms
- **Online DPO**: Training with real-time human feedback

## 📚 References

- [TRL Library Documentation](https://huggingface.co/docs/trl/)
- [DPO Paper: "Direct Preference Optimization"](https://arxiv.org/abs/2305.18290)
- [Llama Stack Documentation](https://github.com/meta-llama/llama-stack)

## 🤝 Contributing

This provider follows the same patterns as other Llama Stack providers. To extend it:

1. Add new configuration options in `config.py`
2. Implement new methods in `post_training.py`
3. Add training logic in `recipes/`
4. Update this README with new features 