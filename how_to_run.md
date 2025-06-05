# How to Run TRL Provider for Llama Stack

This guide provides complete A-Z instructions for setting up and running DPO (Direct Preference Optimization) training using the TRL provider for Llama Stack.

## 📋 Prerequisites

- Python 3.10+
- Git
- At least 4GB RAM
- Optional: CUDA-compatible GPU for faster training

## 🚀 Complete Setup (A-Z)

### Step 1: Clone Repository

```bash
# Clone the repository
git clone <your-repo-url>
cd llama-stack-provider-trl
```

### Step 2: Build Environment with Llama Stack

```bash
# Build the TRL environment using Llama Stack
llama stack build --template experimental-post-training --image-type venv --image-name trl-post-training
```

### Step 3: Install Dependencies

```bash
# Activate the environment and install dependencies
source trl-post-training/bin/activate
pip uninstall torchao -y
rm -rf ./trl-post-training/lib/python3.10/site-packages/torchao*
pip install trl==0.18.1 transformers==4.52.4
uv pip install -e . --python ./trl-post-training/bin/python --force-reinstall --no-cache
```

### Step 4: Start the TRL Provider Server

```bash
# Start the server (will run on http://localhost:8321)
llama stack run --image-type venv --image-name trl-post-training simple-trl-run.yaml
```

You should see:
```
INFO: Loading external providers from /path/to/llama-stack-provider-trl/providers.d
INFO: Loaded inline provider spec for inline::trl
INFO: Application startup complete.
INFO: Uvicorn running on http://localhost:8321
```

## ✅ Verify Server is Running

```bash
# Check if server is responding
curl -s http://localhost:8321/v1/providers | jq '.data[] | select(.api=="post_training")'

# Should return:
# {
#   "provider_id": "trl",
#   "provider_type": "inline::trl",
#   "api": "post_training"
# }
```

## 📊 Sample Datasets

### Example 1: Simple Preference Dataset

Create a file `sample_preference_data.json`:

```json
[
  {
    "prompt": "What is the capital of France?",
    "chosen": "The capital of France is Paris. It's a beautiful city known for its art, culture, and the Eiffel Tower.",
    "rejected": "France's capital is Paris, I think. It's got some tower or something."
  },
  {
    "prompt": "Explain quantum computing in simple terms.",
    "chosen": "Quantum computing uses quantum mechanical properties like superposition and entanglement to process information in ways classical computers cannot. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in multiple states simultaneously, potentially solving certain problems exponentially faster.",
    "rejected": "Quantum computing is like super fast computers that use quantum stuff to be really fast and do things normal computers can't do well."
  },
  {
    "prompt": "How do you make a good first impression?",
    "chosen": "To make a good first impression: 1) Be punctual and well-groomed, 2) Maintain eye contact and offer a firm handshake, 3) Listen actively and ask thoughtful questions, 4) Show genuine interest in others, 5) Be positive and authentic in your interactions.",
    "rejected": "Just be yourself and try to look nice. Smile and stuff. Don't be weird."
  },
  {
    "prompt": "What's the difference between AI and machine learning?",
    "chosen": "AI (Artificial Intelligence) is the broader concept of machines being able to carry out tasks in a way that we would consider 'smart.' Machine Learning is a subset of AI that focuses on algorithms that can learn and improve from data without being explicitly programmed for every scenario.",
    "rejected": "AI is like robots and stuff. Machine learning is when computers learn things. They're basically the same thing I guess."
  }
]
```

### Example 2: Instruction-Following Dataset

Create a file `instruction_preference_data.json`:

```json
[
  {
    "prompt": "Write a professional email declining a job offer.",
    "chosen": "Subject: Thank you for the offer - [Your Name]\n\nDear [Hiring Manager's Name],\n\nThank you for offering me the [Position Title] role at [Company Name]. I appreciate the time you invested in the interview process and your confidence in my abilities.\n\nAfter careful consideration, I have decided to decline this offer as I have accepted a position that better aligns with my career goals.\n\nI hope we can stay in touch for future opportunities.\n\nBest regards,\n[Your Name]",
    "rejected": "hey thanks for the job but i'm gonna pass. got something better. maybe next time.\n\nthanks"
  },
  {
    "prompt": "Explain how to change a tire step by step.",
    "chosen": "Here's how to change a tire safely:\n\n1. **Safety first**: Pull over to a safe, flat location away from traffic\n2. **Secure the vehicle**: Turn on hazard lights and apply parking brake\n3. **Gather tools**: Locate spare tire, jack, and lug wrench\n4. **Loosen lug nuts**: Remove hubcap if needed, then loosen (don't remove) lug nuts\n5. **Jack up the car**: Place jack under vehicle's jack point and raise until flat tire is off ground\n6. **Remove flat tire**: Fully remove lug nuts and pull tire toward you\n7. **Mount spare tire**: Align spare with bolts and push onto wheel hub\n8. **Replace lug nuts**: Hand-tighten, then use wrench to secure firmly\n9. **Lower vehicle**: Lower jack until spare tire touches ground but still supports some weight\n10. **Final tightening**: Fully tighten lug nuts in a star pattern\n11. **Finish**: Lower vehicle completely, remove jack, and stow equipment",
    "rejected": "take off old tire, put on new tire. use the jack thing to lift the car up. tighten the nuts. done."
  }
]
```

## 🔧 API Usage Examples

**Note**: Before running the curl commands below, make sure you've created the JSON dataset files (`sample_preference_data.json`, `instruction_preference_data.json`, `quick_test_data.json`) as shown in the Sample Datasets section above.

**Requirements**: Install `jq` for JSON processing:
```bash
# Ubuntu/Debian
sudo apt-get install jq

# macOS
brew install jq

# Or use Method 2 (temporary file approach) if jq is not available
```

### Step 5: Register a Dataset

```bash
# Method 1: Register dataset using jq to properly merge JSON
curl -X POST "http://localhost:8321/v1/datasets" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --argjson rows "$(cat sample_preference_data.json)" '{
    dataset_id: "preference_sample",
    purpose: "post-training/messages",
    source: {
      type: "rows",
      rows: $rows
    }
  }')"

# Method 2: Alternative using temporary file approach
cat > temp_dataset_request.json << EOF
{
  "dataset_id": "preference_sample",
  "purpose": "post-training/messages",
  "source": {
    "type": "rows",
    "rows": $(cat sample_preference_data.json)
  }
}
EOF

curl -X POST "http://localhost:8321/v1/datasets" \
  -H "Content-Type: application/json" \
  -d @temp_dataset_request.json

# Clean up
rm temp_dataset_request.json

# Register the instruction-following dataset
curl -X POST "http://localhost:8321/v1/datasets" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --argjson rows "$(cat instruction_preference_data.json)" '{
    dataset_id: "instruction_sample",
    purpose: "post-training/messages",
    source: {
      type: "rows",
      rows: $rows
    }
  }')"
```

### Step 6: Start DPO Training

```bash
# Start a DPO training job using the registered dataset
curl -X POST "http://localhost:8321/v1/post-training/preference-optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "job_uuid": "my-dpo-training",
    "model": "distilgpt2",
    "finetuned_model": "my-dpo-model",
    "training_config": {
      "data_config": {
        "dataset_id": "preference_sample",
        "data_format": "instruct",
        "batch_size": 1,
        "shuffle": true
      },
      "n_epochs": 1,
      "max_steps_per_epoch": 2,
      "gradient_accumulation_steps": 1
    },
    "algorithm_config": {
      "type": "dpo",
      "reward_scale": 1.0,
      "reward_clip": 5.0,
      "epsilon": 0.1,
      "gamma": 0.99
    },
    "optimizer_config": {
      "lr": 1e-6,
      "lr_scheduler": "linear_with_warmup",
      "warmup_steps": 0
    },
    "hyperparam_search_config": {
      "enabled": false
    },
    "logger_config": {
      "log_level": "INFO"
    }
  }'

# Or use the instruction-following dataset:
curl -X POST "http://localhost:8321/v1/post-training/preference-optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "job_uuid": "instruction-training",
    "model": "distilgpt2",
    "finetuned_model": "my-instruction-model",
    "training_config": {
      "data_config": {
        "dataset_id": "instruction_sample",
        "data_format": "instruct",
        "batch_size": 1,
        "shuffle": true
      },
      "n_epochs": 1,
      "max_steps_per_epoch": 2,
      "gradient_accumulation_steps": 1
    },
    "algorithm_config": {
      "type": "dpo",
      "reward_scale": 1.0,
      "reward_clip": 5.0,
      "epsilon": 0.1,
      "gamma": 0.99
    },
    "optimizer_config": {
      "lr": 1e-6,
      "lr_scheduler": "linear_with_warmup",
      "warmup_steps": 0
    },
    "hyperparam_search_config": {
      "enabled": false
    },
    "logger_config": {
      "log_level": "INFO"
    }
  }'
```

### Step 7: Monitor Training Progress

```bash
# Check job status
curl "http://localhost:8321/v1/post-training/job/status?job_uuid=my-dpo-training"

# Example response:
# {
#   "job_uuid": "my-dpo-training",
#   "status": "completed",
#   "resources_allocated": {...},
#   "checkpoints": [
#     {
#       "identifier": "distilgpt2-dpo-1",
#       "path": "/path/to/my-dpo-training/dpo_model",
#       "created_at": "2025-01-05T01:17:08.885000+00:00"
#     }
#   ]
# }
```

### Step 8: List All Training Jobs

```bash
# Get all post-training jobs
curl "http://localhost:8321/v1/post-training/jobs"
```

## 🎯 Quick Test Example

First, create a minimal test dataset file `quick_test_data.json`:

```json
[
  {
    "prompt": "Hello",
    "chosen": "Hello! How can I help you today?",
    "rejected": "hi"
  }
]
```

Here's a complete quick test you can run:

```bash
# 1. Register a minimal dataset using the JSON file
curl -X POST "http://localhost:8321/v1/datasets" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --argjson rows "$(cat quick_test_data.json)" '{
    dataset_id: "quick_test",
    purpose: "post-training/messages",
    source: {
      type: "rows",
      rows: $rows
    }
  }')"

# 2. Start training (1 step only for quick test)
curl -X POST "http://localhost:8321/v1/post-training/preference-optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "job_uuid": "quick-test",
    "model": "distilgpt2",
    "finetuned_model": "quick-test-model",
    "training_config": {
      "data_config": {
        "dataset_id": "quick_test",
        "data_format": "instruct",
        "batch_size": 1,
        "shuffle": true
      },
      "n_epochs": 1,
      "max_steps_per_epoch": 1,
      "gradient_accumulation_steps": 1
    },
    "algorithm_config": {
      "type": "dpo",
      "reward_scale": 1.0,
      "reward_clip": 5.0,
      "epsilon": 0.1,
      "gamma": 0.99
    },
    "optimizer_config": {
      "lr": 1e-6,
      "lr_scheduler": "linear_with_warmup",
      "warmup_steps": 0
    },
    "hyperparam_search_config": {
      "enabled": false
    },
    "logger_config": {
      "log_level": "INFO"
    }
  }'

# 3. Check status (wait ~10 seconds for training to complete)
curl "http://localhost:8321/v1/post-training/job/status?job_uuid=quick-test"
```

## 📁 File Structure After Training

After successful training, you'll find:

```
quick-test/
├── dpo_model/
│   ├── config.json           # Model configuration
│   ├── model.safetensors     # Trained model weights (~327MB for distilgpt2)
│   ├── tokenizer.json        # Tokenizer data
│   ├── tokenizer_config.json # Tokenizer configuration
│   └── special_tokens_map.json
└── checkpoint-1/             # Intermediate checkpoint
    ├── config.json
    ├── model.safetensors
    └── ...
```

## 🔍 Server Configuration

The server configuration is in `simple-trl-run.yaml`:

```yaml
providers:
  post_training:
  - config:
      device: cpu                    # Change to "cuda" for GPU
      dpo_beta: 0.1                 # DPO strength parameter
      max_seq_length: 2048          # Maximum sequence length
      use_reference_model: true     # Use reference model for DPO
      gradient_checkpointing: false # Memory optimization
      logging_steps: 10             # Logging frequency
      warmup_ratio: 0.1            # Learning rate warmup
      weight_decay: 0.01           # L2 regularization
    provider_id: trl
    provider_type: inline::trl
```

## 🐛 Troubleshooting

### Common Issues:

1. **Server won't start**: Check if port 8321 is available
   ```bash
   lsof -i :8321  # Check if port is in use
   ```

2. **Import errors**: Re-run Step 3 (dependency installation)
   ```bash
   source trl-post-training/bin/activate
   pip uninstall torchao -y
   rm -rf ./trl-post-training/lib/python3.10/site-packages/torchao*
   pip install trl==0.18.1 transformers==4.52.4
   uv pip install -e . --python ./trl-post-training/bin/python --force-reinstall --no-cache
   ```

3. **CUDA out of memory**: Reduce batch size or use CPU
   ```yaml
   device: cpu  # In simple-trl-run.yaml
   ```

4. **Training hangs**: Use smaller models or fewer steps
   ```json
   "max_steps_per_epoch": 1
   ```

### Debug Logs:

Check server logs for detailed information:
```bash
# The server outputs detailed logs including:
# - Dataset loading progress
# - Training metrics
# - Memory usage
# - Checkpoint saving
```

## 🎉 Success Indicators

You'll know it's working when you see:

1. **Server startup**: "Uvicorn running on http://localhost:8321"
2. **Provider loaded**: "Loaded inline provider spec for inline::trl"
3. **Training progress**: Progress bars and training metrics
4. **Completion**: "DPO training completed successfully"
5. **Artifacts**: Model files saved in output directory
6. **API response**: Job status shows "completed"

## 🚀 Next Steps

Once you have the basic setup working:

1. **Scale up**: Use larger models like `microsoft/DialoGPT-medium`
2. **More data**: Create larger preference datasets
3. **GPU training**: Configure CUDA for faster training
4. **Production**: Deploy on cloud instances with GPUs
5. **Integration**: Connect to larger Llama Stack deployments

## 📖 Additional Resources

- [TRL Documentation](https://huggingface.co/docs/trl/)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [Llama Stack Documentation](https://llama-stack.readthedocs.io/)

---

🎯 **You now have a fully functional TRL provider for Llama Stack!** The setup supports both CPU and GPU training, handles real datasets, and provides production-ready DPO training capabilities.