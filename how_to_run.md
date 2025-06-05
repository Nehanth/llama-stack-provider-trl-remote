# How to Run the TRL Provider for Llama Stack

This guide walks you through the complete process of setting up and running DPO (Direct Preference Optimization) training using the TRL provider for Llama Stack.

## Prerequisites

Before starting, ensure you have:

- **Python 3.10+** installed
- **Llama Stack** framework installed
- **Git** for cloning repositories
- **CUDA** (optional, for GPU training)
- At least **8GB RAM** (16GB+ recommended for larger models)

## Step 1: Environment Setup

### 1.1 Create Llama Stack Environment

First, create a virtual environment specifically for this TRL provider:

```bash
# Build the Llama Stack environment with experimental post-training template
llama stack build --template experimental-post-training --image-type venv --image-name trl-post-training

# Activate the environment
source trl-post-training/bin/activate
```

### 1.2 Install Dependencies

Clean installation to avoid version conflicts:

```bash
# Remove any conflicting packages
pip uninstall torchao -y
rm -rf ./trl-post-training/lib/python3.10/site-packages/torchao*

# Install specific compatible versions
pip install trl==0.18.1 transformers==4.52.4

# Install the TRL provider package
uv pip install -e . --python ./trl-post-training/bin/python --force-reinstall --no-cache
```

### 1.3 Verify Installation

Check that the package is correctly installed:

```bash
python -c "import llama_stack_provider_trl; print('TRL provider installed successfully')"
```

## Step 2: Start the Server

### 2.1 Launch the Llama Stack Server

Start the server with the TRL provider configuration:

```bash
llama stack run --image-type venv --image-name trl-post-training run.yaml
```

### 2.2 Verify Server is Running

You should see log messages indicating successful startup:

```
INFO Loaded inline provider spec for inline::trl
Uvicorn running on http://['::', '0.0.0.0']:8321
```

### 2.3 Test Server Connection

In a new terminal window, verify the server is accessible:

```bash
# Check if server is responding
curl http://localhost:8321/health

# Verify TRL provider is registered
curl -s http://localhost:8321/v1/providers | jq '.data[] | select(.api=="post_training")'
```

Expected response:
```json
{
  "api": "post_training",
  "provider_id": "trl",
  "provider_type": "inline::trl"
}
```

## Step 3: Running DPO Training

### 3.1 Prepare Training Data

The provider expects data in preference format. Use the included sample data or create your own:

**Using included test data:**
```bash
# The test_dpo_data.json file is ready to use
ls test_dpo_data.json
```

**Create custom data:**
```json
{
  "data": [
    {
      "prompt": "What is machine learning?",
      "chosen": "Machine learning is a branch of artificial intelligence that enables computers to learn from data...",
      "rejected": "Machine learning is just computers doing math stuff."
    }
  ]
}
```

### 3.2 Register Dataset with Llama Stack

Before training, you must register your dataset with Llama Stack. You have two options:

**Option 1: File-Based Dataset (if you have a JSON file):**
```bash
curl -X POST http://localhost:8321/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "test-dpo-dataset",
    "purpose": "post-training/messages",
    "dataset_type": "preference",
    "source": {
      "type": "uri",
      "uri": "file://test_dpo_data.json",
      "format": "json"
    },
    "metadata": {
      "provider_id": "localfs",
      "description": "DPO preference training dataset"
    }
  }'
```

**Option 2: Inline Dataset (recommended for testing):**
```bash
curl -X POST http://localhost:8321/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "test-dpo-dataset-inline",
    "purpose": "post-training/messages",
    "dataset_type": "preference",
    "source": {
      "type": "rows",
      "rows": [
        {
          "prompt": "What is machine learning?",
          "chosen": "Machine learning is a branch of artificial intelligence that enables computers to learn from data and improve their performance on specific tasks without being explicitly programmed. It uses algorithms to find patterns in data and make predictions or decisions.",
          "rejected": "Machine learning is just computers doing math stuff with data."
        },
        {
          "prompt": "Write a hello world program",
          "chosen": "Here is a simple hello world program in Python:\n\n```python\nprint(\"Hello, World!\")\n```\n\nThis program uses the print() function to display the text to the console.",
          "rejected": "print hello world"
        },
        {
          "prompt": "Explain the concept of fine-tuning",
          "chosen": "Fine-tuning is the process of taking a pre-trained model and further training it on a specific dataset to adapt it for a particular task or domain while leveraging its existing knowledge. This approach is more efficient than training from scratch.",
          "rejected": "Fine-tuning means making a model better by training it more."
        }
      ]
    },
    "metadata": {
      "provider_id": "localfs",
      "description": "Inline DPO preference training dataset"
    }
  }'
```

**Verify registration:**
```bash
# Check registered datasets
curl -s http://localhost:8321/v1/datasets | jq '.'
```

You should see your dataset listed with ID `test-dpo-dataset`.

### 3.3 Start DPO Training Job

Launch a DPO training job using the REST API:

**Basic Training (no model saving):**
```bash
curl -X POST http://localhost:8321/v1/post-training/preference-optimize \
  -H "Content-Type: application/json" \
  -d '{
    "job_uuid": "dpo-training-001",
    "model": "distilgpt2",
    "finetuned_model": "my-dpo-model",
    "algorithm_config": {
      "type": "dpo",
      "reward_scale": 1.0,
      "reward_clip": 5.0,
      "epsilon": 0.1,
      "gamma": 0.99
    },
    "training_config": {
      "n_epochs": 1,
      "max_steps_per_epoch": 10,
      "learning_rate": 1e-6,
      "data_config": {
        "dataset_id": "test-dpo-dataset-inline",
        "batch_size": 2,
        "shuffle": true,
        "data_format": "instruct",
        "train_split_percentage": 0.8
      }
    },
    "hyperparam_search_config": {},
    "logger_config": {}
  }'
```

**Training with Model Saving:**
```bash
curl -X POST http://localhost:8321/v1/post-training/preference-optimize \
  -H "Content-Type: application/json" \
  -d '{
    "job_uuid": "dpo-training-with-checkpoints",
    "model": "distilgpt2",
    "finetuned_model": "my-dpo-model",
    "checkpoint_dir": "./checkpoints",
    "algorithm_config": {
      "type": "dpo",
      "reward_scale": 1.0,
      "reward_clip": 5.0,
      "epsilon": 0.1,
      "gamma": 0.99
    },
    "training_config": {
      "n_epochs": 1,
      "max_steps_per_epoch": 10,
      "learning_rate": 1e-6,
      "data_config": {
        "dataset_id": "test-dpo-dataset-inline",
        "batch_size": 2,
        "shuffle": true,
        "data_format": "instruct",
        "train_split_percentage": 0.8
      }
    },
    "hyperparam_search_config": {},
    "logger_config": {}
  }'
```

**Important Notes:**
- Use the registered dataset ID from step 3.2 (either `test-dpo-dataset` for file-based or `test-dpo-dataset-inline` for inline data)
- Add `"checkpoint_dir": "./checkpoints"` to save the trained model to your current directory
- The trained model will be saved in `./checkpoints/dpo_model/` and include all necessary files (model weights, tokenizer, config)

**Expected Response:**
```json
{
  "job_uuid": "dpo-training-with-checkpoints"
}
```

### 3.4 Monitor Training Progress

#### Check Job Status
```bash
curl "http://localhost:8321/v1/post-training/job/status?job_uuid=dpo-training-001"
```

Response shows current status:
```json
{
  "job_uuid": "dpo-training-001",
  "status": "in_progress",
  "scheduled_at": "2024-01-10T10:00:00Z",
  "started_at": "2024-01-10T10:00:30Z",
  "checkpoints": [],
  "resources_allocated": {
    "device": "cpu",
    "memory_allocated": "2.5GB"
  }
}
```

#### List All Jobs
```bash
curl http://localhost:8321/v1/post-training/jobs
```

#### View Server Logs
Monitor the training logs in the server terminal:
- Job scheduling messages
- Training progress updates
- Checkpoint saves
- Completion notifications

### 3.5 Retrieve Training Results

Once training completes (status: "completed"), get the artifacts:

```bash
curl "http://localhost:8321/v1/post-training/job/artifacts?job_uuid=dpo-training-001"
```

Response includes checkpoints:
```json
{
  "job_uuid": "dpo-training-001",
  "checkpoints": [
    {
      "identifier": "checkpoint-final",
      "path": "/tmp/dpo-training-001/final",
      "metadata": {
        "epoch": 1,
        "step": 10,
        "loss": 0.45
      }
    }
  ]
}
```

## Step 4: Configuration Options

### 4.1 Provider Configuration

Modify `run.yaml` to adjust provider settings:

```yaml
providers:
  post_training:
  - provider_id: trl
    provider_type: inline::trl
    config:
      device: "cuda"              # Use "cuda" for GPU training
      dpo_beta: 0.1               # DPO beta parameter (0.1-0.5)
      use_reference_model: true   # Enable reference model
      max_seq_length: 2048        # Maximum sequence length
      gradient_checkpointing: true # Enable for memory efficiency
      logging_steps: 5            # Log every N steps
      warmup_ratio: 0.1           # Learning rate warmup
      weight_decay: 0.01          # Regularization
```

### 4.2 Training Configuration

Adjust training parameters in your API call:

```json
{
  "training_config": {
    "n_epochs": 3,                    // Number of training epochs
    "max_steps_per_epoch": 100,       // Steps per epoch
    "batch_size": 4,                  // Batch size (increase with more GPU memory)
    "learning_rate": 1e-6,            // Learning rate (lower for stability)
    "data_config": {
      "dataset_id": "test-dpo-dataset",  // Registered dataset ID from /v1/datasets
      "train_split_percentage": 0.8   // 80% train, 20% validation
    }
  }
}
```

### 4.3 DPO Algorithm Configuration

Fine-tune DPO behavior:

```json
{
  "algorithm_config": {
    "type": "dpo",
    "reward_scale": 1.0,      // Scale factor for rewards
    "reward_clip": 5.0        // Clip rewards to prevent instability
  }
}
```

## Step 5: Production Considerations

### 5.1 GPU Training

For faster training with CUDA:

1. **Update configuration:**
   ```yaml
   config:
     device: "cuda"
     gradient_checkpointing: true
   ```

2. **Increase batch size:**
   ```json
   "batch_size": 8
   ```

3. **Monitor GPU memory:**
   ```bash
   nvidia-smi
   ```

### 5.2 Large Dataset Training

For larger datasets:

1. **Increase dataloader workers:**
   ```yaml
   config:
     dataloader_num_workers: 4
   ```

2. **Use gradient checkpointing:**
   ```yaml
   config:
     gradient_checkpointing: true
   ```

3. **Adjust sequence length:**
   ```yaml
   config:
     max_seq_length: 1024  # Reduce if memory constrained
   ```

### 5.3 Monitoring and Logging

1. **Server logs:** Monitor the Llama Stack server terminal
2. **Job status:** Poll the status endpoint regularly
3. **Resource usage:** Check the resources_allocated field
4. **Checkpoints:** Retrieve artifacts for model evaluation

## Step 6: Troubleshooting

### 6.1 Server Won't Start

**Import errors:**
```bash
# Reinstall dependencies
source trl-post-training/bin/activate
pip install trl==0.18.1 transformers==4.52.4
uv pip install -e . --force-reinstall --no-cache
```

**Port conflicts:**
```bash
# Check what's using port 8321
lsof -ti:8321

# Kill conflicting processes
lsof -ti:8321 | xargs kill -9
```

### 6.2 Training Fails

**Memory issues:**
- Reduce batch size to 1 or 2
- Enable gradient checkpointing
- Use CPU instead of GPU

**Data format errors:**
- Verify JSON format matches the expected structure
- Ensure all entries have "prompt", "chosen", and "rejected" fields
- Check that dataset is properly registered with Llama Stack

**Dataset registration errors:**
- Verify dataset file exists and is accessible
- Check that dataset_id matches the registered ID
- Ensure JSON file follows the correct structure

**Model loading errors:**
- Verify model name is valid (e.g., "distilgpt2", "gpt2")
- Check internet connection for downloading models

### 6.3 Job Management

**Cancel stuck jobs:**
```bash
curl -X POST "http://localhost:8321/v1/post-training/job/cancel" \
  -H "Content-Type: application/json" \
  -d '{"job_uuid": "your-job-uuid"}'
```

**Clear all jobs:**
- Restart the server to clear job queue

## Step 7: API Reference

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/datasets` | Register a dataset |
| GET | `/v1/datasets` | List registered datasets |
| POST | `/v1/post-training/preference-optimize` | Start DPO training |
| GET | `/v1/post-training/job/status?job_uuid=<id>` | Get job status |
| GET | `/v1/post-training/job/artifacts?job_uuid=<id>` | Get artifacts |
| POST | `/v1/post-training/job/cancel` | Cancel job |
| GET | `/v1/post-training/jobs` | List all jobs |
| GET | `/v1/providers` | List providers |
| GET | `/health` | Server health |

### API Documentation

Access interactive API docs at: http://localhost:8321/docs

## Next Steps

1. **Experiment with parameters:** Try different learning rates, batch sizes, and DPO settings
2. **Use your own data:** Create custom preference datasets for your use case
3. **Evaluate results:** Load trained checkpoints and test on validation data
4. **Scale up:** Move to GPU training for larger models and datasets
5. **Integration:** Integrate trained models into your applications

For more advanced usage and troubleshooting, refer to the main README.md file.

## Sample Dataset

### Data Format Requirements

The TRL provider expects training data in JSON format with the following structure:

```json
{
  "data": [
    {
      "prompt": "Your question or instruction here",
      "chosen": "The preferred/better response",
      "rejected": "The less preferred/worse response"
    }
  ]
}
```

### Complete Sample Dataset

Here's a comprehensive sample dataset (`sample_training_data.json`) covering various types of prompts:

```json
{
  "data": [
    {
      "prompt": "Explain quantum computing in simple terms.",
      "chosen": "Quantum computing is a revolutionary technology that uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or 'qubits' that can exist in multiple states simultaneously. This allows them to perform certain calculations exponentially faster than classical computers, particularly for problems involving cryptography, optimization, and simulation of quantum systems.",
      "rejected": "Quantum computing is just really fast computers that use quantum stuff to be better than normal computers."
    },
    {
      "prompt": "Write a Python function to check if a number is prime.",
      "chosen": "Here's an efficient Python function to check if a number is prime:\n\n```python\ndef is_prime(n):\n    \"\"\"Check if a number is prime.\"\"\"\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    \n    # Check odd divisors up to sqrt(n)\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n```\n\nThis function handles edge cases and uses an optimized algorithm that only checks odd divisors up to the square root of n.",
      "rejected": "def is_prime(n):\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True"
    },
    {
      "prompt": "How do you handle stress at work?",
      "chosen": "Here are effective strategies for managing work stress:\n\n1. **Time Management**: Prioritize tasks using methods like the Eisenhower Matrix\n2. **Set Boundaries**: Learn to say no to non-essential requests\n3. **Take Breaks**: Regular short breaks improve focus and reduce burnout\n4. **Exercise**: Physical activity releases endorphins and reduces stress hormones\n5. **Mindfulness**: Practice deep breathing or meditation during stressful moments\n6. **Communicate**: Talk to supervisors about workload concerns\n7. **Work-Life Balance**: Maintain clear separation between work and personal time\n8. **Seek Support**: Connect with colleagues or professionals when needed\n\nRemember, some stress is normal, but chronic stress requires proactive management.",
      "rejected": "Just deal with it. Everyone has stress at work, you just need to push through and work harder."
    },
    {
      "prompt": "What's the difference between machine learning and artificial intelligence?",
      "chosen": "Artificial Intelligence (AI) and Machine Learning (ML) are related but distinct concepts:\n\n**Artificial Intelligence:**\n- Broader field focused on creating systems that can perform tasks requiring human intelligence\n- Includes rule-based systems, expert systems, robotics, and machine learning\n- Goal: Simulate human cognitive functions like reasoning, learning, and problem-solving\n\n**Machine Learning:**\n- Subset of AI that focuses on algorithms that improve through experience\n- Systems learn patterns from data without being explicitly programmed\n- Includes supervised, unsupervised, and reinforcement learning\n\n**Key Relationship:**\nML is a method to achieve AI. Not all AI uses ML (like rule-based chatbots), but ML is currently the most successful approach to creating intelligent systems.\n\n**Example:** A chess-playing AI could use hand-coded rules (traditional AI) or learn strategies from millions of games (ML approach).",
      "rejected": "AI is the big thing and machine learning is part of it. They're basically the same thing - computers that think like humans."
    },
    {
      "prompt": "Explain the concept of blockchain technology.",
      "chosen": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records (blocks) linked and secured using cryptography:\n\n**Key Components:**\n1. **Blocks**: Containers holding transaction data, timestamps, and cryptographic hashes\n2. **Chain**: Blocks linked chronologically through hash references\n3. **Decentralization**: No single authority controls the network\n4. **Consensus**: Network participants agree on valid transactions\n\n**How it Works:**\n- Transactions are broadcast to the network\n- Network nodes validate transactions using consensus mechanisms\n- Valid transactions are bundled into blocks\n- Blocks are added to the chain and distributed across all nodes\n\n**Benefits:**\n- Transparency and immutability\n- Reduced need for intermediaries\n- Enhanced security through cryptography\n- Global accessibility\n\n**Applications:** Cryptocurrencies, supply chain tracking, digital identity, smart contracts, and voting systems.",
      "rejected": "Blockchain is the technology behind Bitcoin. It's like a digital ledger that can't be hacked because it's on many computers."
    },
    {
      "prompt": "How do you debug a program that's running slowly?",
      "chosen": "Here's a systematic approach to debugging performance issues:\n\n**1. Profiling and Measurement:**\n- Use profiling tools (cProfile for Python, Chrome DevTools for JavaScript)\n- Measure execution time for different code sections\n- Monitor memory usage and CPU utilization\n\n**2. Identify Bottlenecks:**\n- Look for functions consuming the most time\n- Check for memory leaks or excessive allocations\n- Identify I/O operations (database, file, network)\n\n**3. Common Performance Issues:**\n- Inefficient algorithms (O(n²) instead of O(n log n))\n- Unnecessary loops or redundant calculations\n- Database queries without proper indexing\n- Synchronous operations that could be asynchronous\n\n**4. Optimization Strategies:**\n- Optimize algorithms and data structures\n- Implement caching for frequently accessed data\n- Use batch operations for database queries\n- Parallelize independent operations\n- Remove or optimize expensive operations\n\n**5. Verify Improvements:**\n- Re-profile after changes\n- Use benchmarks to measure improvements\n- Test with realistic data volumes",
      "rejected": "Just add more print statements everywhere to see what's slow, then fix the slow parts."
    },
    {
      "prompt": "What are the best practices for API design?",
      "chosen": "Here are essential best practices for designing robust APIs:\n\n**1. RESTful Principles:**\n- Use standard HTTP methods (GET, POST, PUT, DELETE)\n- Design resource-based URLs (/users/123, not /getUser?id=123)\n- Implement proper status codes (200, 201, 400, 401, 404, 500)\n\n**2. Consistency:**\n- Follow consistent naming conventions (snake_case or camelCase)\n- Use standard response formats across endpoints\n- Maintain consistent error handling patterns\n\n**3. Documentation:**\n- Provide comprehensive API documentation\n- Include examples for all endpoints\n- Document request/response schemas\n- Use tools like OpenAPI/Swagger\n\n**4. Security:**\n- Implement proper authentication (JWT, OAuth)\n- Use HTTPS for all endpoints\n- Validate and sanitize all inputs\n- Implement rate limiting\n\n**5. Performance:**\n- Support pagination for large datasets\n- Implement caching strategies\n- Use compression (gzip)\n- Optimize database queries\n\n**6. Versioning:**\n- Version your API (/v1/users)\n- Maintain backward compatibility\n- Provide migration guides for breaking changes",
      "rejected": "Just make URLs that work and return JSON. Add authentication if you need it."
    },
    {
      "prompt": "Explain the importance of code reviews.",
      "chosen": "Code reviews are a critical practice in software development that provides multiple benefits:\n\n**Quality Improvement:**\n- Catch bugs and logic errors before production\n- Ensure adherence to coding standards and best practices\n- Identify potential security vulnerabilities\n- Improve code readability and maintainability\n\n**Knowledge Sharing:**\n- Share domain knowledge across team members\n- Expose developers to different coding approaches\n- Help junior developers learn from senior team members\n- Increase bus factor (reduce dependency on single developers)\n\n**Collaboration Benefits:**\n- Foster team communication and discussion\n- Build collective code ownership\n- Ensure multiple eyes on critical changes\n- Create opportunities for mentoring\n\n**Process Improvements:**\n- Maintain consistent code style across the project\n- Ensure documentation is updated with code changes\n- Verify that tests are adequate and passing\n- Check that changes align with architectural decisions\n\n**Best Practices:**\n- Keep reviews focused and reasonably sized\n- Be constructive and specific in feedback\n- Use automated tools to catch basic issues\n- Review promptly to avoid blocking teammates\n- Follow up on requested changes",
      "rejected": "Code reviews are good because other people can check your code and find mistakes. It's required by most companies."
    }
  ]
}
```

### Creating Quality Training Data

**Guidelines for Effective DPO Data:**

1. **Clear Contrast**: Ensure significant quality difference between chosen and rejected responses
2. **Realistic Scenarios**: Use prompts that reflect real user needs
3. **Diverse Topics**: Cover various domains and question types
4. **Consistent Quality**: Chosen responses should consistently demonstrate better reasoning, accuracy, and helpfulness
5. **Appropriate Length**: Balance detail with conciseness in both chosen and rejected responses

**Common Response Quality Differences:**
- **Accuracy**: Factually correct vs. incorrect information
- **Completeness**: Comprehensive vs. superficial explanations
- **Structure**: Well-organized vs. disorganized content
- **Helpfulness**: Practical, actionable advice vs. vague suggestions
- **Professional Tone**: Professional language vs. casual or inappropriate tone

**Data Size Recommendations:**
- **Minimum**: 50-100 examples for basic fine-tuning
- **Recommended**: 500-1000 examples for good performance
- **Optimal**: 1000+ examples for production use

Save your data as a JSON file and reference it in your training configuration using the `dataset_id` parameter. 