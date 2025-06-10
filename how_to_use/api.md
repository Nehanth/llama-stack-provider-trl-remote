# API Documentation

This document describes the API endpoints for the ML training platform running on `http://127.0.0.1:8321`.

## Base Configuration

```python
base_url = "http://127.0.0.1:8321"

headers_get = {
    "accept": "application/json"
}

headers_post = {
    "Content-Type": "application/json"
}
```

## API Endpoints

### 1. List Providers

**GET** `/v1/providers`

```python
url_providers = f"{base_url}/v1/providers"
response_providers = requests.get(url_providers, headers=headers_get)
```

### 2. List Datasets

**GET** `/v1/datasets`

```python
url_datasets = f"{base_url}/v1/datasets"
response_datasets = requests.get(url_datasets, headers=headers_get)
```

### 3. Upload DPO Dataset

**POST** `/v1/datasets`

```python
url_upload_dataset = f"{base_url}/v1/datasets"

dataset_payload = {
    "dataset_id": "test-dpo-dataset-inline-large",
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
                "chosen": "Here is a simple hello world program in Python:\n\n```python\nprint(\"Hello, World!\")\n```",
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
}

response_dataset = requests.post(url_upload_dataset, headers=headers_post, json=dataset_payload)
print("Dataset Upload Status:", response_dataset.status_code)
print("Dataset Upload Response:", response_dataset.json())
```

### 4. Get All Post-Training Jobs

**GET** `/v1/post-training/jobs`

```python
url_jobs = f"{base_url}/v1/post-training/jobs"
response_jobs = requests.get(url_jobs, headers=headers_get)
print("Jobs Status:", response_jobs.status_code)
print("Jobs Response:", response_jobs.json())
```

### 5. Get Specific Job Status

**GET** `/v1/post-training/job/status?job_uuid={job_uuid}`

```python
job_uuid = "dpo-training-granite-3.3-2b"
url_job_status = f"{base_url}/v1/post-training/job/status?job_uuid={job_uuid}"
response_job_status = requests.get(url_job_status, headers=headers_get)
print("Job Status:", response_job_status.status_code)
print("Job Status Response:", response_job_status.json())
```

### 6. Trigger New Training Job

**POST** `/v1/post-training/preference-optimize`

```python
url_train_model = f"{base_url}/v1/post-training/preference-optimize"

train_model_data = {
    "job_uuid": "dpo-training-granite-3.3-2b",
    "model": "ibm-granite/granite-3.3-2b-base",
    "finetuned_model": "granite-3.3-2b-dpo",
    "checkpoint_dir": "./checkpoints",
    "algorithm_config": {
        "type": "dpo",
        "reward_scale": 1.0,
        "reward_clip": 5.0,
        "epsilon": 0.1,
        "gamma": 0.99
    },
    "training_config": {
        "n_epochs": 3,
        "max_steps_per_epoch": 50,
        "learning_rate": 1e-4,
        "warmup_steps": 0,
        "lr_scheduler_type": "constant",
        "data_config": {
            "dataset_id": "test-dpo-dataset-inline-large",
            "batch_size": 2,
            "shuffle": True,
            "data_format": "instruct",
            "train_split_percentage": 0.8
        }
    },
    "hyperparam_search_config": {},
    "logger_config": {}
}

response_train_model = requests.post(url_train_model, headers=headers_post, json=train_model_data)
print("Train Model Status:", response_train_model.status_code)
print("Train Model Response:", response_train_model.json())
```

### 7. Get Job Artifacts

**GET** `/v1/post-training/job/artifacts?job_uuid={job_uuid}`

```python
url_job_artifacts = f"{base_url}/v1/post-training/job/artifacts?job_uuid={job_uuid}"
response_job_artifacts = requests.get(url_job_artifacts, headers=headers_get)
print("Job Artifacts Status:", response_job_artifacts.status_code)
print("Job Artifacts Response:", response_job_artifacts.json())
```

## Usage Flow

1. **Upload Dataset** - First upload a DPO (Direct Preference Optimization) dataset
2. **Trigger Training** - Start a new training job using the uploaded dataset
3. **Monitor Progress** - Check job status and retrieve artifacts
4. **List Resources** - Query available providers, datasets, and jobs

## Configuration Parameters

### Algorithm Config (DPO)
- `reward_scale`: 1.0
- `reward_clip`: 5.0  
- `epsilon`: 0.1
- `gamma`: 0.99

### Training Config
- `n_epochs`: 3
- `max_steps_per_epoch`: 50
- `learning_rate`: 1e-4
- `batch_size`: 2
- `train_split_percentage`: 0.8 