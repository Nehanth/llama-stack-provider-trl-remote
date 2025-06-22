"""
TRL Remote Training Service
===========================

FastAPI service that wraps the TRL inline provider and exposes it via HTTP.
This allows running TRL training jobs remotely while reusing all the existing 
training logic, recipes, and configurations.
"""

import asyncio
import json
import tempfile
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from config import TrlPostTrainingConfig
from dpo_training_single_device import DPOTrainingSingleDevice
from llama_stack.apis.post_training import (
    DPOAlignmentConfig,
    TrainingConfig,
)


# === REMOTE DATASET HANDLER ===

class RemoteDatasetHandler:
    """Simple handler for dataset data passed from the client"""
    
    def __init__(self, dataset_data: list):
        self.dataset_data = dataset_data
    
    async def get_rows(self, dataset_id: str, limit: int = -1):
        """Return dataset rows directly"""
        rows = self.dataset_data
        if limit > 0:
            rows = rows[:limit]
        return rows


# === FASTAPI SERVICE ===

app = FastAPI(title="TRL Remote Training Service")

# Global provider instance (will be initialized on startup)
trl_provider = None


@app.on_event("startup")
async def startup_event():
    """Initialize the TRL provider on service startup"""
    print("Starting TRL Remote Training Service...")
    print("Service initialized successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "trl-training-service"}


@app.get("/training-jobs")
async def get_training_jobs():
    """Get list of training jobs. Returns empty list since jobs are executed synchronously."""
    return {"data": []}


@app.get("/training-job-status")
async def get_training_job_status(job_uuid: str):
    """Get training job status. Jobs are executed synchronously so this returns None."""
    return None


@app.post("/cancel-training-job")
async def cancel_training_job(request: dict):
    """Cancel training job. Jobs run synchronously and cannot be cancelled."""
    return {"status": "not_implemented", "message": "Synchronous jobs cannot be cancelled"}


@app.get("/training-job-artifacts")
async def get_training_job_artifacts(job_uuid: str):
    """Get training job artifacts. Artifacts are saved to checkpoint directory."""
    return None


@app.post("/preference-optimize")
async def preference_optimize_endpoint(request: dict):
    """Handle DPO training request from client"""
    try:
        # Extract request data
        job_uuid = request.get("job_uuid")
        model = request.get("model")
        finetuned_model = request.get("finetuned_model")
        algorithm_config_dict = request.get("algorithm_config", {})
        training_config_dict = request.get("training_config", {})
        hyperparam_search_config = request.get("hyperparam_search_config", {})
        logger_config_dict = request.get("logger_config", {})
        checkpoint_dir = request.get("checkpoint_dir")
        provider_config_dict = request.get("provider_config", {})
        dataset_data = request.get("dataset_data")
        
        # Create provider config
        config = TrlPostTrainingConfig(**provider_config_dict) if provider_config_dict else TrlPostTrainingConfig()
        
        # Create dataset handler if data provided
        if dataset_data:
            handler = RemoteDatasetHandler(dataset_data)
        else:
            handler = None
        
        # Create mock deps for the provider
        from llama_stack.distribution.datatypes import Api
        
        class MockDatasetIO:
            async def iterrows(self, dataset_id: str, limit: int = -1):
                if handler:
                    rows = await handler.get_rows(dataset_id, limit)
                    class MockResponse:
                        def __init__(self, data):
                            self.data = data
                    return MockResponse(rows)
                else:
                    raise ValueError(f"No dataset data available for {dataset_id}")
        
        class MockDatasets:
            async def get_dataset(self, dataset_id: str):
                return {"identifier": dataset_id, "data": dataset_data or []}
        
        deps = {
            Api.datasetio: MockDatasetIO(),
            Api.datasets: MockDatasets()
        }
        
        # Validate required parameters first
        if not job_uuid:
            raise HTTPException(status_code=400, detail="job_uuid is required")
        if not model:
            raise HTTPException(status_code=400, detail="model is required")
        if not finetuned_model:
            raise HTTPException(status_code=400, detail="finetuned_model is required")
        
        # Create DPO training instance with direct dataset access
        dpo_trainer = DPOTrainingSingleDevice(
            job_uuid=job_uuid,
            datasetio_api=deps[Api.datasetio],
            datasets_api=deps[Api.datasets]
        )
        
        # Create algorithm config
        algorithm_config = DPOAlignmentConfig(
            reward_scale=algorithm_config_dict.get("reward_scale", 1.0),
            reward_clip=algorithm_config_dict.get("reward_clip", 5.0),
            epsilon=algorithm_config_dict.get("epsilon", 0.1),
            gamma=algorithm_config_dict.get("gamma", 0.99)
        )
        
        # Create training config using the dict directly
        training_config = TrainingConfig(**training_config_dict) if training_config_dict else None
        
        if not training_config:
            raise HTTPException(status_code=400, detail="training_config is required")
        
        # Execute DPO training directly
        memory_stats, checkpoints = await dpo_trainer.train(
            model=model,
            output_dir=checkpoint_dir,
            job_uuid=job_uuid,
            dpo_config=algorithm_config,
            config=training_config,
            provider_config=config,
        )
        
        # Create result similar to what the inline provider would return
        class TrainingResult:
            def __init__(self, job_uuid):
                self.job_uuid = job_uuid
        
        result = TrainingResult(job_uuid)
        
        return {"job_uuid": result.job_uuid}
        
    except Exception as e:
        print(f"Error in preference_optimize: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "service:app",
        host="0.0.0.0",
        port=8080,
        reload=False  # Disable auto-reload to stop the chaos
    ) 