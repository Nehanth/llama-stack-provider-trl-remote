"""
TRL Remote Training Service
===========================

FastAPI service that wraps the TRL inline provider and exposes it via HTTP.
This allows running TRL training jobs remotely while reusing all the existing 
training logic, recipes, and configurations.

Now includes asynchronous job tracking and status monitoring.
"""

import asyncio
import json
import tempfile
import os
import concurrent.futures
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from enum import Enum

from fastapi import FastAPI, HTTPException
from config import TrlPostTrainingConfig
from recipes import DPOTrainingUnified
from llama_stack.apis.post_training import (
    DPOAlignmentConfig,
    TrainingConfig,
    Checkpoint,
)


# === JOB TRACKING SYSTEM ===

class JobStatus(str, Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingJob:
    """Represents a training job with full lifecycle tracking"""
    
    def __init__(self, job_uuid: str, model: str, finetuned_model: str):
        self.job_uuid = job_uuid
        self.model = model
        self.finetuned_model = finetuned_model
        self.status = JobStatus.SCHEDULED
        self.scheduled_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.checkpoints: list[Checkpoint] = []
        self.training_task: Optional[asyncio.Task] = None
        
    def start(self):
        """Mark job as started"""
        self.status = JobStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc)
        
    def complete(self, checkpoints: Optional[list[Checkpoint]] = None):
        """Mark job as completed"""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        if checkpoints:
            self.checkpoints = checkpoints
            
    def fail(self, error: str):
        """Mark job as failed"""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error_message = error
        
    def to_dict(self):
        """Convert job to dictionary for API responses"""
        return {
            "job_uuid": self.job_uuid,
            "model": self.model,
            "finetuned_model": self.finetuned_model,
            "status": self.status.value,
            "scheduled_at": self.scheduled_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "checkpoints": [
                {
                    "identifier": cp.identifier,
                    "created_at": cp.created_at.isoformat(),
                    "epoch": cp.epoch,
                    "post_training_job_id": cp.post_training_job_id,
                    "path": cp.path
                } for cp in self.checkpoints
            ] if self.checkpoints else []
        }


class JobManager:
    """Manages training jobs with async execution and status tracking"""
    
    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        
    def create_job(self, job_uuid: str, model: str, finetuned_model: str) -> TrainingJob:
        """Create a new training job"""
        if job_uuid in self.jobs:
            raise ValueError(f"Job {job_uuid} already exists")
            
        job = TrainingJob(job_uuid, model, finetuned_model)
        self.jobs[job_uuid] = job
        return job
        
    def get_job(self, job_uuid: str) -> Optional[TrainingJob]:
        """Get job by UUID"""
        return self.jobs.get(job_uuid)
        
    def list_jobs(self) -> list[TrainingJob]:
        """List all jobs"""
        return list(self.jobs.values())
        
    def _run_training_sync(self, job: TrainingJob, training_params: dict):
        """Synchronous training function to run in thread pool"""
        try:
            job.start()
            print(f"Starting training job {job.job_uuid}")
            
            # Extract training parameters
            model = training_params["model"]
            checkpoint_dir = training_params["checkpoint_dir"]
            algorithm_config = training_params["algorithm_config"]
            training_config = training_params["training_config"]
            provider_config = training_params["provider_config"]
            dataset_data = training_params["dataset_data"]
            
            # Use dataset data directly - no mock layer needed
            
            # Create unified DPO trainer (handles both single and multi-GPU automatically)
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            print(f"Using unified DPO trainer (WORLD_SIZE={world_size})")
            
            dpo_trainer = DPOTrainingUnified(
                job_uuid=job.job_uuid,
                dataset_data=dataset_data
            )
            
            # Execute training (synchronous)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            memory_stats, checkpoints = loop.run_until_complete(dpo_trainer.train(
                model=model,
                output_dir=checkpoint_dir,
                job_uuid=job.job_uuid,
                dpo_config=algorithm_config,
                config=training_config,
                provider_config=provider_config,
            ))
            
            loop.close()
            
            # Mark job as completed
            job.complete(checkpoints)
            print(f"Training job {job.job_uuid} completed successfully")
            
        except Exception as e:
            # Mark job as failed
            job.fail(str(e))
            print(f"Training job {job.job_uuid} failed: {str(e)}")
            import traceback
            traceback.print_exc()

    async def execute_training(
        self,
        job: TrainingJob,
        training_params: dict
    ):
        """Execute training in background thread pool to keep server responsive"""
        loop = asyncio.get_event_loop()
        
        # Use thread pool executor for CPU-bound training
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(
                executor,
                self._run_training_sync,
                job,
                training_params
            )


# No more mock/handler classes needed - dataset_data passed directly


# === FASTAPI SERVICE ===

app = FastAPI(title="TRL Remote Training Service")

# Global job manager
job_manager = JobManager()


@app.on_event("startup")
async def startup_event():
    """Initialize the TRL provider on service startup"""
    print("Starting TRL Remote Training Service...")
    print("Job tracking system initialized")
    print("Service initialized successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "trl-training-service"}


@app.get("/training-jobs")
async def get_training_jobs():
    """Get list of all training jobs with their current status"""
    jobs = job_manager.list_jobs()
    return {
        "data": [job.to_dict() for job in jobs]
    }


@app.get("/training-job-status")
async def get_training_job_status(job_uuid: str):
    """Get training job status by UUID"""
    job = job_manager.get_job(job_uuid)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_uuid} not found")
    
    return {
        "job_uuid": job.job_uuid,
        "status": job.status.value,
        "scheduled_at": job.scheduled_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "resources_allocated": None,
        "checkpoints": [
            {
                "identifier": cp.identifier,
                "created_at": cp.created_at.isoformat(),
                "epoch": cp.epoch,
                "post_training_job_id": cp.post_training_job_id,
                "path": cp.path
            } for cp in job.checkpoints
        ] if job.checkpoints else []
    }


@app.post("/cancel-training-job")
async def cancel_training_job(request: dict):
    """Cancel a training job"""
    job_uuid = request.get("job_uuid")
    if not job_uuid:
        raise HTTPException(status_code=400, detail="job_uuid is required")
    
    job = job_manager.get_job(job_uuid)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_uuid} not found")
    
    if job.status == JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Cannot cancel completed job")
    elif job.status == JobStatus.FAILED:
        raise HTTPException(status_code=400, detail="Cannot cancel failed job")
    
    # Cancel the training task if it's running
    if job.training_task and not job.training_task.done():
        job.training_task.cancel()
        job.fail("Cancelled by user")
        return {"status": "cancelled", "job_uuid": job_uuid}
    else:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")


@app.get("/training-job-artifacts")
async def get_training_job_artifacts(job_uuid: str):
    """Get training job artifacts and checkpoints"""
    job = job_manager.get_job(job_uuid)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_uuid} not found")
    
    return {
        "job_uuid": job.job_uuid,
        "checkpoints": [
            {
                "identifier": cp.identifier,
                "created_at": cp.created_at.isoformat(),
                "epoch": cp.epoch,
                "post_training_job_id": cp.post_training_job_id,
                "path": cp.path,
                "training_metrics": None  # Could be extended to include metrics
            } for cp in job.checkpoints
        ] if job.checkpoints else []
    }


@app.post("/preference-optimize")
async def preference_optimize_endpoint(request: dict):
    """Submit DPO training job for asynchronous execution"""
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
        
        # Validate required parameters
        if not job_uuid:
            raise HTTPException(status_code=400, detail="job_uuid is required")
        if not model:
            raise HTTPException(status_code=400, detail="model is required")
        if not finetuned_model:
            raise HTTPException(status_code=400, detail="finetuned_model is required")
        
        # Create job
        try:
            job = job_manager.create_job(job_uuid, model, finetuned_model)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        
        # Create configurations
        config = TrlPostTrainingConfig(**provider_config_dict) if provider_config_dict else TrlPostTrainingConfig()
        
        algorithm_config = DPOAlignmentConfig(
            reward_scale=algorithm_config_dict.get("reward_scale", 1.0),
            reward_clip=algorithm_config_dict.get("reward_clip", 5.0),
            epsilon=algorithm_config_dict.get("epsilon", 0.1),
            gamma=algorithm_config_dict.get("gamma", 0.99)
        )
        
        training_config = TrainingConfig(**training_config_dict) if training_config_dict else None
        if not training_config:
            raise HTTPException(status_code=400, detail="training_config is required")
        
        # Prepare training parameters
        training_params = {
            "model": model,
            "checkpoint_dir": checkpoint_dir,
            "algorithm_config": algorithm_config,
            "training_config": training_config,
            "provider_config": config,
            "dataset_data": dataset_data
        }
        
        # Start training in background
        job.training_task = asyncio.create_task(
            job_manager.execute_training(job, training_params)
        )
        
        print(f"Job {job_uuid} submitted for training")
        
        # Return job info immediately (non-blocking)
        return {"job_uuid": job.job_uuid}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in preference_optimize: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Handle distributed training: only rank 0 runs the FastAPI server
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        print(f"Distributed training detected: WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
        
        if local_rank == 0:
            print("Rank 0: Starting FastAPI server on port 8080")
            uvicorn.run(
                "service:app",
                host="0.0.0.0",
                port=8080,
                reload=False
            )
        else:
            print(f"Rank {local_rank}: Waiting for training jobs (no HTTP server)")
            # Keep process alive to participate in distributed training when needed
            import time
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print(f"Rank {local_rank}: Shutting down")
    else:
        print("Single-device mode: Starting FastAPI server on port 8080")
        uvicorn.run(
            "service:app",
            host="0.0.0.0",
            port=8080,
            reload=False
        ) 