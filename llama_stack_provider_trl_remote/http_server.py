#!/usr/bin/env python3
"""
TRL Remote HTTP Server
======================

Standalone FastAPI server that accepts training jobs and launches
multi-GPU distributed training via torchrun when needed.

Architecture:
- HTTP Server (this file): Accepts jobs, manages job queue
- Training Workers: Multi-GPU torchrun processes that execute training
- Communication: Shared job files and status files
"""

import asyncio
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from fastapi import FastAPI, HTTPException

# Job status tracking
class JobStatus(str, Enum):
    QUEUED = "queued"
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
        self.status = JobStatus.QUEUED  # Jobs start as queued
        self.scheduled_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.checkpoints: list = []
        self.torchrun_process: Optional[subprocess.Popen] = None
        self.training_logs: list = []  # Store training output
        
    def start(self):
        """Mark job as started"""
        self.status = JobStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc)
        
    def complete(self, checkpoints: Optional[list] = None):
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
            "checkpoints": self.checkpoints
        }


class JobManager:
    """Manages training jobs and launches torchrun workers"""
    
    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self.job_queue_dir = Path("./job_queue")
        self.job_queue_dir.mkdir(exist_ok=True)
        self.monitor_task: Optional[asyncio.Task] = None
        self.current_training_job: Optional[TrainingJob] = None  # Track active job
        self.training_lock = asyncio.Lock()  # Ensure only one job runs at a time
        self.job_queue: asyncio.Queue = asyncio.Queue()  # Queue for pending jobs
        self.queue_processor_task: Optional[asyncio.Task] = None  # Task to process queue
        
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
        
    async def queue_job(self, job: TrainingJob, training_params: dict):
        """Add a job to the queue for processing"""
        await self.job_queue.put((job, training_params))
        print(f"Job {job.job_uuid} added to queue (queue size: {self.job_queue.qsize()})")
        
    def capture_training_logs(self, job: TrainingJob):
        """Capture training logs from subprocess output"""
        if job.torchrun_process and job.torchrun_process.stdout:
            try:
                # Read available output without blocking
                import select
                if select.select([job.torchrun_process.stdout], [], [], 0)[0]:
                    line = job.torchrun_process.stdout.readline()
                    if line:
                        job.training_logs.append({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "message": line.strip()
                        })
                        # Keep only last 100 log lines to prevent memory issues
                        if len(job.training_logs) > 100:
                            job.training_logs.pop(0)
            except Exception as e:
                pass  # Ignore errors reading logs
        
    async def submit_training_job(self, job: TrainingJob, training_params: dict):
        """Submit job to distributed training workers"""
        try:
            # Set this as the current training job
            self.current_training_job = job
            
            # Detect available GPUs dynamically
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise RuntimeError("No CUDA GPUs available for training")
            
            # Write job file for workers to pick up
            job_file = self.job_queue_dir / f"{job.job_uuid}.json"
            with open(job_file, 'w') as f:
                json.dump(training_params, f, indent=2)
            
            # Launch multi-GPU torchrun workers (dynamic GPU count)
            cmd = [
                "torchrun",
                f"--nproc_per_node={num_gpus}",
                "--nnodes=1", 
                "--node_rank=0",
                "--master_addr=localhost",
                "--master_port=29500",
                "training_worker.py",
                job.job_uuid
            ]
            
            # Set environment for better NCCL stability
            gpu_list = ",".join(str(i) for i in range(num_gpus))
            env = os.environ.copy()
            env.update({
                "NCCL_DEBUG": "INFO",
                "NCCL_SOCKET_IFNAME": "lo", 
                "NCCL_P2P_DISABLE": "1",
                "CUDA_VISIBLE_DEVICES": gpu_list
            })
            
            print(f"Launching {num_gpus}-GPU training for job {job.job_uuid}")
            job.torchrun_process = subprocess.Popen(
                cmd,
                cwd=Path(__file__).parent,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            job.start()
            print(f"Job {job.job_uuid} started with PID {job.torchrun_process.pid} on {num_gpus} GPUs")
            
        except Exception as e:
            job.fail(str(e))
            self.current_training_job = None  # Clear current job on failure
            print(f"Failed to start job {job.job_uuid}: {e}")
            raise
            
    async def wait_for_completion(self, job: TrainingJob, check_interval: float = 2.0) -> None:
        """Wait for a job to complete by monitoring its status"""
        while job.status == JobStatus.IN_PROGRESS:
            if job.torchrun_process:
                # Capture any new output from the process
                self.capture_training_logs(job)
                
                # Check if process is still running
                if job.torchrun_process.poll() is not None:
                    # Process finished, check status file
                    status_file = self.job_queue_dir / f"{job.job_uuid}_status.json"
                    
                    # Wait a bit for the status file to be written
                    for _ in range(5):  # Try for up to 10 seconds
                        if status_file.exists():
                            break
                        await asyncio.sleep(2)
                    
                    if status_file.exists():
                        with open(status_file) as f:
                            status_data = json.load(f)
                        
                        if status_data.get("status") == "completed":
                            job.complete(status_data.get("checkpoints", []))
                            print(f"Job {job.job_uuid} completed successfully")
                        else:
                            error = status_data.get("error", "Unknown error")
                            job.fail(error)
                            print(f"Job {job.job_uuid} failed: {error}")
                            
                        # Cleanup files
                        status_file.unlink(missing_ok=True)
                        (self.job_queue_dir / f"{job.job_uuid}.json").unlink(missing_ok=True)
                    else:
                        job.fail("Training process exited without status")
                    
                    # Clear current job when done
                    if self.current_training_job == job:
                        self.current_training_job = None
                    
                    # Job is done, exit the loop
                    break
                    
            await asyncio.sleep(check_interval)
            
    async def process_job_queue(self):
        """Background task to process queued jobs sequentially"""
        while True:
            try:
                # Wait for a job in the queue
                job, training_params = await self.job_queue.get()
                
                print(f"Processing job {job.job_uuid} from queue")
                
                # Update status to scheduled now that we're processing it
                job.status = JobStatus.SCHEDULED
                
                # Wait for any current job to complete
                while self.current_training_job and self.current_training_job.status == JobStatus.IN_PROGRESS:
                    await asyncio.sleep(2)
                
                # Submit the job
                await self.submit_training_job(job, training_params)
                
                # Wait for this job to complete before processing the next one
                await self.wait_for_completion(job)
                
            except Exception as e:
                print(f"Error processing job queue: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                
    async def monitor_jobs(self):
        """Background task to monitor job status"""
        while True:
            try:
                for job in self.jobs.values():
                    if job.status == JobStatus.IN_PROGRESS and job.torchrun_process:
                        # Capture any new output from the process
                        self.capture_training_logs(job)
                        
                        # Check if process is still running
                        if job.torchrun_process.poll() is not None:
                            # Process finished, check status file
                            status_file = self.job_queue_dir / f"{job.job_uuid}_status.json"
                            if status_file.exists():
                                with open(status_file) as f:
                                    status_data = json.load(f)
                                
                                if status_data.get("status") == "completed":
                                    job.complete(status_data.get("checkpoints", []))
                                    print(f"Job {job.job_uuid} completed successfully")
                                else:
                                    error = status_data.get("error", "Unknown error")
                                    job.fail(error)
                                    print(f"Job {job.job_uuid} failed: {error}")
                                    
                                # Cleanup files
                                status_file.unlink(missing_ok=True)
                                (self.job_queue_dir / f"{job.job_uuid}.json").unlink(missing_ok=True)
                                
                                # Clear current job when done
                                if self.current_training_job == job:
                                    self.current_training_job = None
                            else:
                                job.fail("Training process exited without status")
                                
            except Exception as e:
                print(f"Error monitoring jobs: {e}")
                
            await asyncio.sleep(2)  # Check every 2 seconds


# FastAPI app
app = FastAPI(title="TRL Remote Training Service")
job_manager = JobManager()


@app.on_event("startup")
async def startup_event():
    """Initialize the HTTP server"""
    num_gpus = torch.cuda.device_count()
    print("Starting TRL Remote HTTP Server...")
    print("Job tracking system initialized")
    print(f"Detected {num_gpus} GPU(s) available")
    print(f"Ready to accept {num_gpus}-GPU training jobs" if num_gpus > 1 else "Ready to accept single-GPU training jobs")
    # Start job monitor task
    job_manager.monitor_task = asyncio.create_task(job_manager.monitor_jobs())
    # Start job queue processor task
    job_manager.queue_processor_task = asyncio.create_task(job_manager.process_job_queue())
    print("Job queue processor started")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "trl-http-server"}


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
        "checkpoints": job.checkpoints
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
    
    # Kill the torchrun process
    if job.torchrun_process and job.torchrun_process.poll() is None:
        job.torchrun_process.terminate()
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
        "checkpoints": job.checkpoints
    }


@app.get("/training-job-logs")
async def get_training_job_logs(job_uuid: str):
    """Get real-time training logs from subprocess output"""
    job = job_manager.get_job(job_uuid)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_uuid} not found")
    
    # Capture any new logs before returning
    if job.status == JobStatus.IN_PROGRESS:
        job_manager.capture_training_logs(job)
    
    return {
        "job_uuid": job.job_uuid,
        "status": job.status.value,
        "logs": job.training_logs[-50:] if job.training_logs else [],  # Return last 50 lines
        "total_lines": len(job.training_logs)
    }


@app.post("/preference-optimize")
async def preference_optimize_endpoint(request: dict):
    """Submit DPO training job for multi-GPU execution"""
    try:
        # Extract request data
        job_uuid = request.get("job_uuid")
        model = request.get("model")
        finetuned_model = request.get("finetuned_model")
        
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
        
        # Add job to queue instead of submitting directly
        await job_manager.queue_job(job, request)
        
        print(f"Job {job_uuid} added to training queue")
        return {
            "job_uuid": job.job_uuid,
            "status": "queued",
            "message": f"Job queued for training. Queue position: {job_manager.job_queue.qsize()}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in preference_optimize: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Starting standalone TRL HTTP Server on port 8080")
    uvicorn.run(
        "http_server:app",
        host="0.0.0.0",
        port=8080,
        reload=False
    ) 