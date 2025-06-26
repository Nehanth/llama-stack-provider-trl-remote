"""
TRL Remote Provider Adapter
===========================

This adapter implements the PostTraining protocol by forwarding requests
to a remote TRL training service over HTTP.

The adapter acts as an HTTP client that:
1. Receives PostTraining method calls from Llama Stack
2. Forwards them as HTTP requests to the remote TRL service
3. Returns the responses from the remote TRL service
"""

import asyncio
import json
from typing import Any
from enum import Enum

import aiohttp
from llama_stack.apis.post_training import (
    AlgorithmConfig,
    DPOAlignmentConfig,
    ListPostTrainingJobsResponse,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)

from .config import TrlRemoteConfig


def serialize_for_json(obj):
    """
    Serialize complex objects (Pydantic models, enums) for JSON transmission.
    
    This handles the DatasetFormat enum and other non-JSON-serializable types
    that are commonly found in Llama Stack data structures.
    """
    if hasattr(obj, 'dict'):
        # Pydantic model - convert to dict first
        data = obj.dict()
        return {k: serialize_for_json(v) for k, v in data.items()}
    elif isinstance(obj, Enum):
        # Enum - use the value
        return obj.value
    elif isinstance(obj, dict):
        # Dict - recursively serialize values
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # List/tuple - recursively serialize elements
        return [serialize_for_json(item) for item in obj]
    else:
        # Primitive type - return as-is
        return obj


class TrlRemoteAdapter:
    """
    Remote adapter for TRL provider that communicates with remote training service.

    """
    
    def __init__(self, config: TrlRemoteConfig):
        """
        Initialize the remote adapter.
        
        Args:
            config: TrlRemoteConfig containing remote service connection settings
                   and training configuration
        """
        self.config = config
    
    async def initialize(self) -> None:
        """
        Initialize the remote adapter and verify connectivity to remote service.
        
        This method is called by Llama Stack during provider initialization
        to ensure the remote service is available and responding.
        """
        try:
            # Test connectivity to remote service with health check
            health_response = await self._make_request("GET", "/health")
            if health_response.get("status") != "healthy":
                raise RuntimeError(f"Remote service is not healthy: {health_response}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize remote TRL adapter: {str(e)}")
            
    async def _make_request(self, method: str, endpoint: str, data: Any = None) -> dict:
        """
        Make HTTP request to remote service with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request payload for POST requests
            
        Returns:
            Response JSON data
        """
        url = f"{self.config.base_url}{endpoint}"
        
        # Serialize data for JSON transmission
        if data is not None:
            data = serialize_for_json(data)
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Create fresh session for each request to avoid event loop issues
                timeout = aiohttp.ClientTimeout(
                    total=self.config.timeout,
                    connect=self.config.connect_timeout
                )
                
                async with aiohttp.ClientSession(
                    timeout=timeout,
                    headers={"Content-Type": "application/json"}
                ) as session:
                    if method.upper() == "GET":
                        async with session.get(url) as response:
                            response.raise_for_status()
                            return await response.json()
                            
                    elif method.upper() == "POST":
                        async with session.post(url, json=data) as response:
                            response.raise_for_status()
                            return await response.json()
                        
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise RuntimeError(f"Request to {url} failed after {self.config.max_retries} retries: {str(e)}")
                    
                # Wait before retry
                await asyncio.sleep(self.config.retry_delay)
                
        # This should never be reached due to the exception handling above
        raise RuntimeError(f"Request to {url} failed unexpectedly")
                
    # === PostTraining Protocol Implementation ===
    # These methods forward requests to the remote service
    
    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
        model: str,
        checkpoint_dir: str | None = None,
        algorithm_config: AlgorithmConfig | None = None,
    ) -> PostTrainingJob:
        """ 
        Note: This will return NotImplementedError from the remote service
        since our TRL provider only supports DPO training.
        """
        raise NotImplementedError(
            "TRL provider only supports DPO training through the preference_optimize endpoint. "
            "Please use preference_optimize instead of supervised_fine_tune."
        )
        
    async def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
    ) -> PostTrainingJob:
        """
        Forward DPO training request to remote service.
        
        This is the proper endpoint for DPO training - uses DPOAlignmentConfig format.
        """
        
        # Get dataset data from client to include in the request
        dataset_data = None
        if training_config and training_config.data_config:
            try:
                # Fetch dataset data from the client's dataset API
                client_base_url = "http://localhost:8321"  # Client API base URL
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{client_base_url}/v1/datasets") as response:
                        if response.status == 200:
                            datasets_response = await response.json()
                            
                            # Find our dataset in the list
                            for dataset in datasets_response.get("data", []):
                                if dataset.get("identifier") == training_config.data_config.dataset_id:
                                    dataset_data = dataset.get("source", {}).get("rows", [])
                                    break
                                    
            except Exception as e:
                print(f"Warning: Could not fetch dataset data: {e}")
        
        algorithm_dict = algorithm_config.dict() if hasattr(algorithm_config, 'dict') else algorithm_config
        
        request_data = {
            "job_uuid": job_uuid,
            "model": finetuned_model,  
            "finetuned_model": finetuned_model,  
            "algorithm_config": algorithm_dict,  
            "training_config": training_config,
            "hyperparam_search_config": hyperparam_search_config,
            "logger_config": logger_config,
            "provider_config": self.config.training_config.dict(),
            "dataset_data": dataset_data  
        }
        
        response_data = await self._make_request("POST", "/preference-optimize", request_data)
        return PostTrainingJob(**response_data)
        
    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        """
        Get list of training jobs from remote service.
        """
        try:
            response_data = await self._make_request("GET", "/training-jobs")
            jobs_data = response_data.get("data", [])
            
            # Convert to PostTrainingJob objects
            jobs = []
            for job_data in jobs_data:
                job = PostTrainingJob(
                    job_uuid=job_data.get("job_uuid", ""),
                    # Add other fields as needed
                )
                jobs.append(job)
            
            return ListPostTrainingJobsResponse(data=jobs)
        except Exception as e:
            print(f"Error getting training jobs: {e}")
            return ListPostTrainingJobsResponse(data=[])
        
    async def get_training_job_status(self, job_uuid: str) -> PostTrainingJobStatusResponse | None:
        """
        Get training job status from remote service.
        """
        try:
            response_data = await self._make_request("GET", f"/training-job-status?job_uuid={job_uuid}")
            
            if not response_data:
                return None
                
            return PostTrainingJobStatusResponse(
                job_uuid=response_data.get("job_uuid", ""),
                status=response_data.get("status", "unknown"),
                scheduled_at=response_data.get("scheduled_at"),
                started_at=response_data.get("started_at"),
                completed_at=response_data.get("completed_at"),
                resources_allocated=response_data.get("resources_allocated"),
                checkpoints=response_data.get("checkpoints", [])
            )
        except Exception as e:
            print(f"Error getting job status for {job_uuid}: {e}")
            return None
        
    async def cancel_training_job(self, job_uuid: str) -> None:
        """
        Cancel training job on remote service.
        """
        try:
            await self._make_request("POST", "/cancel-training-job", {"job_uuid": job_uuid})
        except Exception as e:
            print(f"Error cancelling job {job_uuid}: {e}")
            # Don't raise since cancel is best-effort
        
    async def get_training_job_artifacts(self, job_uuid: str) -> PostTrainingJobArtifactsResponse | None:
        """
        Get training job artifacts from remote service.
        """
        try:
            response_data = await self._make_request("GET", f"/training-job-artifacts?job_uuid={job_uuid}")
            
            if not response_data:
                return None
                
            return PostTrainingJobArtifactsResponse(
                job_uuid=response_data.get("job_uuid", ""),
                checkpoints=response_data.get("checkpoints", [])
            )
        except Exception as e:
            print(f"Error getting job artifacts for {job_uuid}: {e}")
            return None 