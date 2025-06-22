"""
TRL Remote Provider Adapter
===========================

This adapter implements the PostTraining protocol by forwarding requests
to a remote TRL training service over HTTP. It reuses all the same data types
and interfaces as the inline provider.

The adapter acts as an HTTP client that:
1. Receives PostTraining method calls from Llama Stack
2. Forwards them as HTTP requests to the remote TRL service
3. Returns responses in the same format as the inline provider
"""

import asyncio
import json
from typing import Any
from enum import Enum

import aiohttp
from llama_stack.apis.post_training import (
    AlgorithmConfig,
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
    
    This adapter implements the same PostTraining interface as the inline provider
    but forwards all requests to a remote service over HTTP.
    
    The remote service runs the same TRL training logic as the inline provider,
    just in a separate process/service.
    """
    
    def __init__(self, config: TrlRemoteConfig):
        """
        Initialize the remote adapter.
        
        Args:
            config: TrlRemoteConfig containing remote service connection settings
                   and training configuration
        """
        self.config = config
        self.session: aiohttp.ClientSession | None = None
        
    async def initialize(self):
        """
        Initialize the HTTP client session and validate remote service connection.
        """
        # Create HTTP client session with timeout configuration
        timeout = aiohttp.ClientTimeout(
            total=self.config.timeout,
            connect=self.config.connect_timeout
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        
        # Validate that the remote service is available
        await self._health_check()
        
    async def shutdown(self):
        """
        Clean shutdown of the HTTP client session.
        """
        if self.session:
            await self.session.close()
            
    async def _health_check(self):
        """
        Check if the remote TRL service is available and healthy.
        """
        if not self.session:
            raise RuntimeError("HTTP session not initialized")
            
        try:
            async with self.session.get(f"{self.config.base_url}/health") as response:
                if response.status != 200:
                    raise RuntimeError(f"Remote TRL service health check failed: {response.status}")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to remote TRL service at {self.config.base_url}: {str(e)}")
            
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
        if not self.session:
            raise RuntimeError("HTTP session not initialized")
            
        url = f"{self.config.base_url}{endpoint}"
        
        # Serialize data for JSON transmission
        if data is not None:
            data = serialize_for_json(data)
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if method.upper() == "GET":
                    async with self.session.get(url) as response:
                        response.raise_for_status()
                        return await response.json()
                        
                elif method.upper() == "POST":
                    async with self.session.post(url, json=data) as response:
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
        Forward supervised fine-tuning request to remote service.
        
        Note: This will return NotImplementedError from the remote service
        since TRL provider only supports DPO training.
        """
        request_data = {
            "job_uuid": job_uuid,
            "training_config": training_config,
            "hyperparam_search_config": hyperparam_search_config,
            "logger_config": logger_config,
            "model": model,
            "checkpoint_dir": checkpoint_dir,
            "algorithm_config": algorithm_config,
            "provider_config": self.config.training_config.dict()
        }
        
        response_data = await self._make_request("POST", "/supervised-fine-tune", request_data)
        return PostTrainingJob(**response_data)
        
    async def preference_optimize(
        self,
        job_uuid: str,
        model: str,
        finetuned_model: str,
        algorithm_config: AlgorithmConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
        checkpoint_dir: str | None = None,
    ) -> PostTrainingJob:
        """
        Forward DPO training request to remote service.
        
        This now supports native DPO requests without any transformation workarounds.
        """
        request_data = {
            "job_uuid": job_uuid,
            "model": model,
            "finetuned_model": finetuned_model,
            "algorithm_config": algorithm_config,
            "training_config": training_config,
            "hyperparam_search_config": hyperparam_search_config,
            "logger_config": logger_config,
            "checkpoint_dir": checkpoint_dir,
            "provider_config": self.config.training_config.dict()
        }
        
        response_data = await self._make_request("POST", "/preference-optimize", request_data)
        return PostTrainingJob(**response_data)
        
    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        """
        Get list of training jobs from remote service.
        """
        response_data = await self._make_request("GET", "/training-jobs")
        return ListPostTrainingJobsResponse(**response_data)
        
    async def get_training_job_status(self, job_uuid: str) -> PostTrainingJobStatusResponse | None:
        """
        Get training job status from remote service.
        """
        response_data = await self._make_request("GET", f"/training-job-status?job_uuid={job_uuid}")
        if response_data is None:
            return None
        return PostTrainingJobStatusResponse(**response_data)
        
    async def cancel_training_job(self, job_uuid: str) -> None:
        """
        Cancel training job on remote service.
        """
        await self._make_request("POST", f"/cancel-training-job", {"job_uuid": job_uuid})
        
    async def get_training_job_artifacts(self, job_uuid: str) -> PostTrainingJobArtifactsResponse | None:
        """
        Get training job artifacts from remote service.
        """
        response_data = await self._make_request("GET", f"/training-job-artifacts?job_uuid={job_uuid}")
        if response_data is None:
            return None
        return PostTrainingJobArtifactsResponse(**response_data) 