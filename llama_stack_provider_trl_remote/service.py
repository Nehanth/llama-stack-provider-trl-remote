"""
TRL Remote Training Service
===========================

This FastAPI service wraps the existing inline TRL provider and exposes it over HTTP.
It reuses all the same training logic, recipes, and configuration but runs as a
standalone service that can be deployed separately.

The service:
1. Creates an instance of the existing TrlPostTrainingImpl
2. Exposes its methods as HTTP endpoints
3. Handles request/response serialization
4. Provides health checks and service management
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Reuse existing TRL provider components
from llama_stack_provider_trl import get_provider_impl
from llama_stack_provider_trl.config import TrlPostTrainingConfig
from llama_stack.apis.post_training import (
    AlgorithmConfig,
    DPOAlignmentConfig,
    ListPostTrainingJobsResponse,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)

# Mock APIs for standalone service (replace with actual implementations)
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.distribution.datatypes import Api

logger = logging.getLogger(__name__)

# Global provider instance
trl_provider = None


# Request/Response Models for HTTP API
class TrainingRequest(BaseModel):
    """Base training request model."""
    job_uuid: str
    model: str
    provider_config: dict | None = None  # TrlPostTrainingConfig as dict
    checkpoint_dir: str | None = None


class SupervisedFinetuneRequest(TrainingRequest):
    """Request model for supervised fine-tuning."""
    training_config: dict  # TrainingConfig as dict
    hyperparam_search_config: dict
    logger_config: dict
    algorithm_config: dict | None = None


class PreferenceOptimizeRequest(TrainingRequest):
    """Request model for DPO training - accepts raw DPO algorithm config."""
    finetuned_model: str
    algorithm_config: dict  # Raw DPO config - can be DPO format from examples.ipynb
    training_config: dict  # TrainingConfig as dict
    hyperparam_search_config: dict
    logger_config: dict


class JobRequest(BaseModel):
    """Request model for job operations."""
    job_uuid: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager to initialize and cleanup the TRL provider.
    """
    global trl_provider
    
    # Initialize the TRL provider using existing components
    logger.info("Initializing TRL provider service...")
    
    # Create default configuration (can be overridden per request)
    config = TrlPostTrainingConfig()
    
    # Create mock API dependencies for standalone service
    # In production, these would be actual API clients
    datasetio_api = MockDatasetIO()
    datasets_api = MockDatasets()
    
    deps = {
        Api.datasetio: datasetio_api,
        Api.datasets: datasets_api,
    }
    
    # Create provider instance using existing get_provider_impl function
    trl_provider = await get_provider_impl(config, deps)
    
    logger.info("TRL provider service initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down TRL provider service...")
    if trl_provider:
        await trl_provider.shutdown()
    logger.info("TRL provider service shutdown complete")


# Create FastAPI app with lifespan management
app = FastAPI(
    title="TRL Training Service",
    description="Remote TRL (Transformer Reinforcement Learning) training service for DPO",
    version="1.0.0",
    lifespan=lifespan
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for the remote service."""
    return {"status": "healthy", "service": "trl-training-service"}


# PostTraining API Endpoints - reuse existing provider methods
@app.post("/supervised-fine-tune")
async def supervised_fine_tune(request: SupervisedFinetuneRequest):
    """Forward supervised fine-tuning to TRL provider."""
    if not trl_provider:
        raise HTTPException(status_code=500, detail="TRL provider not initialized")
        
    try:
        # Properly reconstruct Pydantic objects from JSON data
        from typing import cast
        training_config = TrainingConfig(**request.training_config)
        # AlgorithmConfig is a Union type, so we pass through as dict and cast for type checking
        algorithm_config = cast(AlgorithmConfig, request.algorithm_config) if request.algorithm_config else None
        
        # Call the existing provider method  
        result = await trl_provider.supervised_fine_tune(
            job_uuid=request.job_uuid,
            training_config=training_config,
            hyperparam_search_config=request.hyperparam_search_config,
            logger_config=request.logger_config,
            model=request.model,
            checkpoint_dir=request.checkpoint_dir,
            algorithm_config=algorithm_config,
        )
        
        return result.dict() if hasattr(result, 'dict') else result
        
    except Exception as e:
        logger.error(f"Supervised fine-tuning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preference-optimize")
async def preference_optimize(request: PreferenceOptimizeRequest):
    """Forward DPO training to TRL provider."""
    if not trl_provider:
        raise HTTPException(status_code=500, detail="TRL provider not initialized")
        
    try:
        # Properly reconstruct Pydantic objects from JSON data
        
        # Reconstruct TrainingConfig from dict
        training_config = TrainingConfig(**request.training_config)
        
        # Reconstruct AlgorithmConfig from dict
        # The algorithm config should be a DPO config when preference_optimize is called
        algorithm_config = DPOAlignmentConfig(**request.algorithm_config)
        
        # Update provider config if provided
        if request.provider_config:
            # Create new config with provided settings
            provider_config = TrlPostTrainingConfig(**request.provider_config)
            # Update the provider's config (simplified - in production might need provider recreation)
            trl_provider.config = provider_config
        
        # Call the existing provider method
        result = await trl_provider.preference_optimize(
            job_uuid=request.job_uuid,
            model=request.model,
            finetuned_model=request.finetuned_model,
            algorithm_config=algorithm_config,
            training_config=training_config,
            hyperparam_search_config=request.hyperparam_search_config,
            logger_config=request.logger_config,
            checkpoint_dir=request.checkpoint_dir,
        )
        
        return result.dict() if hasattr(result, 'dict') else result
        
    except Exception as e:
        logger.error(f"DPO training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training-jobs")
async def get_training_jobs():
    """Get list of training jobs."""
    if not trl_provider:
        raise HTTPException(status_code=500, detail="TRL provider not initialized")
        
    try:
        result = await trl_provider.get_training_jobs()
        return result.dict() if hasattr(result, 'dict') else result
    except Exception as e:
        logger.error(f"Getting training jobs failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training-job-status")
async def get_training_job_status(job_uuid: str):
    """Get training job status."""
    if not trl_provider:
        raise HTTPException(status_code=500, detail="TRL provider not initialized")
        
    try:
        result = await trl_provider.get_training_job_status(job_uuid)
        if result is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return result.dict() if hasattr(result, 'dict') else result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Getting job status failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cancel-training-job")
async def cancel_training_job(request: JobRequest):
    """Cancel training job."""
    if not trl_provider:
        raise HTTPException(status_code=500, detail="TRL provider not initialized")
        
    try:
        await trl_provider.cancel_training_job(request.job_uuid)
        return {"status": "cancelled", "job_uuid": request.job_uuid}
    except Exception as e:
        logger.error(f"Cancelling job failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training-job-artifacts")
async def get_training_job_artifacts(job_uuid: str):
    """Get training job artifacts."""
    if not trl_provider:
        raise HTTPException(status_code=500, detail="TRL provider not initialized")
        
    try:
        result = await trl_provider.get_training_job_artifacts(job_uuid)
        if result is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return result.dict() if hasattr(result, 'dict') else result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Getting job artifacts failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Mock API implementations for standalone service
class MockDatasetIO:
    """Mock DatasetIO for standalone service."""
    async def iterrows(self, dataset_id: str, limit: int = -1):
        # In production, this would connect to actual dataset storage
        raise NotImplementedError("Mock DatasetIO - implement with actual dataset source")


class MockDatasets:
    """Mock Datasets for standalone service."""
    async def get_dataset(self, dataset_id: str):
        # In production, this would connect to actual dataset API
        raise NotImplementedError("Mock Datasets - implement with actual dataset API")


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the service
    uvicorn.run(
        "service:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    ) 