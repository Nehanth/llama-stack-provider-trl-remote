# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
TRL Post-Training Provider Implementation
========================================

This file contains the main implementation of the TRL (Transformer Reinforcement Learning) provider
for Llama Stack. It implements the PostTraining protocol to provide DPO (Direct Preference
Optimization) training capabilities.

Architecture Overview:
1. TrlPostTrainingImpl: Main provider class that handles requests from Llama Stack
2. Job Scheduling: Uses Scheduler to run training jobs asynchronously in separate processes
3. Training Recipes: Delegates actual training to specialized recipe classes
4. Artifact Management: Tracks checkpoints and training statistics
5. Status Monitoring: Provides real-time status updates for training jobs

Key Design Patterns:
- Protocol Implementation: Implements Llama Stack's PostTraining interface
- Async Job Management: Non-blocking training with status monitoring
- Process Isolation: Training runs in separate process to avoid memory issues
- Artifact Tracking: Structured logging of checkpoints and metrics
"""

from enum import Enum
from typing import Any

# Import Llama Stack API interfaces that our provider needs to interact with
from llama_stack.apis.datasetio import DatasetIO    # For loading training datasets
from llama_stack.apis.datasets import Datasets      # For dataset operations
from llama_stack.apis.post_training import (
    AlgorithmConfig,                    # Base class for algorithm configurations
    Checkpoint,                         # Represents a saved model checkpoint
    DPOAlignmentConfig,                # Configuration for DPO algorithm
    JobStatus,                         # Enum for job status (scheduled, running, completed, etc.)
    ListPostTrainingJobsResponse,      # Response for listing all jobs
    PostTrainingJob,                   # Represents a training job
    PostTrainingJobArtifactsResponse,  # Response containing job artifacts (checkpoints)
    PostTrainingJobStatusResponse,     # Response containing job status and metadata
    TrainingConfig,                    # General training configuration
)

# Import our TRL-specific configuration
from llama_stack_provider_trl.config import TrlPostTrainingConfig

# Import our training recipe that does the actual DPO training
from llama_stack_provider_trl.recipes.dpo_training_single_device import DPOTrainingSingleDevice

# Import Llama Stack's job scheduling utilities
from llama_stack.providers.utils.scheduler import JobArtifact, Scheduler
from llama_stack.providers.utils.scheduler import JobStatus as SchedulerJobStatus

# Import decorator for web method registration (used for API endpoints)
from llama_stack.schema_utils import webmethod


class TrainingArtifactType(Enum):
    """
    Types of artifacts that can be produced during training.
    
    Artifacts are files or data produced during training that need to be
    tracked and made available to users after training completes.
    """
    CHECKPOINT = "checkpoint"        # Saved model weights and configuration
    RESOURCES_STATS = "resources_stats"  # Memory usage, GPU utilization, etc.


# Constant for identifying DPO training jobs in the scheduler
# This helps distinguish DPO jobs from other types of training jobs
_JOB_TYPE_DPO_TRAINING = "dpo-training"


class TrlPostTrainingImpl:
    """
    Main implementation class for the TRL post-training provider.
    
    This class implements Llama Stack's PostTraining protocol to provide
    DPO (Direct Preference Optimization) training capabilities using the
    TRL (Transformer Reinforcement Learning) library.
    
    Key Responsibilities:
    1. Handle training job requests from Llama Stack clients
    2. Schedule and manage asynchronous training jobs
    3. Provide status updates and artifact access for training jobs
    4. Coordinate with datasets and datasetio APIs for data loading
    5. Delegate actual training to specialized recipe classes
    
    Architecture:
    - Uses async job scheduling to run training in separate processes
    - Tracks training artifacts (checkpoints, statistics) throughout the process
    - Provides real-time status monitoring for long-running training jobs
    """
    
    def __init__(
        self,
        config: TrlPostTrainingConfig,  # Our TRL-specific configuration
        datasetio_api: DatasetIO,       # API for loading datasets from storage
        datasets: Datasets,             # API for dataset operations and transformations
    ) -> None:
        """
        Initialize the TRL post-training provider.
        
        Args:
            config: TrlPostTrainingConfig containing all DPO training settings
                   (device, DPO beta, reference model settings, etc.)
            datasetio_api: DatasetIO API for loading training datasets from storage
            datasets: Datasets API for dataset operations and transformations
        """
        # Store configuration and API references for use in training methods
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets
        
        # Create a scheduler for managing asynchronous training jobs
        # The scheduler handles running training in separate processes,
        # status tracking, and artifact collection
        self._scheduler = Scheduler()

    async def shutdown(self) -> None:
        """
        Clean shutdown of the provider.
        
        This method is called when the provider is being shut down.
        It ensures that all scheduled jobs are properly cleaned up
        and resources are released.
        """
        await self._scheduler.shutdown()

    @staticmethod
    def _checkpoint_to_artifact(checkpoint: Checkpoint) -> JobArtifact:
        """
        Convert a Checkpoint object to a JobArtifact for tracking.
        
        Checkpoints are model saves that happen during training. We need to
        convert them to JobArtifact format so the scheduler can track them
        and make them available through the API.
        
        Args:
            checkpoint: Checkpoint object containing model save information
            
        Returns:
            JobArtifact: Scheduler-compatible artifact representation
        """
        return JobArtifact(
            type=TrainingArtifactType.CHECKPOINT.value,  # Mark as checkpoint artifact
            name=checkpoint.identifier,                  # Use checkpoint ID as name
            uri=checkpoint.path,                        # Path to saved checkpoint files
            metadata=dict(checkpoint),                  # Include all checkpoint metadata
        )

    @staticmethod
    def _resources_stats_to_artifact(resources_stats: dict[str, Any]) -> JobArtifact:
        """
        Convert resource statistics to a JobArtifact for tracking.
        
        Resource statistics include information like memory usage, GPU utilization,
        training speed, etc. We track these as artifacts so users can monitor
        the efficiency of their training jobs.
        
        Args:
            resources_stats: Dictionary containing resource usage statistics
            
        Returns:
            JobArtifact: Scheduler-compatible artifact representation
        """
        return JobArtifact(
            type=TrainingArtifactType.RESOURCES_STATS.value,  # Mark as resource stats
            name=TrainingArtifactType.RESOURCES_STATS.value,  # Standard name for stats
            metadata=resources_stats,                         # Store stats in metadata
        )

    async def supervised_fine_tune(
        self,
        job_uuid: str,                               # Unique ID for this training job
        training_config: TrainingConfig,             # General training settings
        hyperparam_search_config: dict[str, Any],    # Hyperparameter search settings
        logger_config: dict[str, Any],               # Logging configuration
        model: str,                                  # Model identifier or path
        checkpoint_dir: str | None = None,           # Directory to save checkpoints
        algorithm_config: AlgorithmConfig | None = None,  # Algorithm-specific config
    ) -> PostTrainingJob:
        """
        Supervised Fine-Tuning method - NOT IMPLEMENTED in TRL provider.
        
        The TRL provider specializes in preference optimization (DPO), not
        supervised fine-tuning (SFT). For SFT, users should use the HuggingFace
        provider instead.
        
        This method raises NotImplementedError to clearly indicate that SFT
        is not supported and users should use a different provider.
        
        Args:
            job_uuid: Unique identifier for the training job
            training_config: General training configuration
            hyperparam_search_config: Hyperparameter search configuration
            logger_config: Logging configuration
            model: Model to fine-tune
            checkpoint_dir: Directory to save checkpoints
            algorithm_config: Algorithm-specific configuration
            
        Returns:
            PostTrainingJob: Would return job information if implemented
            
        Raises:
            NotImplementedError: Always raised since SFT is not supported
        """
        raise NotImplementedError(
            "Supervised fine-tuning is not implemented in TRL provider. "
            "Use preference_optimize instead for DPO training, or use the "
            "HuggingFace provider for supervised fine-tuning."
        )

    async def preference_optimize(
        self,
        job_uuid: str,                               # Unique ID for this training job
        model: str,                                  # Base model to train (e.g., "distilgpt2")
        finetuned_model: str,                       # Output model name (e.g., "my-dpo-model")
        algorithm_config: DPOAlignmentConfig,       # DPO-specific configuration
        training_config: TrainingConfig,            # General training settings
        hyperparam_search_config: dict[str, Any],   # Hyperparameter search settings
        logger_config: dict[str, Any],              # Logging configuration
        checkpoint_dir: str | None = None,          # Directory to save checkpoints
    ) -> PostTrainingJob:
        """
        Start a DPO (Direct Preference Optimization) training job.
        
        This is the main entry point for DPO training. It sets up an asynchronous
        training job that will run in a separate process, allowing the API to
        remain responsive while training is happening.
        
        The training process:
        1. Creates an async job handler function
        2. Schedules the job with the scheduler (returns immediately)
        3. The handler runs the actual training in a separate process
        4. Training artifacts and status updates are collected throughout
        5. Job completes and final artifacts are made available
        
        Args:
            job_uuid: Unique identifier for this training job
            model: Base model to train (HuggingFace model identifier like "distilgpt2")
            finetuned_model: Name for the output/fine-tuned model (used for saving)
            algorithm_config: DPOAlignmentConfig containing DPO-specific settings
                            like reward scaling, clipping, etc.
            training_config: TrainingConfig containing general training settings
                           like epochs, batch size, learning rate, dataset info
            hyperparam_search_config: Configuration for hyperparameter search
                                    (not fully implemented yet)
            logger_config: Configuration for training logging and monitoring
            checkpoint_dir: Directory where model checkpoints should be saved
                          (if None, a default directory will be created)
                          
        Returns:
            PostTrainingJob: Job object containing the job UUID for status tracking
        """
        
        async def handler(on_log_message_cb, on_status_change_cb, on_artifact_collected_cb):
            """
            Async job handler that runs the actual DPO training.
            
            This function is executed by the scheduler in a separate process.
            It coordinates the training and reports progress back through callbacks.
            
            Args:
                on_log_message_cb: Callback for sending log messages to the scheduler
                on_status_change_cb: Callback for updating job status
                on_artifact_collected_cb: Callback for reporting collected artifacts
            """
            # Log the start of training
            on_log_message_cb("Starting DPO training with TRL")

            # Create an instance of our DPO training recipe
            # The recipe contains all the actual training logic
            recipe = DPOTrainingSingleDevice(
                job_uuid=job_uuid,
                datasetio_api=self.datasetio_api,  # For loading datasets
                datasets_api=self.datasets_api,    # For dataset operations
            )

            # Run the actual DPO training
            # This is where the main training work happens
            resources_allocated, checkpoints = await recipe.train(
                model=model,                        # Base model to train (e.g., "distilgpt2")
                output_dir=checkpoint_dir,          # Where to save results
                job_uuid=job_uuid,                  # Job identifier
                dpo_config=algorithm_config,        # DPO algorithm settings
                config=training_config,             # General training settings
                provider_config=self.config,        # TRL provider configuration
            )

            # Report resource usage statistics as an artifact
            on_artifact_collected_cb(self._resources_stats_to_artifact(resources_allocated))
            
            # Report each checkpoint as an artifact
            if checkpoints:
                for checkpoint in checkpoints:
                    artifact = self._checkpoint_to_artifact(checkpoint)
                    on_artifact_collected_cb(artifact)

            # Update job status to completed and log completion
            on_status_change_cb(SchedulerJobStatus.completed)
            on_log_message_cb("DPO training completed")

        # Schedule the training job with the scheduler
        # This returns immediately while the training runs asynchronously
        job_uuid = self._scheduler.schedule(_JOB_TYPE_DPO_TRAINING, job_uuid, handler)
        
        # Return a PostTrainingJob object for the client to track progress
        return PostTrainingJob(job_uuid=job_uuid)

    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        """
        Get a list of all training jobs managed by this provider.
        
        This method returns information about all training jobs that have been
        scheduled through this provider instance, regardless of their current status
        (scheduled, running, completed, failed).
        
        Returns:
            ListPostTrainingJobsResponse: Contains list of PostTrainingJob objects
        """
        return ListPostTrainingJobsResponse(
            # Convert each scheduler job to a PostTrainingJob
            data=[PostTrainingJob(job_uuid=job.id) for job in self._scheduler.get_jobs()]
        )

    @staticmethod
    def _get_artifacts_metadata_by_type(job, artifact_type):
        """
        Extract metadata for artifacts of a specific type from a job.
        
        Helper method to filter job artifacts by type and return their metadata.
        This is useful for getting all checkpoints or all resource stats.
        
        Args:
            job: Scheduler job object containing artifacts
            artifact_type: Type of artifacts to extract (checkpoint, resources_stats, etc.)
            
        Returns:
            List of metadata dictionaries for artifacts of the specified type
        """
        return [artifact.metadata for artifact in job.artifacts if artifact.type == artifact_type]

    @classmethod
    def _get_checkpoints(cls, job):
        """
        Get all checkpoint artifacts from a job.
        
        Args:
            job: Scheduler job object
            
        Returns:
            List of checkpoint metadata dictionaries
        """
        return cls._get_artifacts_metadata_by_type(job, TrainingArtifactType.CHECKPOINT.value)

    @classmethod
    def _get_resources_allocated(cls, job):
        """
        Get resource allocation statistics from a job.
        
        Args:
            job: Scheduler job object
            
        Returns:
            Resource statistics dictionary, or None if no stats available
        """
        data = cls._get_artifacts_metadata_by_type(job, TrainingArtifactType.RESOURCES_STATS.value)
        return data[0] if data else None

    @webmethod(route="/post-training/job/status")
    async def get_training_job_status(self, job_uuid: str) -> PostTrainingJobStatusResponse | None:
        """
        Get the current status of a training job.
        
        This method provides real-time information about a training job,
        including its current status, timing information, checkpoints,
        and resource usage.
        
        Args:
            job_uuid: Unique identifier of the job to check
            
        Returns:
            PostTrainingJobStatusResponse: Detailed status information, or None if job not found
        """
        # Get the job from the scheduler
        job = self._scheduler.get_job(job_uuid)

        # Convert scheduler job status to API job status
        # The scheduler uses internal status values, but the API has its own enum
        match job.status:
            # New and scheduled jobs are both reported as "scheduled" to the API
            case SchedulerJobStatus.new | SchedulerJobStatus.scheduled:
                status = JobStatus.scheduled
            case SchedulerJobStatus.running:
                status = JobStatus.in_progress
            case SchedulerJobStatus.completed:
                status = JobStatus.completed
            case SchedulerJobStatus.failed:
                status = JobStatus.failed
            case _:
                # This should not happen, but we raise an error if it does
                raise NotImplementedError(f"Unknown job status: {job.status}")

        # Return comprehensive status information
        return PostTrainingJobStatusResponse(
            job_uuid=job_uuid,                              # Job identifier
            status=status,                                  # Current status
            scheduled_at=job.scheduled_at,                  # When job was scheduled
            started_at=job.started_at,                      # When training started
            completed_at=job.completed_at,                  # When training completed
            checkpoints=self._get_checkpoints(job),         # All saved checkpoints
            resources_allocated=self._get_resources_allocated(job),  # Resource usage stats
        )

    @webmethod(route="/post-training/job/cancel")
    async def cancel_training_job(self, job_uuid: str) -> None:
        """
        Cancel a running or scheduled training job.
        
        This method attempts to cancel a training job. If the job is still
        scheduled, it will be removed from the queue. If it's currently running,
        the training process will be terminated.
        
        Args:
            job_uuid: Unique identifier of the job to cancel
        """
        self._scheduler.cancel(job_uuid)

    @webmethod(route="/post-training/job/artifacts")
    async def get_training_job_artifacts(self, job_uuid: str) -> PostTrainingJobArtifactsResponse | None:
        """
        Get artifacts produced by a training job.
        
        Artifacts include model checkpoints saved during or after training.
        This method is useful for retrieving the final trained model or
        intermediate checkpoints for analysis.
        
        Args:
            job_uuid: Unique identifier of the job
            
        Returns:
            PostTrainingJobArtifactsResponse: Contains list of checkpoints and artifacts,
                                            or None if job not found
        """
        # Get the job from the scheduler
        job = self._scheduler.get_job(job_uuid)
        
        # Return artifacts response with checkpoint information
        return PostTrainingJobArtifactsResponse(
            job_uuid=job_uuid,
            checkpoints=self._get_checkpoints(job)  # List of all saved checkpoints
        ) 