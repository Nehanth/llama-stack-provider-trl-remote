# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
DPO Training Recipe for Single Device
=====================================

This file contains the core implementation of DPO (Direct Preference Optimization) training
using HuggingFace's TRL (Transformer Reinforcement Learning) library. This recipe handles
the entire DPO training pipeline from dataset loading to model saving.

DPO Training Overview:
DPO is a method for training language models to follow human preferences. It works by:
1. Taking a base model (usually SFT-trained)
2. Training on preference data (prompt, chosen response, rejected response)
3. Optimizing the model to prefer "chosen" responses over "rejected" ones
4. Using a reference model to prevent the model from drifting too far

Key Components:
- Dataset Processing: Validates and formats preference data
- Model Setup: Loads base model and reference model for DPO
- Training Loop: Uses TRL's DPOTrainer for optimization
- Checkpoint Management: Saves model at regular intervals
- Resource Monitoring: Tracks memory and GPU usage

This implementation supports single-device training (CPU, single GPU, or MPS).
For multi-GPU training, a separate recipe would be needed.
"""

# Standard library imports for system operations and utilities
import asyncio                  # Async programming for concurrent operations
import gc                        # Garbage collection for memory cleanup
import json                      # JSON parsing (if needed for configs)
import logging                   # Logging training progress and debugging
import multiprocessing          # For CPU count and process management
import os                       # Operating system interface for paths and environment
import signal                   # Signal handling for graceful shutdown
import sys                      # System-specific parameters and functions
import tempfile                 # Temporary file and directory creation
from datetime import datetime, timezone  # Date/time utilities for timestamps
from pathlib import Path        # Object-oriented path handling
from typing import Any          # Type hints for better code documentation

# Third-party system monitoring
import psutil                   # Process and system monitoring (memory, CPU usage)

# === ENVIRONMENT VARIABLE CONFIGURATION ===
# These settings optimize performance and avoid conflicts with different backends

# Disable tokenizer parallelism to avoid deadlocks in multiprocessing environments
# This is important when using HuggingFace tokenizers in async or multi-process setups
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force PyTorch to use OpenBLAS instead of Intel MKL for better compatibility
# These settings prevent issues with Intel MKL threading in some environments
os.environ["MKL_THREADING_LAYER"] = "GNU"      # Use GNU threading layer
os.environ["MKL_SERVICE_FORCE_INTEL"] = "0"    # Don't force Intel optimizations
os.environ["MKL_NUM_THREADS"] = "1"            # Limit MKL threads to avoid conflicts

# === PYTORCH AND MACHINE LEARNING IMPORTS ===
import torch                    # PyTorch deep learning framework
from datasets import Dataset    # HuggingFace datasets for data handling
from transformers import (      # HuggingFace transformers library
    AutoConfig,                 # Automatic model configuration loading
    AutoModelForCausalLM,      # Automatic causal language model loading
    AutoTokenizer,             # Automatic tokenizer loading
    TrainingArguments,         # Training configuration (may be used for compatibility)
)
from trl import DPOConfig, DPOTrainer  # TRL library for DPO training

# === LLAMA STACK API IMPORTS ===
# These are the Llama Stack APIs that our provider integrates with
from llama_stack.apis.datasetio import DatasetIO      # For loading datasets from storage
from llama_stack.apis.datasets import Datasets        # For dataset operations
from llama_stack.apis.post_training import (           # Post-training API types
    Checkpoint,                 # Represents a saved model checkpoint
    DataConfig,                # Dataset configuration settings
    DPOAlignmentConfig,        # DPO-specific algorithm configuration
    TrainingConfig,            # General training configuration
)

# Import our provider's configuration
from ..config import TrlPostTrainingConfig

# Set up logging for this module
logger = logging.getLogger(__name__)


def get_gb(to_convert: int) -> str:
    """
    Convert memory values from bytes to gigabytes with formatting.
    
    This utility function helps make memory statistics human-readable by converting
    large byte values to GB format with 2 decimal places.
    
    Args:
        to_convert: Memory value in bytes
        
    Returns:
        str: Memory value in GB formatted to 2 decimal places (e.g., "2.34")
    """
    return f"{(to_convert / (1024**3)):.2f}"


def get_memory_stats(device: torch.device) -> dict[str, Any]:
    """
    Get comprehensive memory statistics for different device types.
    
    This function collects memory usage information that's specific to the device
    being used for training. It's essential for monitoring resource usage and
    debugging memory issues during DPO training.
    
    Args:
        device: PyTorch device object (cuda, cpu, mps, etc.)
        
    Returns:
        dict: Dictionary containing memory statistics organized by type:
              - system_memory: Overall system RAM usage
              - device_memory: Device-specific memory (GPU VRAM, process memory, etc.)
    """
    # Start with system memory stats (available on all systems)
    stats = {
        "system_memory": {
            "total": get_gb(psutil.virtual_memory().total),        # Total system RAM
            "available": get_gb(psutil.virtual_memory().available), # Available RAM
            "used": get_gb(psutil.virtual_memory().used),          # Used RAM
            "percent": psutil.virtual_memory().percent,            # Usage percentage
        }
    }

    # Add device-specific memory statistics
    if device.type == "cuda":
        # NVIDIA GPU memory statistics
        stats["device_memory"] = {
            "allocated": get_gb(torch.cuda.memory_allocated(device)),     # Memory allocated by PyTorch
            "reserved": get_gb(torch.cuda.memory_reserved(device)),       # Memory reserved by CUDA
            "max_allocated": get_gb(torch.cuda.max_memory_allocated(device)), # Peak memory usage
        }
    elif device.type == "mps":
        # Apple Silicon (M1/M2) GPU - MPS doesn't provide direct memory stats
        stats["device_memory"] = {
            "note": "MPS memory stats not directly available",            # Explanation
            "system_memory_used": get_gb(psutil.virtual_memory().used),   # Fall back to system memory
        }
    elif device.type == "cpu":
        # CPU training - track process-specific memory usage
        process = psutil.Process()
        stats["device_memory"] = {
            "process_rss": get_gb(process.memory_info().rss),      # Resident Set Size (physical memory)
            "process_vms": get_gb(process.memory_info().vms),      # Virtual Memory Size
            "process_percent": process.memory_percent(),           # Percentage of system memory used
        }

    return stats


def setup_torch_device(device_str: str) -> torch.device:
    """
    Initialize and validate a PyTorch device with comprehensive error checking.
    
    This function handles device initialization and validation for different device types.
    It ensures that the requested device is available and properly configured before
    training begins, preventing runtime errors later in the training process.
    
    Supported devices:
    - CUDA: NVIDIA GPUs (validates CUDA availability and device selection)
    - MPS: Apple Silicon GPUs (validates MPS availability for M1/M2 Macs)
    - CPU: CPU-only training (basic validation)
    - HPU: Intel Gaudi (raises error as it's not supported)
    
    Args:
        device_str: String specifying the device ('cuda', 'cpu', 'mps', etc.)
        
    Returns:
        torch.device: The initialized and validated PyTorch device
        
    Raises:
        RuntimeError: If device initialization fails or device is not supported
    """
    try:
        # Create PyTorch device object from string
        device = torch.device(device_str)
    except RuntimeError as e:
        raise RuntimeError(f"Error getting Torch Device {str(e)}") from e

    # Validate device capabilities and availability
    if device.type == "cuda":
        # NVIDIA CUDA GPU validation
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{device.type}: Torch has no CUDA/ROCm support or could not detect a compatible device."
            )
        # If no specific GPU index provided, use current device
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
            
    elif device.type == "mps":
        # Apple Silicon GPU validation
        if not torch.backends.mps.is_available():
            raise RuntimeError(f"{device.type}: Torch has no MPS support or could not detect a compatible device.")
            
    elif device.type == "hpu":
        # Intel Gaudi HPU - not supported in this implementation
        raise RuntimeError(f"{device.type}: training does not support Intel Gaudi.")

    return device


class DPOTrainingSingleDevice:
    """
    Single-device DPO training recipe using TRL.
    
    This class implements the complete DPO training pipeline for single-device setups.
    It handles everything from data loading to model saving, providing a complete
    solution for preference optimization training.
    
    Key Features:
    - Dataset validation and formatting for DPO
    - Model and tokenizer setup with proper configurations
    - Reference model management for DPO loss calculation
    - Training execution with progress monitoring
    - Checkpoint saving and artifact management
    - Resource usage tracking
    
    Workflow:
    1. Load and validate preference dataset (prompt/chosen/rejected format)
    2. Load base model and tokenizer
    3. Setup reference model (copy of base model)
    4. Configure DPO trainer with appropriate settings
    5. Execute training with periodic checkpointing
    6. Save final model and collect artifacts
    """
    
    def __init__(
        self,
        job_uuid: str,              # Unique identifier for this training job
        datasetio_api: DatasetIO,   # API for loading datasets from storage
        datasets_api: Datasets,     # API for dataset operations
    ):
        """
        Initialize the DPO training recipe.
        
        Args:
            job_uuid: Unique identifier for this training job
            datasetio_api: DatasetIO API for loading training data
            datasets_api: Datasets API for dataset operations
        """
        self.job_uuid = job_uuid
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api

    def validate_dataset_format(self, rows: list[dict]) -> bool:
        """
        Validate that the dataset has the required fields for DPO training.
        
        DPO training requires a specific dataset format where each example contains
        exactly three fields: prompt, chosen, and rejected. This method checks that
        all examples in the dataset have these required fields.
        
        DPO Dataset Format Requirements:
        - prompt: The input question or instruction (string)
        - chosen: The preferred/better response (string)  
        - rejected: The less preferred/worse response (string)
        
        Args:
            rows: List of dataset examples (dictionaries)
            
        Returns:
            bool: True if all examples have required DPO fields, False otherwise
        """
        required_fields = ["prompt", "chosen", "rejected"]
        
        # Check if we have any data at all
        if not rows:
            logger.warning("Dataset is empty")
            return False
        
        # Check each row for required fields
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                logger.warning(f"Row {i} is not a dictionary")
                return False
                
            for field in required_fields:
                if field not in row:
                    logger.warning(f"Row {i} missing required DPO field: {field}")
                    return False
                    
                if not isinstance(row[field], str):
                    logger.warning(f"Row {i} field '{field}' is not a string")
                    return False
                    
                if not row[field].strip():
                    logger.warning(f"Row {i} field '{field}' is empty")
                    return False
        
        logger.info(f"Dataset validation passed: {len(rows)} examples with required DPO fields")
        return True

    def _process_preference_format(self, row: dict) -> tuple[str | None, str | None, str | None]:
        """
        Process a single row in preference format for DPO training.
        
        This method extracts the three key components needed for DPO training from
        a single dataset example. It validates that all required fields are present
        and returns them in a standardized format.
        
        Args:
            row: Dictionary containing a single dataset example
            
        Returns:
            tuple: (prompt, chosen, rejected) where each is a string or None if missing
        """
        if "prompt" in row and "chosen" in row and "rejected" in row:
            return row["prompt"], row["chosen"], row["rejected"]
        return None, None, None

    def _format_conversation(self, prompt: str, response: str, provider_config: TrlPostTrainingConfig) -> str:
        """
        Format prompt and response based on model requirements.
        
        This method applies the chat template defined in the provider configuration
        to format the conversation properly. Different models may require different
        formatting (e.g., special tokens, specific structure).
        
        Args:
            prompt: The input prompt/question
            response: The model's response to format
            provider_config: Configuration containing chat template
            
        Returns:
            str: Formatted conversation string ready for training
        """
        if hasattr(provider_config, "chat_template"):
            return provider_config.chat_template.format(prompt=prompt, response=response)
        return f"{prompt}\n{response}"

    def _create_dataset(
        self, rows: list[dict], config: TrainingConfig, provider_config: TrlPostTrainingConfig
    ) -> Dataset:
        """
        Create and preprocess the DPO dataset from raw data.
        
        This method takes raw dataset rows and converts them into a HuggingFace Dataset
        object that's properly formatted for DPO training. It filters out invalid
        examples and ensures all examples have the required fields.
        
        Args:
            rows: Raw dataset examples from the data loader
            config: Training configuration with dataset settings
            provider_config: Provider configuration with formatting settings
            
        Returns:
            Dataset: HuggingFace Dataset object ready for DPO training
            
        Raises:
            ValueError: If no valid examples are found in the dataset
        """
        formatted_rows = []
        for row in rows:
            # Extract prompt, chosen, and rejected responses
            prompt, chosen, rejected = self._process_preference_format(row)

            if prompt and chosen and rejected:
                # DPO requires specific format with prompt, chosen, rejected
                formatted_row = {
                    "prompt": prompt,        # The input question/instruction
                    "chosen": chosen,        # The preferred response
                    "rejected": rejected,    # The less preferred response
                }
                formatted_rows.append(formatted_row)

        if not formatted_rows:
            raise ValueError("No valid prompt/chosen/rejected triplets found in the dataset")

        return Dataset.from_list(formatted_rows)

    def _preprocess_dataset(
        self, ds: Dataset, tokenizer: AutoTokenizer, provider_config: TrlPostTrainingConfig
    ) -> Dataset:
        """
        Preprocess the dataset for DPO training.
        
        Unlike SFT which pre-tokenizes the data, DPO training with TRL's DPOTrainer
        expects the dataset to contain raw text fields (prompt, chosen, rejected)
        that the trainer will tokenize internally during training.
        
        This method applies any necessary text formatting and ensures the dataset
        has the correct structure for DPOTrainer.
        
        Args:
            ds: HuggingFace Dataset with prompt/chosen/rejected fields
            tokenizer: Tokenizer for any needed text processing
            provider_config: Configuration with formatting settings
            
        Returns:
            Dataset: Preprocessed dataset ready for DPOTrainer
        """

        def format_for_dpo(examples):
            """
            Format examples for DPO training.
            
            This function applies chat templates and ensures proper formatting
            of the prompt, chosen, and rejected responses for DPO training.
            """
            formatted_examples = {
                "prompt": [],
                "chosen": [],
                "rejected": []
            }
            
            for i in range(len(examples["prompt"])):
                prompt = examples["prompt"][i]
                chosen = examples["chosen"][i]
                rejected = examples["rejected"][i]
                
                # Apply chat template if available
                if hasattr(provider_config, "chat_template") and provider_config.chat_template:
                    # Format chosen response with chat template
                    chosen_formatted = provider_config.chat_template.format(prompt=prompt, response=chosen)
                    rejected_formatted = provider_config.chat_template.format(prompt=prompt, response=rejected)
                else:
                    # Use simple concatenation if no chat template
                    chosen_formatted = f"{prompt}\n{chosen}"
                    rejected_formatted = f"{prompt}\n{rejected}"
                
                formatted_examples["prompt"].append(prompt)
                formatted_examples["chosen"].append(chosen_formatted)
                formatted_examples["rejected"].append(rejected_formatted)
            
            return formatted_examples

        # Apply formatting to the dataset
        formatted_ds = ds.map(
            format_for_dpo,
            batched=True,        # Process in batches for efficiency
            desc="Formatting dataset for DPO training"
        )
        
        logger.info(f"Preprocessed dataset with {len(formatted_ds)} examples for DPO training")
        return formatted_ds

    async def _setup_data(self, dataset_id: str) -> list[dict[str, Any]]:
        """
        Load dataset from Llama Stack dataset provider.
        
        This method interfaces with Llama Stack's dataset system to load the
        preference dataset that will be used for DPO training.
        
        Args:
            dataset_id: Identifier of the dataset in Llama Stack
            
        Returns:
            list: List of dataset examples as dictionaries
            
        Raises:
            RuntimeError: If dataset loading fails or returns unexpected format
        """
        try:
            # Load all rows from the dataset (limit=-1 means no limit)
            all_rows = await self.datasetio_api.iterrows(
                dataset_id=dataset_id,
                limit=-1,        # Load all available examples
            )
            if not isinstance(all_rows.data, list):
                raise RuntimeError("Expected dataset data to be a list")
            return all_rows.data
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}") from e

    def _run_training_sync(
        self,
        model: str,
        provider_config: dict[str, Any],
        dpo_config: dict[str, Any],
        config: dict[str, Any],
        output_dir_path: Path | None,
    ) -> None:
        """Synchronous wrapper for running DPO training process."""
        import asyncio

        logger.info("Starting DPO training process with async wrapper")
        asyncio.run(
            self._run_training(
                model=model,
                provider_config=provider_config,
                dpo_config=dpo_config,
                config=config,
                output_dir_path=output_dir_path,
            )
        )

    async def load_dataset(
        self,
        model: str,
        config: TrainingConfig,
        provider_config: TrlPostTrainingConfig,
    ) -> tuple[Dataset, Dataset, AutoTokenizer]:
        """
        Load and prepare the complete dataset pipeline for DPO training.
        
        This method orchestrates the entire dataset loading and preparation process:
        1. Loads raw data from Llama Stack's dataset registry
        2. Validates the dataset format for DPO requirements (prompt/chosen/rejected)
        3. Initializes and configures the tokenizer
        4. Processes and tokenizes the dataset for DPO
        5. Splits data into training and validation sets
        
        Args:
            model: HuggingFace model identifier for tokenizer loading
            config: Training configuration containing dataset settings
            provider_config: TRL provider configuration with tokenization settings
            
        Returns:
            tuple: (train_dataset, eval_dataset, tokenizer) where:
                - train_dataset: Processed training data ready for DPO
                - eval_dataset: Processed validation data for monitoring
                - tokenizer: Configured tokenizer used for processing
                
        Raises:
            ValueError: If dataset format is invalid or processing fails
            RuntimeError: If tokenizer initialization fails
        """
        # Validate data config
        if not config.data_config:
            raise ValueError("DataConfig is required for training")

        # Load dataset
        logger.info(f"Loading dataset: {config.data_config.dataset_id}")
        rows = await self._setup_data(config.data_config.dataset_id)
        if not self.validate_dataset_format(rows):
            raise ValueError("Dataset is missing required fields for DPO: prompt, chosen, rejected")
        logger.info(f"Loaded {len(rows)} rows from dataset")

        # Initialize tokenizer
        logger.info(f"Initializing tokenizer for model: {model}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, **provider_config.model_specific_config)

            # Set pad token to eos token if not present
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

            # Set padding side to left for DPO (different from SFT which uses right)
            # Left padding is important for DPO to ensure proper sequence alignment
            tokenizer.padding_side = "left"
            tokenizer.truncation_side = "left"
            tokenizer.model_max_length = provider_config.max_seq_length

            logger.info("Tokenizer initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tokenizer: {str(e)}") from e

        # Create and preprocess dataset for DPO
        logger.info("Creating and preprocessing DPO dataset")
        try:
            ds = self._create_dataset(rows, config, provider_config)
            ds = self._preprocess_dataset(ds, tokenizer, provider_config)
            logger.info(f"Dataset created with {len(ds)} examples")
        except Exception as e:
            raise ValueError(f"Failed to create dataset: {str(e)}") from e

        # Split dataset
        logger.info("Splitting dataset into train and validation sets")
        train_val_split = ds.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_val_split["train"]
        eval_dataset = train_val_split["test"]
        logger.info(f"Split dataset into {len(train_dataset)} training and {len(eval_dataset)} validation examples")

        return train_dataset, eval_dataset, tokenizer

    def load_model(
        self,
        model: str,
        device: torch.device,
        provider_config: TrlPostTrainingConfig,
    ) -> AutoModelForCausalLM:
        """
        Load and initialize the model for DPO training.
        
        This method loads a causal language model with appropriate configurations
        for DPO training. It handles device placement, precision settings, and
        model-specific configurations.
        
        Args:
            model: HuggingFace model identifier or path to local model
            device: PyTorch device where the model should be placed
            provider_config: Configuration with model-specific settings
            
        Returns:
            AutoModelForCausalLM: Loaded and configured model ready for training
            
        Raises:
            RuntimeError: If model loading or device placement fails
        """
        logger.info("Loading the base model for DPO training")
        try:
            # Load model configuration first
            model_config = AutoConfig.from_pretrained(model, **provider_config.model_specific_config)
            
            # Load the actual model with appropriate settings
            model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype="auto" if device.type != "cpu" else "float32",  # Use appropriate precision
                quantization_config=None,                                   # No quantization for training
                config=model_config,                                        # Use loaded config
                **provider_config.model_specific_config,                   # Additional config options
            )
            
            # Always move model to specified device
            model_obj = model_obj.to(device)
            logger.info(f"Model loaded and moved to device: {model_obj.device}")
            return model_obj
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}") from e

    def setup_training_args(
        self,
        config: TrainingConfig,
        provider_config: TrlPostTrainingConfig,
        dpo_config: DPOAlignmentConfig,
        device: torch.device,
        output_dir_path: Path | None,
        steps_per_epoch: int,
    ) -> DPOConfig:
        """
        Setup comprehensive DPO training arguments.
        
        This method creates a complete DPOConfig object with all the necessary
        settings for DPO training. It calculates training steps, configures
        logging and checkpointing, and sets DPO-specific parameters.
        
        Args:
            config: General training configuration from Llama Stack
            provider_config: TRL provider configuration with DPO settings
            dpo_config: DPO algorithm configuration with preference settings
            device: PyTorch device for hardware-specific settings
            output_dir_path: Directory for saving checkpoints and logs
            steps_per_epoch: Number of training steps per epoch
            
        Returns:
            DPOConfig: Complete configuration object for TRL's DPOTrainer
            
        Raises:
            ValueError: If required configuration is missing
        """
        logger.info("Configuring DPO training arguments")
        
        # Set learning rate (DPO typically uses lower learning rates than SFT)
        lr = 1e-6  # Conservative default for DPO
        if config.optimizer_config:
            lr = config.optimizer_config.lr
            logger.info(f"Using custom learning rate: {lr}")

        # Validate data config
        if not config.data_config:
            raise ValueError("DataConfig is required for training")
        data_config = config.data_config

        # Calculate training steps and intervals
        total_steps = steps_per_epoch * config.n_epochs
        max_steps = min(config.max_steps_per_epoch, total_steps) if config.max_steps_per_epoch > 0 else total_steps
        eval_steps = max(1, steps_per_epoch // 10)      # Evaluate 10 times per epoch
        save_steps = max(1, steps_per_epoch // 5)       # Save 5 times per epoch
        logging_steps = max(1, steps_per_epoch // 50)   # Log 50 times per epoch

        # Log training configuration for debugging
        logger.info("DPO training configuration:")
        logger.info(f"- Steps per epoch: {steps_per_epoch}")
        logger.info(f"- Total steps: {total_steps}")
        logger.info(f"- Max steps: {max_steps}")
        logger.info(f"- DPO beta: {provider_config.dpo_beta}")
        logger.info(f"- Learning rate: {lr}")

        # Configure save strategy
        save_strategy = "no"
        if output_dir_path:
            save_strategy = "steps"
            logger.info(f"Will save checkpoints to {output_dir_path}")

        return DPOConfig(
            # === Training Steps and Duration ===
            max_steps=max_steps,                                        # Maximum training steps
            output_dir=str(output_dir_path) if output_dir_path is not None else None,  # Output directory
            num_train_epochs=config.n_epochs,                          # Number of epochs
            
            # === Batch Size and Accumulation ===
            per_device_train_batch_size=data_config.batch_size,        # Batch size per device
            gradient_accumulation_steps=config.gradient_accumulation_steps,  # Gradient accumulation
            
            # === Hardware and Precision Settings ===
            fp16=device.type == "cuda",                                 # Use half precision on GPU
            bf16=False,                                                # Don't use bfloat16 for compatibility
            use_cpu=True if device.type == "cpu" and not torch.backends.mps.is_available() else False,
            
            # === Evaluation and Monitoring ===
            eval_strategy="steps" if eval_steps > 0 else "no",         # Evaluate during training
            eval_steps=eval_steps if eval_steps > 0 else None,         # How often to evaluate
            
            # === Saving and Checkpointing ===
            save_strategy=save_strategy,                               # Save checkpoints during training
            save_steps=save_steps if save_strategy == "steps" else None,  # How often to save
            load_best_model_at_end=True if output_dir_path else False, # Load best model when done
            metric_for_best_model="eval_loss",                        # Metric for best model selection
            greater_is_better=False,                                   # Lower loss is better
            save_total_limit=3,                                        # Keep only 3 latest checkpoints
            
            # === Logging and Reporting ===
            report_to=[],                                              # Don't report to external services
            logging_steps=logging_steps,                               # How often to log
            
            # === Sequence and Memory Settings ===
            max_length=provider_config.max_seq_length,                 # Maximum sequence length
            max_prompt_length=provider_config.max_seq_length // 2,     # Max prompt length (half of total)
            gradient_checkpointing=provider_config.gradient_checkpointing,  # Memory optimization
            remove_unused_columns=False,                               # Keep all dataset columns
            
            # === Optimizer Settings ===
            learning_rate=lr,                                          # Learning rate
            warmup_ratio=0.1,                                          # Learning rate warmup (10%)
            weight_decay=0.01,                                         # L2 regularization
            
            # === Data Loading Settings ===
            dataloader_pin_memory=device.type == "cuda",              # Pin memory for GPU only
            dataloader_num_workers=0,                                 # Single process for compatibility
            
            # === DPO-Specific Parameters ===
            beta=provider_config.dpo_beta,                             # DPO beta parameter (strength)
            loss_type=provider_config.dpo_loss_type,                   # DPO loss function type
            label_smoothing=0.0,                                       # No label smoothing for DPO
        )

    async def _run_training(
        self,
        model: str,
        provider_config: dict[str, Any],
        dpo_config: dict[str, Any],
        config: dict[str, Any],
        output_dir_path: Path | None,
    ) -> None:
        """Run the DPO training process with signal handling."""

        def signal_handler(signum, frame):
            """Handle termination signals gracefully."""
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Convert config dicts back to objects
        logger.info("Initializing configuration objects")
        provider_config_obj = TrlPostTrainingConfig(**provider_config)
        config_obj = TrainingConfig(**config)
        dpo_config_obj = DPOAlignmentConfig(**dpo_config)

        # Initialize and validate device
        device = setup_torch_device(provider_config_obj.device)
        logger.info(f"Using device '{device}'")

        # Load dataset and tokenizer
        train_dataset, eval_dataset, tokenizer = await self.load_dataset(model, config_obj, provider_config_obj)

        # Calculate steps per epoch
        if not config_obj.data_config:
            raise ValueError("DataConfig is required for training")
        steps_per_epoch = len(train_dataset) // config_obj.data_config.batch_size

        # Setup training arguments
        training_args = self.setup_training_args(
            config_obj,
            provider_config_obj,
            dpo_config_obj,
            device,
            output_dir_path,
            steps_per_epoch,
        )

        # Load model
        model_obj = self.load_model(model, device, provider_config_obj)

        # Load reference model for DPO (can be the same as the main model)
        ref_model = None
        if provider_config_obj.use_reference_model:
            logger.info("Loading reference model for DPO")
            ref_model = self.load_model(model, device, provider_config_obj)

        # Initialize DPO trainer
        logger.info("Initializing DPOTrainer")
        logger.info(f"DPOTrainer available parameters: {DPOTrainer.__init__.__code__.co_varnames}")
        
        trainer = DPOTrainer(
            model=model_obj,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,  # Use processing_class instead of tokenizer
        )

        try:
            # Train
            logger.info("Starting DPO training")
            trainer.train()
            logger.info("DPO training completed successfully")

            # Save final model if output directory is provided
            if output_dir_path:
                logger.info("Saving final DPO model")
                save_path = output_dir_path / "dpo_model"
                logger.info(f"Saving model to {save_path}")
                trainer.model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

        finally:
            # Clean up resources
            logger.info("Cleaning up resources")
            if hasattr(trainer, "model"):
                self._clean_up_model(trainer.model, device.type)
            if ref_model:
                self._clean_up_model(ref_model, device.type)
            del trainer
            gc.collect()
            logger.info("Cleanup completed")

    async def train(
        self,
        model: str,                                # Model identifier or path to base model
        output_dir: str | None,                    # Directory to save checkpoints and final model
        job_uuid: str,                            # Job identifier for tracking
        dpo_config: DPOAlignmentConfig,           # DPO-specific algorithm configuration
        config: TrainingConfig,                   # General training configuration
        provider_config: TrlPostTrainingConfig,   # TRL provider configuration
    ) -> tuple[dict[str, Any], list[Checkpoint]]:
        """
        Execute DPO training on a single device.
        
        This is the main entry point for DPO training. It coordinates all aspects
        of the training process from data loading to model saving.
        
        Args:
            model: HuggingFace model identifier or path to local model
            output_dir: Directory where checkpoints and final model will be saved
            job_uuid: Unique identifier for this training job
            dpo_config: DPO algorithm configuration (beta, loss type, etc.)
            config: General training configuration (epochs, batch size, learning rate, etc.)
            provider_config: TRL provider configuration (device, sequence length, etc.)
            
        Returns:
            tuple: (resource_stats, checkpoints) where:
                - resource_stats: Dictionary with memory usage and training metrics
                - checkpoints: List of Checkpoint objects for saved models
        """
        # Initialize and validate device
        device = setup_torch_device(provider_config.device)
        logger.info(f"Using device '{device}'")

        output_dir_path = None
        if output_dir:
            output_dir_path = Path(output_dir)

        # Track memory stats
        memory_stats = {
            "initial": get_memory_stats(device),
            "after_training": None,
            "final": None,
        }

        # Validate data config
        if not config.data_config:
            raise ValueError("DataConfig is required for training")

        # Train in a separate process
        logger.info("Starting DPO training in separate process")
        try:
            # Set multiprocessing start method to 'spawn' for CUDA/MPS compatibility
            if device.type in ["cuda", "mps"]:
                multiprocessing.set_start_method("spawn", force=True)

            process = multiprocessing.Process(
                target=self._run_training_sync,
                kwargs={
                    "model": model,
                    "provider_config": provider_config.model_dump(),
                    "dpo_config": dpo_config.model_dump(),
                    "config": config.model_dump(),
                    "output_dir_path": output_dir_path,
                },
            )
            process.start()

            # Monitor the process
            while process.is_alive():
                process.join(timeout=1)  # Check every second
                if not process.is_alive():
                    break

            # Get the return code
            if process.exitcode != 0:
                raise RuntimeError(f"DPO training failed with exit code {process.exitcode}")

            memory_stats["after_training"] = get_memory_stats(device)

            checkpoints = None
            if output_dir_path:
                # Create checkpoint
                checkpoint = Checkpoint(
                    identifier=f"{model}-dpo-{config.n_epochs}",
                    created_at=datetime.now(timezone.utc),
                    epoch=config.n_epochs,
                    post_training_job_id=job_uuid,
                    path=str(output_dir_path / "dpo_model"),
                )
                checkpoints = [checkpoint]

            return memory_stats, checkpoints
        finally:
            memory_stats["final"] = get_memory_stats(device)
            gc.collect()

    async def _train_in_process(
        self,
        model: str,
        output_dir: str,
        dpo_config: DPOAlignmentConfig,
        config: TrainingConfig,
        provider_config: TrlPostTrainingConfig,
    ) -> tuple[dict[str, Any], list[Checkpoint]]:
        """
        This method is no longer needed - we follow HuggingFace pattern exactly.
        All training logic is in _run_training called via _run_training_sync.
        """
        raise NotImplementedError("Use _run_training via _run_training_sync instead")

    def _load_preference_dataset(self, config: TrainingConfig) -> list[dict]:
        """
        This method is no longer needed - we follow HuggingFace pattern exactly.
        Dataset loading is done in async _run_training method via load_dataset.
        """
        raise NotImplementedError("Use load_dataset in async context instead")

    def _load_model_and_tokenizer(
        self, 
        model_path: str, 
        provider_config: TrlPostTrainingConfig
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        This method is no longer needed - we follow HuggingFace pattern exactly.
        Model loading is done in async _run_training method via load_model.
        """
        raise NotImplementedError("Use load_model in async context instead")

    def _create_dpo_config(
        self,
        output_dir: str,
        config: TrainingConfig,
        provider_config: TrlPostTrainingConfig,
    ) -> DPOConfig:
        """
        This method is no longer needed - we follow HuggingFace pattern exactly.
        Training args setup is done in async _run_training method via setup_training_args.
        """
        raise NotImplementedError("Use setup_training_args in async context instead")

    def _create_dpo_trainer(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer, 
        train_dataset: list[dict],
        training_args: DPOConfig,
        provider_config: TrlPostTrainingConfig,
        dpo_config: DPOAlignmentConfig,
    ) -> DPOTrainer:
        """
        This method is no longer needed - we follow HuggingFace pattern exactly.
        Trainer creation is done in async _run_training method.
        """
        raise NotImplementedError("Use trainer creation in async context instead")

    def _format_dataset_for_dpo(
        self, 
        dataset: list[dict], 
        tokenizer: AutoTokenizer,
        provider_config: TrlPostTrainingConfig
    ) -> list[dict]:
        """
        Format preference dataset for DPOTrainer consumption.
        
        This method converts the raw preference dataset into the format
        expected by TRL's DPOTrainer, applying chat templates and ensuring
        proper field names.
        
        Args:
            dataset: Raw preference dataset (prompt/chosen/rejected)
            tokenizer: Tokenizer for processing text
            provider_config: Provider configuration with chat template
            
        Returns:
            Formatted dataset ready for DPOTrainer
        """
        formatted_dataset = []
        
        for example in dataset:
            # Apply chat template to format the conversations
            prompt = example["prompt"]
            chosen = provider_config.chat_template.format(prompt=prompt, response=example["chosen"])
            rejected = provider_config.chat_template.format(prompt=prompt, response=example["rejected"])
            
            # Create formatted example for DPOTrainer
            formatted_example = {
                "prompt": prompt,         # Original prompt
                "chosen": chosen,         # Formatted chosen response
                "rejected": rejected,     # Formatted rejected response
            }
            
            formatted_dataset.append(formatted_example)
        
        logger.info(f"Formatted {len(formatted_dataset)} examples for DPO training")
        return formatted_dataset

    def _format_example_for_dpo(self, example: dict, provider_config: TrlPostTrainingConfig) -> dict:
        """
        Format a single example for DPO training.
        
        This is the formatting function passed to DPOTrainer that processes
        each training example during training.
        
        Args:
            example: Single training example
            provider_config: Provider configuration
            
        Returns:
            Formatted example dictionary
        """
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"], 
            "rejected": example["rejected"],
        }

    def _collect_resource_stats(self) -> dict[str, Any]:
        """
        Collect resource usage statistics during training.
        
        This method gathers information about memory usage, GPU utilization,
        and other training metrics for monitoring purposes.
        
        Returns:
            Dictionary containing resource usage statistics
        """
        stats = {
            "cpu_count": multiprocessing.cpu_count(),                    # Number of CPU cores
            "process_id": os.getpid(),                      # Training process ID
        }
        
        # Add GPU memory statistics if using CUDA
        if torch.cuda.is_available():
            stats.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(),      # Current GPU memory usage
                "gpu_memory_reserved": torch.cuda.memory_reserved(),        # Reserved GPU memory
                "gpu_device_count": torch.cuda.device_count(),              # Number of GPUs
                "gpu_device_name": torch.cuda.get_device_name(0),           # GPU model name
            })
        
        logger.info(f"Collected resource statistics: {stats}")
        return stats

    def _collect_checkpoints(self, output_dir: str) -> list[Checkpoint]:
        """
        Collect information about saved model checkpoints.
        
        This method scans the output directory for saved checkpoints and
        creates Checkpoint objects that can be tracked by Llama Stack.
        
        Args:
            output_dir: Directory containing saved checkpoints
            
        Returns:
            List of Checkpoint objects representing saved models
        """
        checkpoints = []
        
        try:
            # Look for checkpoint directories in the output directory
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                
                # Check if this looks like a checkpoint directory
                if os.path.isdir(item_path) and ("checkpoint" in item or "final" in item):
                    # Create Checkpoint object
                    checkpoint = Checkpoint(
                        identifier=item,              # Checkpoint name/ID
                        path=item_path,              # Full path to checkpoint
                        metadata={                   # Additional metadata
                            "created_by": "trl_dpo_trainer",
                            "job_uuid": self.job_uuid,
                            "checkpoint_type": "dpo_model",
                        }
                    )
                    checkpoints.append(checkpoint)
                    logger.info(f"Found checkpoint: {item} at {item_path}")
            
            logger.info(f"Collected {len(checkpoints)} checkpoints")
            
        except Exception as e:
            logger.warning(f"Error collecting checkpoints: {e}")
        
        return checkpoints

    def _clean_up_model(self, model: AutoModelForCausalLM, device_type: str) -> None:
        """
        Clean up a model from GPU memory.
        
        This method ensures that the model is properly deallocated from GPU memory
        before the training process ends.
        
        Args:
            model: PyTorch model to be cleaned up
            device_type: Type of the device where the model was loaded
        """
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "mps":
            # MPS cleanup is handled automatically by Python garbage collection
            pass
        elif device_type == "cpu":
            # CPU cleanup is handled automatically by Python garbage collection
            pass
        else:
            raise RuntimeError(f"Unsupported device type: {device_type}") 