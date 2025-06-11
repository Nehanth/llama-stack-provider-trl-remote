"""
DPO Training Recipe for Single Device
=====================================

This file implements DPO (Direct Preference Optimization) training using HuggingFace's TRL library.
This recipe is specifically designed for DPO training and handles the complete pipeline from
dataset loading to model saving.

DPO Training Overview:
DPO trains language models to follow human preferences without requiring a separate reward model.
It works by:
1. Taking a pre-trained base model
2. Training on preference data (prompt, chosen response, rejected response)
3. Optimizing the model to prefer "chosen" responses over "rejected" ones
4. Using a reference model to prevent the model from drifting too far from the original

This implementation supports single-device training (CPU, single GPU).
"""

# Standard library imports
import gc
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# System monitoring
import psutil

# === ENVIRONMENT CONFIGURATION ===
# Optimize for single-device DPO training
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "0"
os.environ["MKL_NUM_THREADS"] = "1"

# === PYTORCH AND ML IMPORTS ===
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DPOConfig, DPOTrainer

# === LLAMA STACK API IMPORTS ===
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    Checkpoint,
    DataConfig,
    DPOAlignmentConfig,
    TrainingConfig,
)

from ..config import TrlPostTrainingConfig

logger = logging.getLogger(__name__)


def get_gb(to_convert: int) -> str:
    """Convert memory values from bytes to gigabytes."""
    return f"{(to_convert / (1024**3)):.2f}"


def get_memory_stats(device: torch.device) -> dict[str, Any]:
    """Get comprehensive memory statistics for monitoring DPO training."""
    stats = {
        "system_memory": {
            "total": get_gb(psutil.virtual_memory().total),
            "available": get_gb(psutil.virtual_memory().available),
            "used": get_gb(psutil.virtual_memory().used),
            "percent": psutil.virtual_memory().percent,
        }
    }

    if device.type == "cuda":
        stats["device_memory"] = {
            "allocated": get_gb(torch.cuda.memory_allocated(device)),
            "reserved": get_gb(torch.cuda.memory_reserved(device)),
            "max_allocated": get_gb(torch.cuda.max_memory_allocated(device)),
        }
    elif device.type == "mps":
        stats["device_memory"] = {
            "note": "MPS memory stats not directly available",
            "system_memory_used": get_gb(psutil.virtual_memory().used),
        }
    elif device.type == "cpu":
        process = psutil.Process()
        stats["device_memory"] = {
            "process_rss": get_gb(process.memory_info().rss),
            "process_vms": get_gb(process.memory_info().vms),
            "process_percent": process.memory_percent(),
        }

    return stats


def setup_torch_device(device_str: str) -> torch.device:
    """Initialize and validate PyTorch device for single-node DPO training."""
    # Ensure single-node training only (Llama Stack requirement for now)
    if "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "Multi-node training detected via WORLD_SIZE environment variable. "
            "Llama Stack only supports single-node training."
        )
    
    if "LOCAL_WORLD_SIZE" in os.environ and int(os.environ.get("LOCAL_WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "Multi-GPU training detected via LOCAL_WORLD_SIZE environment variable. "
            "This recipe is designed for single-device training only."
        )

    try:
        device = torch.device(device_str)
    except RuntimeError as e:
        raise RuntimeError(f"Error getting Torch Device {str(e)}") from e

    # Force single device selection
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{device.type}: Torch has no CUDA/ROCm support or could not detect a compatible device."
            )
        # Always use device 0 for single-node training
        device = torch.device("cuda", 0)
        logger.info(f"Using single CUDA device: {device}")
            
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(f"{device.type}: Torch has no MPS support or could not detect a compatible device.")
        logger.info("Using single MPS device for training")
            
    elif device.type == "cpu":
        logger.info("Using CPU for single-node training")
        
    elif device.type == "hpu":
        raise RuntimeError(f"{device.type}: training does not support Intel Gaudi.")

    return device


class DPOTrainingSingleDevice:
    """
    Single-device DPO training implementation using TRL.
    
    *** IMPORTANT: This recipe is designed exclusively for single-node training ***
    *** as required by Llama Stack. Multi-node/multi-GPU training is NOT supported ***
    
    This class implements the complete DPO training pipeline:
    - Loads preference datasets (prompt/chosen/rejected format)
    - Configures model and tokenizer for DPO
    - Sets up reference model for preference optimization
    - Executes DPO training with TRL's DPOTrainer
    - Saves trained models and tracks checkpoints
    
    Supported single-node configurations:
    - Single CPU training
    - Single CUDA GPU training
    - Single MPS device training (Apple Silicon)
    """
    
    def __init__(
        self,
        job_uuid: str,
        datasetio_api: DatasetIO,  # Required API for loading datasets from storage
        datasets_api: Datasets,   # Required API for dataset operations
    ) -> None:
        """
        Initialize the DPO training recipe.
        
        Args:
            job_uuid: Unique identifier for the training job
            datasetio_api: DatasetIO API for loading datasets from storage (required)
            datasets_api: Datasets API for dataset operations (required)
        """
        self.job_uuid = job_uuid
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api

    def validate_preference_dataset(self, rows: list[dict]) -> bool:
        """
        Validate that the dataset has the required fields for DPO training.
        
        DPO requires exactly three fields per example:
        - prompt: The input question or instruction
        - chosen: The preferred response
        - rejected: The less preferred response
        """
        required_fields = ["prompt", "chosen", "rejected"]
        
        if not rows:
            logger.warning("Dataset is empty")
            return False
        
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
        
        logger.info(f"DPO dataset validation passed: {len(rows)} preference examples")
        return True

    def create_dpo_dataset(
        self, rows: list[dict], config: TrainingConfig, provider_config: TrlPostTrainingConfig
    ) -> Dataset:
        """
        Create HuggingFace Dataset from preference data for DPO training.
        
        Following the TRL DPO pattern, the dataset should contain raw text fields
        that DPOTrainer will tokenize internally during training.
        """
        dpo_examples = []
        for row in rows:
            if all(field in row for field in ["prompt", "chosen", "rejected"]):
                dpo_examples.append({
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                })

        if not dpo_examples:
            raise ValueError("No valid preference examples found in dataset")

        logger.info(f"Created DPO dataset with {len(dpo_examples)} preference pairs")
        return Dataset.from_list(dpo_examples)

    def preprocess_dpo_dataset(
        self, ds: Dataset, tokenizer: AutoTokenizer, provider_config: TrlPostTrainingConfig
    ) -> Dataset:
        """
        Preprocess dataset for DPO training.
        
        DPOTrainer expects raw text that it will tokenize internally, so we only
        apply basic formatting here.
        """
        def format_for_dpo(examples):
            """Apply chat template formatting if available."""
            formatted = {
                "prompt": [],
                "chosen": [],
                "rejected": []
            }
            
            for i in range(len(examples["prompt"])):
                prompt = examples["prompt"][i]
                chosen = examples["chosen"][i]
                rejected = examples["rejected"][i]
                
                # Apply chat template if configured
                if hasattr(provider_config, "chat_template") and provider_config.chat_template:
                    chosen_formatted = provider_config.chat_template.format(prompt=prompt, response=chosen)
                    rejected_formatted = provider_config.chat_template.format(prompt=prompt, response=rejected)
                else:
                    chosen_formatted = f"{prompt}\n{chosen}"
                    rejected_formatted = f"{prompt}\n{rejected}"
                
                formatted["prompt"].append(prompt)
                formatted["chosen"].append(chosen_formatted)
                formatted["rejected"].append(rejected_formatted)
            
            return formatted

        formatted_ds = ds.map(
            format_for_dpo,
            batched=True,
            desc="Formatting dataset for DPO training"
        )
        
        logger.info(f"Preprocessed {len(formatted_ds)} examples for DPO training")
        return formatted_ds

    async def load_preference_data(self, dataset_id: str) -> list[dict[str, Any]]:
        """Load preference dataset from Llama Stack dataset provider."""
        try:
            all_rows = await self.datasetio_api.iterrows(dataset_id=dataset_id, limit=-1)
            if not isinstance(all_rows.data, list):
                raise RuntimeError("Expected dataset data to be a list")
            return all_rows.data
        except Exception as e:
            raise RuntimeError(f"Failed to load preference dataset: {str(e)}") from e

    async def load_dataset(
        self,
        model: str,
        config: TrainingConfig,
        provider_config: TrlPostTrainingConfig,
    ) -> tuple[Dataset, Dataset, AutoTokenizer]:
        """
        Load and prepare preference dataset for DPO training.
        
        Following TRL DPO patterns:
        1. Load preference data (prompt/chosen/rejected)
        2. Initialize tokenizer with proper settings for DPO
        3. Format dataset for DPOTrainer consumption
        4. Split into train/eval sets
        """
        if not config.data_config:
            raise ValueError("DataConfig is required for DPO training")

        # Load preference dataset
        logger.info(f"Loading preference dataset: {config.data_config.dataset_id}")
        rows = await self.load_preference_data(config.data_config.dataset_id)
        if not self.validate_preference_dataset(rows):
            raise ValueError("Dataset missing required DPO fields: prompt, chosen, rejected")
        logger.info(f"Loaded {len(rows)} preference examples")

        # Initialize tokenizer for DPO
        logger.info(f"Initializing tokenizer for model: {model}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, **provider_config.model_specific_config)

            # Configure tokenizer for DPO training (CRITICAL FIX)
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

            # RIGHT padding for DPO training (not left!)
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = "right"
            tokenizer.model_max_length = provider_config.max_seq_length

            logger.info("Tokenizer configured for DPO training with RIGHT padding")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tokenizer: {str(e)}") from e

        # Create DPO dataset
        logger.info("Creating DPO preference dataset")
        try:
            ds = self.create_dpo_dataset(rows, config, provider_config)
            ds = self.preprocess_dpo_dataset(ds, tokenizer, provider_config)
            logger.info(f"DPO dataset created with {len(ds)} examples")
        except Exception as e:
            raise ValueError(f"Failed to create DPO dataset: {str(e)}") from e

        # Split for training and evaluation
        logger.info("Splitting dataset for DPO training")
        train_val_split = ds.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_val_split["train"]
        eval_dataset = train_val_split["test"]
        logger.info(f"Split: {len(train_dataset)} train, {len(eval_dataset)} eval examples")

        return train_dataset, eval_dataset, tokenizer

    def load_model(
        self,
        model: str,
        device: torch.device,
        provider_config: TrlPostTrainingConfig,
    ) -> AutoModelForCausalLM:
        """Load model for DPO training."""
        logger.info("Loading model for DPO training")
        try:
            model_config = AutoConfig.from_pretrained(model, **provider_config.model_specific_config)
            
            model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype="auto" if device.type != "cpu" else "float32",
                quantization_config=None,  # No quantization for DPO training
                config=model_config,
                **provider_config.model_specific_config,
            )
            
            model_obj = model_obj.to(device)
            logger.info(f"Model loaded and moved to device: {model_obj.device}")
            return model_obj
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}") from e

    def setup_dpo_config(
        self,
        config: TrainingConfig,
        provider_config: TrlPostTrainingConfig,
        dpo_config: DPOAlignmentConfig,
        device: torch.device,
        output_dir_path: Path | None,
        steps_per_epoch: int,
    ) -> DPOConfig:
        """Setup DPO training configuration for single-node training."""
        logger.info("Configuring DPO training arguments for single-node setup")
        
        # Enforce single-node training
        if provider_config.distributed_backend is not None:
            raise ValueError(
                f"Distributed training backend '{provider_config.distributed_backend}' is not supported. "
                "Llama Stack only supports single-node training."
            )
        
        # DPO typically uses lower learning rates than standard fine-tuning
        lr = 1e-4
        if config.optimizer_config:
            lr = config.optimizer_config.lr
            logger.info(f"Using custom learning rate: {lr}")

        if not config.data_config:
            raise ValueError("DataConfig is required for DPO training")
        data_config = config.data_config

        # Calculate training steps
        total_steps = steps_per_epoch * config.n_epochs
        max_steps = min(config.max_steps_per_epoch, total_steps) if config.max_steps_per_epoch > 0 else total_steps
        eval_steps = max(1, steps_per_epoch // 10)
        save_steps = max(1, steps_per_epoch // 5)
        logging_steps = max(1, steps_per_epoch // 50)

        logger.info("Single-node DPO training configuration:")
        logger.info(f"- Device: {device}")
        logger.info(f"- Steps per epoch: {steps_per_epoch}")
        logger.info(f"- Total steps: {total_steps}")
        logger.info(f"- Max steps: {max_steps}")
        logger.info(f"- DPO beta: {provider_config.dpo_beta}")
        logger.info(f"- Learning rate: {lr}")

        save_strategy = "steps" if output_dir_path else "no"

        return DPOConfig(
            # Training steps and duration
            max_steps=max_steps,
            output_dir=str(output_dir_path) if output_dir_path is not None else None,
            num_train_epochs=config.n_epochs,
            
            # Single-device batch settings
            per_device_train_batch_size=data_config.batch_size,
            per_device_eval_batch_size=min(data_config.batch_size, 4),  # Smaller eval batch
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            
            # Single-node hardware settings
            fp16=device.type == "cuda",
            bf16=False,
            use_cpu=device.type == "cpu",
            no_cuda=device.type != "cuda",  # Explicit CUDA control
            
            # Evaluation and monitoring
            eval_strategy="steps" if eval_steps > 0 else "no",
            eval_steps=eval_steps if eval_steps > 0 else None,
            
            # Checkpointing
            save_strategy=save_strategy,
            save_steps=save_steps if save_strategy == "steps" else None,
            load_best_model_at_end=True if output_dir_path else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            
            # Logging (no external reporting for single-node)
            report_to=[],
            logging_steps=logging_steps,
            
            # DPO-specific sequence settings
            max_length=provider_config.max_seq_length,
            max_prompt_length=provider_config.max_seq_length // 2,
            gradient_checkpointing=provider_config.gradient_checkpointing,
            remove_unused_columns=False,
            
            # Optimizer settings
            learning_rate=lr,
            warmup_ratio=0.1,
            weight_decay=0.01,
            
            # Single-node data loading settings
            dataloader_pin_memory=device.type == "cuda",
            dataloader_num_workers=0,  # Single process for single-node
            dataloader_drop_last=False,  # Don't drop last batch for small datasets
            
            # Single-node distributed settings (disabled)
            ddp_find_unused_parameters=False,
            ddp_backend=None,
            local_rank=-1,  # No distributed training
            
            # DPO algorithm parameters
            beta=provider_config.dpo_beta,
            loss_type=provider_config.dpo_loss_type,
            label_smoothing=0.0,
        )

    async def train(
        self,
        model: str,
        output_dir: str | None,
        job_uuid: str,
        dpo_config: DPOAlignmentConfig,
        config: TrainingConfig,
        provider_config: TrlPostTrainingConfig,
    ) -> tuple[dict[str, Any], list[Checkpoint]]:
        """
        Execute single-node DPO training.
        
        This method enforces single-node training as required by Llama Stack
        and executes the complete DPO training pipeline.
        """
        # Enforce single-node training
        logger.info("Starting single-node DPO training (Llama Stack requirement)")
        
        device = setup_torch_device(provider_config.device)
        logger.info(f"Initialized single device: {device}")

        output_dir_path = None
        if output_dir:
            output_dir_path = Path(output_dir)

        # Track memory usage
        memory_stats = {
            "initial": get_memory_stats(device),
            "after_training": None,
            "final": None,
        }

        if not config.data_config:
            raise ValueError("DataConfig is required for DPO training")

        try:
            # Load preference dataset and tokenizer
            train_dataset, eval_dataset, tokenizer = await self.load_dataset(model, config, provider_config)

            # Calculate steps
            steps_per_epoch = len(train_dataset) // config.data_config.batch_size

            # Setup single-node DPO configuration
            training_args = self.setup_dpo_config(
                config,
                provider_config,
                dpo_config,
                device,
                output_dir_path,
                steps_per_epoch,
            )

            # Load model and reference model
            model_obj = self.load_model(model, device, provider_config)
            ref_model = None
            if provider_config.use_reference_model:
                logger.info("Loading reference model for DPO")
                ref_model = self.load_model(model, device, provider_config)

            # Initialize DPO trainer for single-node training
            logger.info("Initializing DPOTrainer for single-node setup")
            trainer = DPOTrainer(
                model=model_obj,
                ref_model=ref_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )

            # Execute DPO training
            logger.info("Starting single-node DPO training")
            trainer.train()
            logger.info("Single-node DPO training completed successfully")

            memory_stats["after_training"] = get_memory_stats(device)

            # Save final DPO model
            checkpoints = None
            if output_dir_path:
                logger.info("Saving final DPO model")
                save_path = output_dir_path / "dpo_model"
                trainer.model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logger.info(f"DPO model saved to {save_path}")

                checkpoint = Checkpoint(
                    identifier=f"{model}-dpo-{config.n_epochs}",
                    created_at=datetime.now(timezone.utc),
                    epoch=config.n_epochs,
                    post_training_job_id=job_uuid,
                    path=str(save_path),
                )
                checkpoints = [checkpoint]
                logger.info(f"Created checkpoint: {checkpoint.identifier}")

            return memory_stats, checkpoints

        except Exception as e:
            logger.error(f"Single-node DPO training failed: {str(e)}")
            raise
        finally:
            # Cleanup
            logger.info("Cleaning up single-node DPO training resources")
            if 'trainer' in locals() and hasattr(trainer, "model"):
                self._cleanup_model(trainer.model, device.type)
            if 'ref_model' in locals() and ref_model:
                self._cleanup_model(ref_model, device.type)
            if 'trainer' in locals():
                del trainer
            if 'model_obj' in locals():
                del model_obj
            if 'ref_model' in locals():
                del ref_model
            gc.collect()
            memory_stats["final"] = get_memory_stats(device)

    def _cleanup_model(self, model: AutoModelForCausalLM, device_type: str) -> None:
        """Clean up model from device memory."""
        if device_type == "cuda":
            torch.cuda.empty_cache()