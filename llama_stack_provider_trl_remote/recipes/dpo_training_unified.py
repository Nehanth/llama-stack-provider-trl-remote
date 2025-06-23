"""
DPO Training Recipe for Multi-GPU (FSDP) - FULLY CLEANED UP VERSION
====================================================================

This file implements DPO (Direct Preference Optimization) training using HuggingFace's TRL library
with native FSDP support.

"""

# Standard library imports
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

from config import TrlPostTrainingConfig

logger = logging.getLogger(__name__)


class DPOTrainingUnified:
    """
    UNIFIED DPO training using TRL's native FSDP support.
    
    Automatically handles both single-device and multi-GPU training!
    - WORLD_SIZE=1: Single-device training
    - WORLD_SIZE>1: Multi-GPU FSDP training
    
    Let TRL handle all the complexity automatically!
    """
    
    def __init__(
        self,
        job_uuid: str,
        dataset_data: list[dict] = None,
    ) -> None:
        """Initialize the multi-GPU DPO training recipe."""
        self.job_uuid = job_uuid
        self.dataset_data = dataset_data or []

    def validate_preference_dataset(self, rows: list[dict]) -> bool:
        """Validate DPO dataset format (prompt, chosen, rejected)."""
        required_fields = ["prompt", "chosen", "rejected"]
        
        if not rows:
            logger.warning("Dataset is empty")
            return False
        
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                logger.warning(f"Row {i} is not a dictionary")
                return False
                
            for field in required_fields:
                if field not in row or not isinstance(row[field], str) or not row[field].strip():
                    logger.warning(f"Row {i} missing or invalid field: {field}")
                    return False
        
        logger.info(f"DPO dataset validation passed: {len(rows)} preference examples")
        return True

    def create_dpo_dataset(self, rows: list[dict]) -> Dataset:
        """Create HuggingFace Dataset from preference data."""
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

    async def load_preference_data(self, dataset_id: str) -> list[dict[str, Any]]:
        """Load preference dataset from direct data."""
        if not self.dataset_data:
            raise RuntimeError(f"No dataset data available for {dataset_id}")
        return self.dataset_data

    async def load_dataset(
        self,
        model: str,
        config: TrainingConfig,
        provider_config: TrlPostTrainingConfig,
    ) -> tuple[Dataset, Dataset, AutoTokenizer]:
        """Load and prepare preference dataset for multi-GPU DPO training."""
        if not config.data_config:
            raise ValueError("DataConfig is required for DPO training")

        # Load preference dataset
        logger.info(f"Loading preference dataset: {config.data_config.dataset_id}")
        rows = await self.load_preference_data(config.data_config.dataset_id)
        if not self.validate_preference_dataset(rows):
            raise ValueError("Dataset missing required DPO fields: prompt, chosen, rejected")
        logger.info(f"Loaded {len(rows)} preference examples")

        # Initialize tokenizer
        logger.info(f"Initializing tokenizer for model: {model}")
        tokenizer = AutoTokenizer.from_pretrained(model, **provider_config.model_specific_config)

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        tokenizer.model_max_length = provider_config.max_seq_length

        # Create DPO dataset (HF will handle distribution automatically)
        logger.info("Creating DPO preference dataset")
        ds = self.create_dpo_dataset(rows)
        logger.info(f"DPO dataset created with {len(ds)} examples")

        # Split for training and evaluation
        logger.info("Splitting dataset for multi-GPU DPO training")
        train_val_split = ds.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_val_split["train"]
        eval_dataset = train_val_split["test"]
        
        logger.info(f"Split: {len(train_dataset)} train, {len(eval_dataset)} eval examples")
        return train_dataset, eval_dataset, tokenizer

    def get_transformer_layer_class(self, model_name: str):
        """DEPRECATED: Using min_num_params instead of transformer layer wrapping."""
        # This function is no longer used - we use min_num_params for FSDP wrapping
        # which is simpler and avoids the mutually exclusive parameter conflict
        return None

    def setup_multi_gpu_dpo_config(
        self,
        model: str,
        config: TrainingConfig,
        provider_config: TrlPostTrainingConfig,
        dpo_config: DPOAlignmentConfig,
        output_dir_path: Path | None,
        steps_per_epoch: int,
    ) -> DPOConfig:
        """Setup DPO training configuration with native FSDP support."""
        logger.info("Configuring multi-GPU DPO training with native FSDP")
        
        # DPO learning rate
        lr = 1e-4
        if config.optimizer_config:
            lr = config.optimizer_config.lr

        if not config.data_config:
            raise ValueError("DataConfig is required for DPO training")
        data_config = config.data_config

        # Calculate training steps
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        effective_batch_size = data_config.batch_size * world_size
        # steps_per_epoch is already calculated correctly
        
        # Calculate max steps - use max_steps_per_epoch if configured, otherwise use calculated steps
        if config.max_steps_per_epoch > 0:
            max_steps = config.max_steps_per_epoch * config.n_epochs
        else:
            max_steps = steps_per_epoch * config.n_epochs
        
        # Ensure we always have at least 1 step
        max_steps = max(1, max_steps)
        eval_steps = max(1, steps_per_epoch // 10)
        save_steps = max(1, steps_per_epoch // 5)
        logging_steps = max(1, steps_per_epoch // 50)

        logger.info(f"Multi-GPU FSDP DPO configuration:")
        logger.info(f"- World size: {world_size}")
        logger.info(f"- Effective batch size: {effective_batch_size}")
        logger.info(f"- Steps per epoch: {steps_per_epoch}")
        logger.info(f"- Max steps: {max_steps}")
        logger.info(f"- Learning rate: {lr}")

        save_strategy = "steps" if output_dir_path else "no"

        return DPOConfig(
            # Training steps and duration
            max_steps=max_steps,
            output_dir=str(output_dir_path) if output_dir_path is not None else None,
            num_train_epochs=config.n_epochs,
            
            # Multi-GPU batch settings
            per_device_train_batch_size=data_config.batch_size,
            per_device_eval_batch_size=min(data_config.batch_size, 4),
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            
            fsdp="full_shard",
            fsdp_config={
                # Use min_num_params instead of transformer_layer_cls_to_wrap (they're mutually exclusive)
                "fsdp_min_num_params": 100_000,  # Wrap modules with 100K+ parameters
                "fsdp_offload_params": False,    # Keep on GPU for speed
                "fsdp_use_orig_params": True,    # Better for checkpointing
                "fsdp_cpu_ram_efficient_loading": False,
                "fsdp_sync_module_states": True,
            },
            
            # Mixed precision for FSDP
            bf16=True,  # bfloat16 works best with FSDP
            fp16=False,
            
            # Evaluation and monitoring
            eval_strategy="steps" if eval_steps > 0 else "no",
            eval_steps=eval_steps if eval_steps > 0 else None,
            
            # Checkpointing - FIXED for FSDP compatibility
            save_strategy=save_strategy,
            save_steps=save_steps if save_strategy == "steps" and save_steps is not None else 500,
            load_best_model_at_end=False,  # Disabled for FSDP compatibility
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            save_only_model=False,  # Save full checkpoint for FSDP
            
            # Logging
            report_to=[],
            logging_steps=logging_steps,
            disable_tqdm=False,  # Let rank 0 show progress
            
            # DPO-specific settings
            max_length=provider_config.max_seq_length,
            max_prompt_length=provider_config.max_seq_length // 2,
            gradient_checkpointing=provider_config.gradient_checkpointing,
            
            # Optimizer settings
            learning_rate=lr,
            warmup_ratio=0.1,
            weight_decay=0.01,
            
            # Multi-GPU data loading (HF handles distribution)
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            dataloader_drop_last=True,
            
            # DPO algorithm parameters
            beta=provider_config.dpo_beta,
            loss_type=provider_config.dpo_loss_type,
            label_smoothing=0.0,
            
            # Let HF handle distributed setup
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
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
        Execute multi-GPU DPO training with native FSDP support.
        
        Let TRL and Hugging Face handle all the distributed complexity!
        """
        logger.info("Starting multi-GPU DPO training with native FSDP")

        output_dir_path = None
        if output_dir:
            output_dir_path = Path(output_dir)

        if not config.data_config:
            raise ValueError("DataConfig is required for DPO training")

        try:
            # Load dataset (HF will handle distribution automatically)
            train_dataset, eval_dataset, tokenizer = await self.load_dataset(model, config, provider_config)

            # Calculate steps - ensure we always have at least 1 step!
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            dataset_size = len(train_dataset)
            effective_batch_size = config.data_config.batch_size * world_size
            
            # Calculate steps per epoch, but ensure at least 1 step
            steps_per_epoch = max(1, dataset_size // effective_batch_size)
            
            logger.info(f"Dataset size: {dataset_size}, Effective batch size: {effective_batch_size}")
            logger.info(f"Calculated steps per epoch: {steps_per_epoch}")

            # Setup FSDP configuration
            training_args = self.setup_multi_gpu_dpo_config(
                model,
                config,
                provider_config,
                dpo_config,
                output_dir_path,
                steps_per_epoch,
            )

            # Load model with proper device placement for FSDP
            logger.info("Loading model for multi-GPU DPO training")
            
            # Set device for current rank
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = f"cuda:{local_rank}"
            
            model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map={"": device},  # Load on specific GPU for this rank
                **provider_config.model_specific_config,
            )
            
            # Load reference model
            ref_model = None
            if provider_config.use_reference_model:
                logger.info("Loading reference model for DPO")
                ref_model = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype=torch.bfloat16,
                    **provider_config.model_specific_config,
                )

            # 🚀 Initialize DPO trainer with FSDP
            logger.info("Initializing DPOTrainer with FSDP")
            
            trainer = DPOTrainer(
                model=model_obj,
                ref_model=ref_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )
            logger.info("DPOTrainer initialized successfully")

            # Execute training - TRL handles everything!
            logger.info("Starting multi-GPU DPO training")
            trainer.train()
            logger.info("Multi-GPU DPO training completed successfully")

            # Save model (TRL handles FSDP consolidation)
            checkpoints = []
            if output_dir_path:
                logger.info("Saving final DPO model")
                save_path = output_dir_path / "dpo_model"
                
                # TRL handles FSDP model saving automatically
                trainer.save_model(str(save_path))
                tokenizer.save_pretrained(str(save_path))
                
                logger.info(f"DPO model saved to {save_path}")

                checkpoint = Checkpoint(
                    identifier=f"{model}-dpo-fsdp-{config.n_epochs}",
                    created_at=datetime.now(timezone.utc),
                    epoch=config.n_epochs,
                    post_training_job_id=job_uuid,
                    path=str(save_path),
                )
                checkpoints = [checkpoint]

            # Simple memory stats
            memory_stats = {"status": "completed"}
            return memory_stats, checkpoints

        except Exception as e:
            logger.error(f"Multi-GPU DPO training failed: {str(e)}")
            raise 