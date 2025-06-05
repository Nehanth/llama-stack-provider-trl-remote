# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
TRL Provider Configuration
==========================

This file defines all the configuration options for the TRL provider.
Configuration controls every aspect of DPO (Direct Preference Optimization) training:
- Where to run training (device)
- How models are saved (checkpoint format)
- Training hyperparameters (learning rates, batch sizes, etc.)
- DPO-specific settings (beta parameter, loss type, etc.)

The configuration is used to create a TrlPostTrainingConfig object that gets
passed to the training logic to control how DPO training works.
"""

from typing import Any, Literal

# Pydantic is used for configuration validation and type checking
# BaseModel provides automatic validation of configuration values
from pydantic import BaseModel


class TrlPostTrainingConfig(BaseModel):
    """
    Configuration class for TRL (Transformer Reinforcement Learning) provider.
    
    This class defines all the settings needed for DPO training, including:
    - Hardware settings (device, distributed training)
    - Model settings (checkpointing, sequence length)
    - Training hyperparameters (learning rate, batch size, etc.)
    - DPO-specific parameters (beta, loss type, reference model)
    
    All parameters have sensible defaults, so you only need to specify
    the ones you want to change from the defaults.
    """
    
    # === HARDWARE AND DEVICE SETTINGS ===
    
    # Device to run training on (cuda, cpu, mps)
    # - "cuda": Use NVIDIA GPU (fastest, requires CUDA)
    # - "cpu": Use CPU only (slower, but works everywhere)
    # - "mps": Use Apple Silicon GPU (M1/M2 Macs)
    device: str = "cuda"

    # Distributed training backend if using multiple devices
    # IMPORTANT: Llama Stack only supports single-node training
    # Multi-node/multi-GPU training is NOT supported
    # - "fsdp": Fully Sharded Data Parallel (NOT SUPPORTED)
    # - "deepspeed": DeepSpeed ZeRO optimization (NOT SUPPORTED)
    # - None: Single device training only (REQUIRED)
    distributed_backend: Literal["fsdp", "deepspeed"] | None = None

    # === MODEL AND CHECKPOINT SETTINGS ===

    # Format for saving model checkpoints
    # - "full_state": Save complete model state (larger files, more compatible)
    # - "huggingface": Save in HuggingFace format (smaller, recommended)
    checkpoint_format: Literal["full_state", "huggingface"] | None = "huggingface"

    # Template for formatting preference inputs
    # This string template is used to format the prompt and response for training
    # {prompt} gets replaced with the actual prompt
    # {response} gets replaced with the chosen or rejected response
    chat_template: str = "<|user|>\n{prompt}\n<|assistant|>\n{response}"

    # Model-specific configuration parameters
    # These get passed directly to HuggingFace's model loading functions
    model_specific_config: dict = {
        "trust_remote_code": True,     # Allow custom model code (needed for some models)
        "attn_implementation": "sdpa", # Use Scaled Dot Product Attention (faster)
    }

    # === TRAINING SEQUENCE AND MEMORY SETTINGS ===

    # Maximum sequence length for training
    # Longer sequences = more context but more memory usage
    # 2048 is a good balance for most use cases
    max_seq_length: int = 2048

    # Enable gradient checkpointing to reduce memory usage
    # Trades computation for memory by recomputing activations during backward pass
    # Set to True if you're running out of GPU memory
    gradient_checkpointing: bool = False

    # === CHECKPOINT AND LOGGING SETTINGS ===

    # Maximum number of checkpoints to keep
    # Older checkpoints are automatically deleted when this limit is reached
    # Set higher if you want to keep more training snapshots
    save_total_limit: int = 3

    # Number of training steps between logging updates
    # Lower values = more frequent logging (but slower training)
    # Higher values = less frequent logging (but faster training)
    logging_steps: int = 10

    # === OPTIMIZER AND LEARNING SETTINGS ===

    # Ratio of training steps used for learning rate warmup
    # During warmup, learning rate gradually increases from 0 to the target
    # Helps stabilize early training, especially important for DPO
    warmup_ratio: float = 0.1

    # L2 regularization coefficient
    # Helps prevent overfitting by penalizing large model weights
    # Higher values = more regularization, lower values = less regularization
    weight_decay: float = 0.01

    # === DATA LOADING SETTINGS (Single-Node Optimized) ===

    # Number of worker processes for data loading
    # For single-node training, 0 is often more stable
    # Higher values can improve data loading speed but use more memory
    # Set to 0 to disable multiprocessing (recommended for single-node)
    dataloader_num_workers: int = 0

    # Whether to pin memory in data loader
    # Can improve data transfer speed to GPU but uses more system memory
    # Generally recommended when using GPU training
    dataloader_pin_memory: bool = True

    # === DPO-SPECIFIC CONFIGURATION PARAMETERS ===
    # These settings are specific to Direct Preference Optimization

    # Beta parameter for DPO loss (controls how much to penalize rejected responses)
    # - Higher values (0.5+): Stronger preference learning, model follows preferences more strictly
    # - Lower values (0.01-0.1): Gentler preference learning, preserves more of original model
    # - Typical range: 0.1 to 0.5
    dpo_beta: float = 0.1

    # Whether to use reference model for DPO (if False, uses the initial model)
    # - True: Use a separate copy of the model as reference (recommended, more stable)
    # - False: Use the initial model state as reference (saves memory but less stable)
    use_reference_model: bool = True

    # Loss type for DPO training
    # - "sigmoid": Standard DPO loss (most common, good default)
    # - "hinge": Hinge loss variant (can be more stable)
    # - "ipo": Identity Preference Optimization variant (experimental)
    dpo_loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid"

    # Whether to normalize rewards in DPO
    # Normalizes the difference between chosen and rejected response rewards
    # Generally recommended for stable training
    normalize_rewards: bool = True

    # Label smoothing for DPO loss
    # Adds small noise to prevent overconfident predictions
    # - 0.0: No smoothing (standard)
    # - 0.1: Light smoothing (can help with overfitting)
    # - Higher values: More smoothing (rarely needed)
    label_smoothing: float = 0.0

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        """
        Provide a sample configuration for testing or examples.
        
        This method returns a basic configuration that should work for most
        testing scenarios. It uses safe defaults that work on most systems.
        
        Args:
            __distro_dir__: Directory for the distribution (not used currently)
            **kwargs: Additional keyword arguments (for future extensibility)
            
        Returns:
            Dictionary containing sample configuration values
        """
        return {
            "checkpoint_format": "huggingface",  # Use HuggingFace format (most compatible)
            "distributed_backend": None,        # Single device only
            "device": "cpu",                    # Use CPU (works everywhere)
            "dpo_beta": 0.1,                   # Standard DPO beta value
            "use_reference_model": True        # Use reference model for stability
        } 