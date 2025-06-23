"""
TRL Training Recipes
===================

This package contains DPO (Direct Preference Optimization) training recipe:

- dpo_training_unified: Unified recipe that handles both single-device and multi-GPU training with FSDP

The recipe automatically detects the training environment and uses appropriate settings.
"""

from .dpo_training_unified import DPOTrainingUnified

__all__ = [
    "DPOTrainingUnified"
] 