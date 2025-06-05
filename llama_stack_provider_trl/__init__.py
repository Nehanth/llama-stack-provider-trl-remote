# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
TRL Provider Entry Point
========================

This file serves as the entry point for the TRL (Transformer Reinforcement Learning) provider.
When Llama Stack needs to create an instance of the TRL provider, it calls the function
in this file to get a properly configured provider instance.

What happens here:
1. Llama Stack calls get_provider_impl() with configuration and dependencies
2. We import our main implementation class (TrlPostTrainingImpl)
3. We create an instance with the provided config and API dependencies
4. We return the ready-to-use provider instance

This follows the standard Llama Stack provider pattern used by all inline providers.
"""

from typing import Any

# Import the Api enum which defines all the different APIs in Llama Stack
# (like datasetio, datasets, post_training, inference, etc.)
from llama_stack.distribution.datatypes import Api

# Import our configuration class that defines all the settings for DPO training
from .config import TrlPostTrainingConfig

# Comment explaining what this provider does
# TRL = Transformer Reinforcement Learning, used for preference-based training like DPO


async def get_provider_impl(
    config: TrlPostTrainingConfig,  # Our TRL-specific configuration (device, DPO settings, etc.)
    deps: dict[Api, Any],          # Dependencies from Llama Stack (datasets, datasetio APIs)
):
    """
    Create and configure a TRL provider instance.
    
    This is the main entry point that Llama Stack calls when it needs to create
    a TRL provider. It's called during provider initialization.
    
    Args:
        config: TrlPostTrainingConfig containing all DPO training settings like:
                - device: Where to run training (cuda/cpu/mps)
                - dpo_beta: How strongly to prefer chosen over rejected responses
                - use_reference_model: Whether to use a separate reference model
                - Training hyperparameters (learning rate, batch size, etc.)
        
        deps: Dictionary of API dependencies that our provider needs:
              - Api.datasetio: For loading and managing training datasets
              - Api.datasets: For dataset operations and transformations
              
    Returns:
        TrlPostTrainingImpl: A fully configured provider instance ready to perform
                            DPO training jobs
    
    Note:
        This function is async even though it doesn't await anything currently,
        because provider initialization might need async operations in the future.
    """
    # Import our main implementation class here (not at top level) to avoid
    # circular import issues and to only import when actually needed
    from .post_training import TrlPostTrainingImpl

    # Create an instance of our TRL provider with:
    # - The configuration settings (device, DPO parameters, etc.)
    # - The datasetio API for loading training data
    # - The datasets API for dataset operations
    impl = TrlPostTrainingImpl(
        config,                    # TRL-specific configuration
        deps[Api.datasetio],      # API for loading datasets from storage
        deps[Api.datasets],       # API for dataset operations
    )
    
    # Return the configured provider instance
    # Llama Stack will use this instance to handle DPO training requests
    return impl 