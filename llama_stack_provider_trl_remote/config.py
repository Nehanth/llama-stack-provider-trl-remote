"""
TRL Remote Provider Configuration
=================================

This file defines the configuration for the TRL Remote Provider.
It reuses the existing TrlPostTrainingConfig and adds remote-specific settings.

The remote provider needs:
1. All the same DPO training configuration as the inline provider
2. Additional remote service connection settings (URL, timeouts, etc.)
"""

from typing import Any
from pydantic import BaseModel

# Reuse the existing TRL configuration
from llama_stack_provider_trl.config import TrlPostTrainingConfig


class TrlRemoteConfig(BaseModel):
    """
    Configuration for TRL Remote Provider adapter.
    
    This config combines:
    1. Remote service connection settings
    2. All existing TRL training configuration (reused)
    
    The adapter will forward training requests to a remote TRL service
    that uses the same training logic as the inline provider.
    """
    
    # === REMOTE SERVICE CONNECTION SETTINGS ===
    
    # Base URL of the remote TRL training service
    # Example: "http://trl-training-service:8080"
    base_url: str = "http://localhost:8080"
    
    # Request timeout in seconds for training operations
    # Training jobs can take a long time, so we need generous timeouts
    timeout: int = 3600  # 1 hour default
    
    # Connection timeout for initial requests
    connect_timeout: int = 30
    
    # Maximum number of retry attempts for failed requests
    max_retries: int = 3
    
    # Delay between retry attempts (seconds)
    retry_delay: int = 5
    
    # === REUSED TRL TRAINING CONFIGURATION ===
    
    # Reuse all the existing TRL training configuration
    # This ensures the remote service gets exactly the same training parameters
    # as the inline provider would use
    training_config: TrlPostTrainingConfig = TrlPostTrainingConfig()
    
    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        """
        Provide a sample configuration for remote TRL provider.
        
        This reuses the inline provider's sample config and adds remote settings.
        """
        # Get the sample config from the inline provider
        inline_sample = TrlPostTrainingConfig.sample_run_config(__distro_dir__, **kwargs)
        
        return {
            "base_url": "http://localhost:8080",
            "timeout": 3600,
            "training_config": inline_sample
        } 