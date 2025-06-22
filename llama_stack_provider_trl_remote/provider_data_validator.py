"""
TRL Remote Provider Data Validator
==================================

This validator extends the default Llama Stack validation to allow DPO algorithm
configs in addition to the standard LoRA/QAT configs. This enables the remote
TRL provider to accept the same DPO format as used in examples.ipynb.

The validator allows:
1. Standard LoRA and QAT algorithm configs (for compatibility)
2. DPO algorithm configs (type: "dpo") for direct preference optimization
3. Transforms incoming requests to ensure proper validation
"""

from typing import Any, Dict
from pydantic import BaseModel, Field
from llama_stack.apis.post_training import (
    LoraFinetuningConfig,
    QATFinetuningConfig,
)


class DPOAlgorithmConfig(BaseModel):
    """
    DPO Algorithm Configuration for preference optimization.
    
    This extends the standard algorithm configs to include DPO-specific
    parameters as used in examples.ipynb.
    """
    type: str = Field(default="dpo", description="Algorithm type - must be 'dpo'")
    reward_scale: float = Field(default=1.0, description="DPO reward scaling factor")
    reward_clip: float = Field(default=5.0, description="DPO reward clipping value")
    epsilon: float = Field(default=0.1, description="DPO epsilon parameter")
    gamma: float = Field(default=0.99, description="DPO gamma discount factor")


class TrlRemoteDataValidator:
    """
    Data validator for TRL Remote Provider that supports DPO algorithm configs.
    
    This validator allows the remote provider to accept DPO format directly
    from the client, matching the format used in examples.ipynb.
    """
    
    @staticmethod
    def validate_algorithm_config(algorithm_config: Any) -> Any:
        """
        Validate algorithm config, allowing DPO in addition to LoRA/QAT.
        
        Args:
            algorithm_config: The algorithm configuration to validate
            
        Returns:
            Validated algorithm configuration
            
        Raises:
            ValueError: If the algorithm config is invalid
        """
        if algorithm_config is None:
            return None
            
        # Handle dict-based configs
        if isinstance(algorithm_config, dict):
            config_type = algorithm_config.get("type")
            
            if config_type == "dpo":
                # Validate DPO config
                return DPOAlgorithmConfig(**algorithm_config)
            elif config_type == "LoRA":
                # Validate LoRA config
                return LoraFinetuningConfig(**algorithm_config)
            elif config_type == "QAT":
                # Validate QAT config
                return QATFinetuningConfig(**algorithm_config)
            else:
                raise ValueError(f"Unsupported algorithm type: {config_type}")
        
        # Handle already-validated objects
        if hasattr(algorithm_config, 'type'):
            if algorithm_config.type in ["dpo", "LoRA", "QAT"]:
                return algorithm_config
                
        raise ValueError("Invalid algorithm configuration format")
    
    @staticmethod
    def validate_preference_optimize_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate preference optimization request, supporting DPO algorithm configs.
        
        Args:
            request_data: The request data to validate
            
        Returns:
            Validated request data with proper algorithm config
        """
        if "algorithm_config" in request_data:
            request_data["algorithm_config"] = TrlRemoteDataValidator.validate_algorithm_config(
                request_data["algorithm_config"]
            )
        
        return request_data 