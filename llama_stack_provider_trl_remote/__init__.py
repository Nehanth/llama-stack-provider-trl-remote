"""
TRL Remote Provider Entry Point
===============================

This file serves as the entry point for the TRL Remote Provider adapter.
When Llama Stack needs to create an instance of the remote TRL provider, it calls the function
in this file to get a properly configured adapter instance.
"""

from typing import Any
from llama_stack.distribution.datatypes import Api
from .config import TrlRemoteConfig


async def get_adapter_impl(
    config: TrlRemoteConfig,
    deps: dict[Api, Any],
):
    """
    Create and configure a TRL remote adapter instance.
    
    This is the main entry point that Llama Stack calls when it needs to create
    a remote TRL provider adapter. It's called during provider initialization.
    
    Args:
        config: TrlRemoteConfig containing remote service connection settings like:
                - base_url: URL of the remote TRL training service
                - timeout: Request timeout for training operations
                - retry_config: Retry settings for failed requests
        
        deps: Dictionary of API dependencies (may be empty for remote providers
              since dependencies are handled by the remote service)
              
    Returns:
        TrlRemoteAdapter: A configured adapter instance ready to communicate
                         with the remote TRL training service
    
    """
    
    from .adapter import TrlRemoteAdapter
    
    # Create an instance of our remote adapter with the configuration
    adapter = TrlRemoteAdapter(config)
    
    # Initialize the adapter (establish connection, validate service, etc.)
    await adapter.initialize()
    
    return adapter 