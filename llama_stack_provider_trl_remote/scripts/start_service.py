#!/usr/bin/env python3
"""
Start TRL Remote Training Service
=================================

This script starts the remote TRL training service that exposes
the same DPO training functionality as the inline provider but
runs as a standalone HTTP service.

Usage:
    python start_service.py [--host HOST] [--port PORT] [--config CONFIG_FILE]

The service will:
1. Load TRL training configuration
2. Initialize the TRL provider using existing components
3. Start FastAPI server with training endpoints
4. Provide health checks and service management
"""

import argparse
import logging
import sys
from pathlib import Path

import uvicorn


def setup_logging(level: str = "INFO"):
    """Configure logging for the service."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("trl_service.log")
        ]
    )


def main():
    """Main entry point for the TRL remote service."""
    parser = argparse.ArgumentParser(description="Start TRL Remote Training Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting TRL Remote Training Service on {args.host}:{args.port}")
    
    if args.config:
        logger.info(f"Using configuration file: {args.config}")
        # TODO: Load configuration from file
    
    try:
        # Import and run the FastAPI service
        uvicorn.run(
            "llama_stack_provider_trl_remote.service:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level.lower(),
            access_log=True,
        )
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 