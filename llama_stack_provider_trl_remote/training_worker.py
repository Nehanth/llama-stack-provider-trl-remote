#!/usr/bin/env python3
"""
TRL 8-GPU Training Worker
=========================

Training worker that runs under torchrun for distributed 8-GPU DPO training.
All ranks participate in training together - no hanging issues!

Usage:
  torchrun --nproc_per_node=8 training_worker.py <job_uuid>
"""

import json
import os
import sys
import asyncio
from pathlib import Path

# Import our training recipe
from recipes.dpo_training_unified import DPOTrainingUnified
from config import TrlPostTrainingConfig
from llama_stack.apis.post_training import (
    DPOAlignmentConfig,
    TrainingConfig,
)


async def main():
    """Main training worker function - all ranks execute this"""
    
    if len(sys.argv) != 2:
        print("Usage: training_worker.py <job_uuid>")
        sys.exit(1)
        
    job_uuid = sys.argv[1]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"Rank {local_rank}/{world_size}: Starting training worker for job {job_uuid}")
    
    # Load job parameters (all ranks read the same file)
    job_queue_dir = Path("./job_queue")
    job_file = job_queue_dir / f"{job_uuid}.json"
    
    if not job_file.exists():
        print(f"Rank {local_rank}: Job file {job_file} not found")
        sys.exit(1)
        
    with open(job_file) as f:
        job_params = json.load(f)
    
    print(f"Rank {local_rank}: Loaded job parameters")
    
    try:
        # Extract parameters
        model = job_params.get("model")
        finetuned_model = job_params.get("finetuned_model")
        algorithm_config_dict = job_params.get("algorithm_config", {})
        training_config_dict = job_params.get("training_config", {})
        provider_config_dict = job_params.get("provider_config", {})
        checkpoint_dir = job_params.get("checkpoint_dir", "./dpo_checkpoints")
        dataset_data = job_params.get("dataset_data", [])
        
        print(f"Rank {local_rank}: Training {model} -> {finetuned_model}")
        
        # Create configurations
        config = TrlPostTrainingConfig(**provider_config_dict) if provider_config_dict else TrlPostTrainingConfig()
        
        algorithm_config = DPOAlignmentConfig(
            reward_scale=algorithm_config_dict.get("reward_scale", 1.0),
            reward_clip=algorithm_config_dict.get("reward_clip", 5.0),
            epsilon=algorithm_config_dict.get("epsilon", 0.1),
            gamma=algorithm_config_dict.get("gamma", 0.99)
        )
        
        training_config = TrainingConfig(**training_config_dict)
        
        # Create DPO trainer with dataset data
        print(f"Rank {local_rank}: Initializing DPO trainer for 8-GPU training")
        dpo_trainer = DPOTrainingUnified(
            job_uuid=job_uuid,
            dataset_data=dataset_data
        )
        
        # All ranks participate in training together!
        print(f"Rank {local_rank}: Starting 8-GPU DPO training")
        memory_stats, checkpoints = await dpo_trainer.train(
            model=model,
            output_dir=checkpoint_dir,
            job_uuid=job_uuid,
            dpo_config=algorithm_config,
            config=training_config,
            provider_config=config,
        )
        
        # Only rank 0 writes the status file (avoid conflicts)
        if local_rank == 0:
            status_file = job_queue_dir / f"{job_uuid}_status.json"
            status_data = {
                "status": "completed",
                "checkpoints": [
                    {
                        "identifier": cp.identifier,
                        "created_at": cp.created_at.isoformat(),
                        "epoch": cp.epoch,
                        "post_training_job_id": cp.post_training_job_id,
                        "path": cp.path
                    } for cp in checkpoints
                ] if checkpoints else []
            }
            
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
            print(f"Rank {local_rank}: Training completed successfully!")
        
    except Exception as e:
        print(f"Rank {local_rank}: Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Only rank 0 writes error status
        if local_rank == 0:
            status_file = job_queue_dir / f"{job_uuid}_status.json"
            status_data = {
                "status": "failed",
                "error": str(e)
            }
            
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 