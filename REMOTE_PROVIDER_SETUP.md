# TRL Remote Provider Setup Guide

## Overview

This guide documents how to run the TRL (Transformer Reinforcement Learning) provider in remote mode, where the training service runs separately from the Llama Stack client.

## Architecture

```
┌─────────────────────┐    HTTP     ┌──────────────────────┐
│ Llama Stack Client  │◄────────────┤ Remote TRL Service   │
│ (Port 8321)         │             │ (Port 8080)          │
│ TrlRemoteAdapter    │             │ TrlPostTrainingImpl  │
└─────────────────────┘             └──────────────────────┘
```

## Setup Steps

### 1. Build the Remote Provider

```bash
llama stack build --config build_remote.yaml
```

This creates the `trl-remote-client` virtual environment and installs dependencies.

### 2. Start the Remote TRL Service

```bash
cd llama_stack_provider_trl_remote
python service.py
```

**Expected Output:**
```
Starting TRL Remote Training Service...
Service initialized successfully
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### 3. Start the Llama Stack Client (Remote Mode)

```bash
python -m llama_stack.distribution.server.server --yaml-config run_remote.yaml
```

**Expected Output:**
```
INFO: Loaded remote provider spec for remote::trl_remote
INFO: Listening on ['::', '0.0.0.0']:8321
INFO: Uvicorn running on http://['::', '0.0.0.0']:8321 (Press CTRL+C to quit)
```

### 4. Verify Both Services Are Running

```bash
# Check client health
curl http://localhost:8321/health

# Check remote service health  
curl http://localhost:8080/health
```

## Configuration Files

### Remote Provider Configuration (`run_remote.yaml`)

Key settings:
- **Client Port**: 8321
- **Remote Service URL**: http://localhost:8080
- **Provider Type**: `remote::trl_remote`
- **DPO Training Config**: device=cuda, dpo_beta=0.1, etc.

### Remote Service Configuration (`providers.d/remote/post_training/trl_remote.yaml`)

Defines the remote provider specification that the client loads.

## Usage

With both services running, you can use the remote provider exactly like the inline version:

1. **Upload Dataset**: Use the dataset API on port 8321
2. **Start Training**: Call the post_training API on port 8321
3. **Monitor Progress**: The client forwards requests to the remote service on port 8080

## Key Differences from Inline Provider

| Aspect | Inline Provider | Remote Provider |
|--------|----------------|-----------------|
| **Deployment** | Single process | Two processes |
| **Communication** | Direct function calls | HTTP requests |
| **Scaling** | Limited to single machine | Can run on separate machines |
| **Training Logic** | Same DPO recipes | Same DPO recipes (100% reuse) |

## Troubleshooting

### Service Won't Start

1. **Check ports**: Ensure 8080 and 8321 are available
2. **Dependencies**: Run in the main conda environment with all packages
3. **Provider Loading**: Check that `providers.d/remote/post_training/trl_remote.yaml` exists

### Training Fails

1. **Dataset Access**: The remote service gets dataset data directly from the client
2. **Validation**: LoRA-to-DPO transformation handles client-side validation
3. **Logs**: Check both client (8321) and service (8080) logs

## Commands Summary

```bash
# Terminal 1: Start Remote Service
cd llama_stack_provider_trl_remote && python service.py

# Terminal 2: Start Client  
python -m llama_stack.distribution.server.server --yaml-config run_remote.yaml

# Verify Setup
curl http://localhost:8321/health && curl http://localhost:8080/health
```

## Success Indicators

✅ **Remote Service Running**: `{"status":"healthy","service":"trl-training-service"}`  
✅ **Client Running**: Logs show "Loaded remote provider spec for remote::trl_remote"  
✅ **Communication**: Client can reach remote service at http://localhost:8080  
✅ **Provider Loaded**: `remote::trl_remote` appears in client configuration  

The remote provider setup is now ready for DPO training with the same capabilities as the inline version! 