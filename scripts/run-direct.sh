#!/bin/sh

. .venv/bin/activate
llama stack run --image-type venv --image-name trl-post-training run.yaml