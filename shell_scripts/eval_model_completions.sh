#!/bin/bash

# Set visible CUDA devices
export CUDA_VISIBLE_DEVICES=0,7,2,3,4,5,6

# Configuration file and script path
AGENT_MODELS_JSON="/playpen-ssd/smerrill/llm_decisions/configs/models.json"
SCRIPT="/playpen-ssd/smerrill/llm_decisions/generate_completion_responses.py"

# Read agent keys from the JSON file
agents=($(jq -r 'keys[]' "$AGENT_MODELS_JSON"))

# Iterate through each agent
for agent in "${agents[@]}"; do
  model_path=$(jq -r --arg agent "$agent" '.[$agent]' "$AGENT_MODELS_JSON")

  echo "Starting training for agent: $agent"
  echo "Using model: $model_path"

  accelerate launch --num_processes 1 "$SCRIPT" \
    --model_path "$model_path" \
    --speaker "$agent" \
    --output_file "/playpen-ssd/smerrill/llm_decisions/results/${agent}_test_responses.json"
done
