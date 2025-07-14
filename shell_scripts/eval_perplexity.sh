#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,7,2,3,4,5,6

MODELS_JSON="/playpen-ssd/smerrill/llm_decisions/configs/models.json"
SCRIPT="/playpen-ssd/smerrill/llm_decisions/evaluate_agent_perplexity.py"

# Read agents and model paths from JSON into bash arrays
mapfile -t AGENTS < <(jq -r 'keys[]' "$MODELS_JSON")

for agent in "${AGENTS[@]}"; do
  model_path=$(jq -r --arg agent "$agent" '.[$agent]' "$MODELS_JSON")

  # Get the parent directory (one above "merged")
  PARENT_DIR=$(dirname "$model_path")
  RESULT_FILE="$PARENT_DIR/perplexity_results.csv"

  if [ -f "$RESULT_FILE" ]; then
    echo "Skipping $agent ($model_path) as results already exist at $RESULT_FILE"
    continue
  fi

  echo "Starting evaluation for agent: $agent"
  CUDA_VISIBLE_DEVICES=0,7,2,3,4,5,6 accelerate launch --num_processes 7 "$SCRIPT" --merged_path "$model_path" --wandb_run_name "$agent"

  if [ $? -ne 0 ]; then
    echo "Eval failed for agent: $agent. Exiting..."
    exit 1
  fi
done
