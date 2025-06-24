#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,7,2,3,4,5,6

AGENT_MODELS_JSON="/playpen-ssd/smerrill/llm_decisions/configs/models.json"
SCRIPT=/playpen-ssd/smerrill/llm_decisions/evaluate_agent_perplexity.py

agents=($(jq -r 'keys[]' "$AGENT_MODELS_JSON"))

for agent in "${agents[@]}"; do
  MODEL=$(jq -r --arg agent "$agent" '.[$agent]' "$AGENT_MODELS_JSON")

  # Get the parent directory (one above "merged")
  PARENT_DIR=$(dirname "$MODEL")
  RESULT_FILE="$PARENT_DIR/perplexity_results.csv"

  if [ -f "$RESULT_FILE" ]; then
    echo "Skipping $MODEL as results already exist at $RESULT_FILE"
    continue
  fi

  echo "Starting evaluation for agent: $agent"
  accelerate launch --main_process_port 0 --num_processes 7 "$SCRIPT" --merged_path "$MODEL" --wandb_run_name "$agent"

  if [ $? -ne 0 ]; then
    echo "Eval failed for agent: $agent. Exiting..."
    exit 1
  fi
done
