#!/bin/bash

export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1

AGENTS=(
  judyle
  kateacuff
  ellenosborne
  grahampaige
  katrinacallsen
  davidoberg
  jonnoalcaro
)

SCRIPT_PATH="/playpen-ssd/smerrill/llm_decisions/LLM_Train.py"
CONFIG_PATH="/playpen-ssd/smerrill/llm_decisions/configs/llamma_3_70b.yaml"

for AGENT in "${AGENTS[@]}"; do
  echo "Starting training for agent: $AGENT"
  torchrun --nproc_per_node=8 "$SCRIPT_PATH" --config "$CONFIG_PATH" --agent_name "$AGENT"
  if [ $? -ne 0 ]; then
    echo "Training failed for agent: $AGENT. Exiting..."
    exit 1
  fi
done
