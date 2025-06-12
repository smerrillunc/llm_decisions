#!/bin/bash
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

AGENTS=(
  kateacuff
  ellenosborne
  grahampaige
  katrinacallsen
  davidoberg
  jonnoalcaro
)

SCRIPT_PATH="/playpen-ssd/smerrill/llm_decisions/train_agent_llm.py"
CONFIG_PATH="/playpen-ssd/smerrill/llm_decisions/configs/llamma_3_70b.yaml"

for AGENT in "${AGENTS[@]}"; do
  echo "Starting training for agent: $AGENT"
  accelerate launch --num_processes 4 train_agent_llm.py --agent_name "$AGENT" --config "$CONFIG_PATH" 

  if [ $? -ne 0 ]; then
    echo "Training failed for agent: $AGENT. Exiting..."
    exit 1
  fi
done
