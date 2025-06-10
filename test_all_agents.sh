#!/bin/bash

export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1

MODELS=(
  /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/ellenosborne_16/merged
  /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/grahampaige_16/merged
  /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/katrinacallsen_16/merged
  /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/kateacuff_16/merged
  /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/jonnoalcaro_16/merged
  /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/judyle_16/merged
)


for MODEL in "${MODELS[@]}"; do
  echo "Starting training for agent: $MODEL"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 accelerate launch --num_processes 1 LLM_Test.py --merged_path "$MODEL" --wandb_run_name "$MODEL"
  if [ $? -ne 0 ]; then
    echo "Training failed for agent: $AGENT. Exiting..."
    exit 1
  fi
done
