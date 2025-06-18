#!/bin/bash
export ACCELERATE_LOG_LEVEL=debug
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export CUDA_VISIBLE_DEVICES=0,7,2,3,4,5,6
export FSDP_CPU_RAM_EFFICIENT_LOADING=1

AGENTS=(
  kateacuff
  ellenosborne
  grahampaige
  katrinacallsen
  davidoberg
  jonnoalcaro
  judyle
)

FACTORS=(8 16 32)
DROPOUTS=(0.05 0.1 0.2)

SCRIPT_PATH="/playpen-ssd/smerrill/llm_decisions/train_agent_llm.py"
MERGE_PATH="/playpen-ssd/smerrill/llm_decisions/tools/merge_lora_adapters.py"
CONFIG_PATH="/playpen-ssd/smerrill/llm_decisions/configs/llamma_3_70b.yaml"

for FACTOR in "${FACTORS[@]}"; do
  for DROPOUT in "${DROPOUTS[@]}"; do
    for AGENT in "${AGENTS[@]}"; do
      echo "Starting training for agent: $AGENT with factor: $FACTOR and dropout: $DROPOUT"

      OUTPUT_DIR="/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/${AGENT}_${FACTOR}_${DROPOUT}"
      echo "Output directory: $OUTPUT_DIR"

      accelerate launch --num_processes 7 "$SCRIPT_PATH" \
        --config "$CONFIG_PATH" \
        --agent_name "$AGENT" \
        --factors "$FACTOR" \
        --dropout "$DROPOUT"

      echo "Attempting to merge directory"
      python "$MERGE_PATH" --output_dir "$OUTPUT_DIR"

      if [ $? -ne 0 ]; then
        echo "Training or merging failed for agent: $AGENT with factor: $FACTOR and dropout: $DROPOUT. Exiting..."
        exit 1
      fi
    done
  done
done
