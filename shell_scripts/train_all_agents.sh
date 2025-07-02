#!/bin/bash
export ACCELERATE_LOG_LEVEL=debug
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export CUDA_VISIBLE_DEVICES=0,7,2,3,4,5,6

# Agent list (order matters!)
AGENTS=(
  kateacuff      # 43,457 context, 22,398 content: factors=12, dropout=0.2
  ellenosborne   # 9,135 context, 5,728 content: factors=8,  dropout=0.4
  grahampaige    # 88,642 context, 45,121 content: factors=16, dropout=0.1
  katrinacallsen # 57,797 context, 41,910 content: factors=12, dropout=0.2
  davidoberg     # 307,800 context, 19,247 content: factors=16, dropout=0.1
  jonnoalcaro    # 49,119 context, 31,320 content: factors=12, dropout=0.2
  judyle         # 26,806 context, 17,462 content: factors=8,  dropout=0.3
)

# Per-agent parameters (must match AGENTS order)
FACTORS=(12 8 16 12 16 12 8)
DROPOUTS=(0.2 0.4 0.1 0.2 0.1 0.2 0.3)

SCRIPT_PATH="/playpen-ssd/smerrill/llm_decisions/train_agent_llm.py"
MERGE_PATH="/playpen-ssd/smerrill/llm_decisions/tools/merge_lora_adapters.py"
CONFIG_PATH="/playpen-ssd/smerrill/llm_decisions/configs/llamma_3_70b.yaml"

for IDX in "${!AGENTS[@]}"; do
  AGENT="${AGENTS[$IDX]}"
  FACTOR="${FACTORS[$IDX]}"
  DROPOUT="${DROPOUTS[$IDX]}"
  OUTPUT_DIR="/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/${AGENT}_${FACTOR}_${DROPOUT}"

  # Display the parameters for clarity
  echo "---------------------------------------------"
  echo "Training agent: $AGENT"
  echo "  lora_factors: $FACTOR"
  echo "  lora_dropout: $DROPOUT"
  echo "  Output dir:   $OUTPUT_DIR"
  echo "---------------------------------------------"

  if [ -d "$OUTPUT_DIR" ]; then
    echo "Skipping: $OUTPUT_DIR already exists."
    continue
  fi

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
