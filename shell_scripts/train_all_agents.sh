#!/bin/bash

# ================================================
# Agent-Specific Fine-Tuning Parameters
# -----------------------------------------------
# | Agent            | Content Words | LoRA Factors | LoRA Dropout | Learning Rate | Train Epochs |
# |------------------|---------------|--------------|--------------|----------------|---------------|
# | ellenosborne     | 5,728         | 4            | 0.175        | 5e-6           | 2             |
# | judyle           | 17,462        | 12           | 0.125        | 1e-5           | 3             |
# | davidoberg       | 19,247        | 12           | 0.125        | 1e-5           | 3             |
# | kateacuff        | 22,398        | 12           | 0.125        | 1e-5           | 3             |
# | jonnoalcaro      | 31,320        | 16           | 0.125        | 4e-5           | 3             |
# | katrinacallsen   | 41,910        | 16           | 0.075        | 2e-5           | 4             |
# | grahampaige      | 45,121        | 16           | 0.075        | 2e-5           | 4             |
# ================================================

export ACCELERATE_LOG_LEVEL=debug
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export CUDA_VISIBLE_DEVICES=0,7,2,3,4,5,6

AGENTS=(
  kateacuff
  ellenosborne
  grahampaige
  katrinacallsen
  davidoberg
  jonnoalcaro
  judyle
)

FACTORS=(12 4 16 16 12 16 12)
DROPOUTS=(0.125 0.175 0.075 0.075 0.125 0.125 0.125)
LRS=(1e-5 5e-6 2e-5 2e-5 1e-5 4e-5 1e-5)
EPOCHS=(3 2 4 4 3 3 3)

SCRIPT_PATH="/playpen-ssd/smerrill/llm_decisions/train_agent_llm.py"
MERGE_PATH="/playpen-ssd/smerrill/llm_decisions/tools/merge_lora_adapters.py"
CONFIG_PATH="/playpen-ssd/smerrill/llm_decisions/configs/llamma_3_70b.yaml"
MODELS_JSON="/playpen-ssd/smerrill/llm_decisions/configs/models.json"

# Temp file to collect model paths
TMP_MODELS_JSON=$(mktemp)

echo "{" > "$TMP_MODELS_JSON"

for IDX in "${!AGENTS[@]}"; do
  AGENT="${AGENTS[$IDX]}"
  FACTOR="${FACTORS[$IDX]}"
  DROPOUT="${DROPOUTS[$IDX]}"
  LR="${LRS[$IDX]}"
  EPOCH="${EPOCHS[$IDX]}"
  OUTPUT_DIR="/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/${AGENT}_${FACTOR}_${DROPOUT}_${LR}_${EPOCH}"

  echo "---------------------------------------------"
  echo "Training agent: $AGENT"
  echo "  lora_factors: $FACTOR"
  echo "  lora_dropout: $DROPOUT"
  echo "  learning_rate: $LR"
  echo "  train_epochs:  $EPOCH"
  echo "  output_dir:    $OUTPUT_DIR"
  echo "---------------------------------------------"

  if [ -d "$OUTPUT_DIR" ]; then
    echo "Skipping: $OUTPUT_DIR already exists."
  else
    accelerate launch --num_processes 7 "$SCRIPT_PATH" \
      --config "$CONFIG_PATH" \
      --agent_name "$AGENT" \
      --factors "$FACTOR" \
      --dropout "$DROPOUT" \
      --lr "$LR" \
      --epochs "$EPOCH" \
      --save_dir "$OUTPUT_DIR"

    if [ $? -ne 0 ]; then
      echo "Training failed for agent: $AGENT. Exiting..."
      exit 1
    fi
  fi

  echo "Attempting to merge directory"
  python "$MERGE_PATH" --output_dir "$OUTPUT_DIR"

  if [ $? -ne 0 ]; then
    echo "Merging failed for agent: $AGENT. Exiting..."
    exit 1
  fi

  MERGED_PATH="${OUTPUT_DIR}/merged"
  echo "  -> Merged to: $MERGED_PATH"

  # Append JSON line
  echo "  \"$AGENT\": \"$MERGED_PATH\"," >> "$TMP_MODELS_JSON"
done

# Finalize models.json (remove trailing comma and close JSON)
sed -i '$ s/,$//' "$TMP_MODELS_JSON"
echo "}" >> "$TMP_MODELS_JSON"

# Move to final location
mv "$TMP_MODELS_JSON" "$MODELS_JSON"
echo "✅ models.json saved to $MODELS_JSON"