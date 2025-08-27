#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ================================================
# Agent-Specific Fine-Tuning Parameters
# -----------------------------------------------
# | Agent            | Content Tokens | LoRA Factors | LoRA Dropout | Learning Rate | Train Epochs |
# |------------------|----------------|--------------|--------------|----------------|---------------|
# | kateacuff        | 128,417        | 16           | 0.125        | 1e-4          | 3             |
# | ellenosborne     | 35,178         | 8            | 0.125        | 1e-4          | 2             |
# | grahampaige      | 326,740        | 32           | 0.125        | 1e-4          | 3             |
# | katrinacallsen   | 197,329        | 16           | 0.125        | 1e-4          | 3             |
# | davidoberg       | 89,394         | 8            | 0.125        | 1e-4          | 2             |
# | jonnoalcaro      | 147,408        | 16           | 0.125        | 1e-4          | 3             |
# | judyle           | 84,197         | 8            | 0.125        | 1e-4          | 2             |
# ================================================

AGENTS=(
  kateacuff
  ellenosborne
  grahampaige
  katrinacallsen
  davidoberg
  jonnoalcaro
  judyle
)

FACTORS=(16 8 32 16 8 16 8)
DROPOUTS=(0.125 0.125 0.125 0.125 0.125 0.125 0.125)
LRS=(1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4)
EPOCHS=(3 2 3 3 2 3 2)

SCRIPT_PATH="/playpen-ssd/smerrill/llm_decisions/train_agent_llm.py"
MERGE_PATH="/playpen-ssd/smerrill/llm_decisions/tools/merge_lora_adapters.py"
CONFIG_PATH="/playpen-ssd/smerrill/llm_decisions/configs/llamma_3_70b.yaml"
MODELS_JSON="/playpen-ssd/smerrill/llm_decisions/configs/models.json"

# Temp file to collect model paths
TMP_MODELS_JSON=$(mktemp)

echo "{" > "$TMP_MODELS_JSON"

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Detected $NUM_GPUS visible GPUs: $CUDA_VISIBLE_DEVICES"

for IDX in "${!AGENTS[@]}"; do
  AGENT="${AGENTS[$IDX]}"
  FACTOR="${FACTORS[$IDX]}"
  DROPOUT="${DROPOUTS[$IDX]}"
  LR="${LRS[$IDX]}"
  EPOCH="${EPOCHS[$IDX]}"

  for SYS_MESSAGE in 0 1; do
      OUTPUT_DIR="/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/${AGENT}_${FACTOR}_${DROPOUT}_${LR}_${EPOCH}_sys${SYS_MESSAGE}"

      echo "---------------------------------------------"
      echo "Training agent: $AGENT"
      echo "  lora_factors: $FACTOR"
      echo "  lora_dropout: $DROPOUT"
      echo "  learning_rate: $LR"
      echo "  train_epochs:  $EPOCH"
      echo "  sys_message:   $SYS_MESSAGE"
      echo "  output_dir:    $OUTPUT_DIR"
      echo "---------------------------------------------"

      accelerate launch --num_processes "$NUM_GPUS" "$SCRIPT_PATH" \
        --config "$CONFIG_PATH" \
        --agent_name "$AGENT" \
        --factors "$FACTOR" \
        --dropout "$DROPOUT" \
        --lr "$LR" \
        --epochs "$EPOCH" \
        --sys_message "$SYS_MESSAGE" \
        --save_dir "$OUTPUT_DIR"

      if [ $? -ne 0 ]; then
        echo "Training failed for agent: $AGENT with sys_message=$SYS_MESSAGE. Exiting..."
        exit 1
      fi

      echo "Attempting to merge directory"
      python "$MERGE_PATH" --output_dir "$OUTPUT_DIR"

      if [ $? -ne 0 ]; then
        echo "Merging failed for agent: $AGENT with sys_message=$SYS_MESSAGE. Exiting..."
        exit 1
      fi

      MERGED_PATH="${OUTPUT_DIR}/merged"
      echo "  -> Merged to: $MERGED_PATH"

      # Append JSON line
      echo "  \"${AGENT}_sys${SYS_MESSAGE}\": \"$MERGED_PATH\"," >> "$TMP_MODELS_JSON"
  done
done

# Finalize models.json (remove trailing comma and close JSON)
sed -i '$ s/,$//' "$TMP_MODELS_JSON"
echo "}" >> "$TMP_MODELS_JSON"

# Move to final location
mv "$TMP_MODELS_JSON" "$MODELS_JSON"
echo "✅ models.json saved to $MODELS_JSON"
