#!/bin/bash
set -uo pipefail  # safer execution; avoids silent bugs

# === Config Paths ===
AGENT_MODELS_JSON="/playpen-ssd/smerrill/llm_decisions/configs/models.json"
SCRIPT="/playpen-ssd/smerrill/llm_decisions/generate_monologue_responses.py"
RESULT_DIR="/playpen-ssd/smerrill/llm_decisions/monologue_results"
PROMPTS_FILE="/playpen-ssd/smerrill/dataset/reverse_prompt_monologues.json"
MAX_RESPONSES=20

# === Sweep Parameters (4 combos) ===
temps=(1.0 0.4 0.9)
top_ps=(0.95 0.6 0.85)
top_ks=(50 20 80)
reps=(1.1 1.3 1.15)

# === Read agent names from JSON ===
agents=($(jq -r 'keys[]' "$AGENT_MODELS_JSON"))

# === Generation Sweep ===
for i in "${!temps[@]}"; do
  temp="${temps[$i]}"
  top_p="${top_ps[$i]}"
  top_k="${top_ks[$i]}"
  rep="${reps[$i]}"

  for agent in "${agents[@]}"; do
    model_path=$(jq -r --arg agent "$agent" '.[$agent]' "$AGENT_MODELS_JSON")
    output_file="${RESULT_DIR}/${agent}_T${temp}_P${top_p}_K${top_k}_R${rep}_responses.json"

    echo "🔄 Running agent: $agent | T=$temp P=$top_p K=$top_k R=$rep"

    if ! accelerate launch --main_process_port 0 --num_processes 1 "$SCRIPT" \
      --model_path "$model_path" \
      --speaker "$agent" \
      --output_file "$output_file" \
      --prompts_file "$PROMPTS_FILE" \
      --temperature "$temp" \
      --top_p "$top_p" \
      --top_k "$top_k" \
      --repetition_penalty "$rep" \
      --max_prompts "$MAX_RESPONSES" \
      --cot; then
        echo "❌ Generation failed: $agent | T=$temp P=$top_p K=$top_k R=$rep"
    fi

  done
done

# === Merge Results ===
echo "📦 Merging responses..."
python /playpen-ssd/smerrill/llm_decisions/tools/merge_agent_responses.py --input_dir "$RESULT_DIR" --delete_original

# === Evaluation ===
for merged_file in "${RESULT_DIR}"/test_responses_T*.json; do
  echo "🔀 Pairwise comparison: $merged_file"
  accelerate launch --main_process_port 0 --num_processes 1 /playpen-ssd/smerrill/llm_decisions/pairwise_comparison_monologue.py \
    --data_file "$merged_file" \
    --overwrite \
    --max_responses "$MAX_RESPONSES"

  echo "🧑‍⚖️ Judging: $merged_file"
  accelerate launch --main_process_port 0 --num_processes 1 /playpen-ssd/smerrill/llm_decisions/judge_monologue_responses.py \
    --data_file "$merged_file" \
    --overwrite \
    --max_responses "$MAX_RESPONSES"
done

echo "✅ Sweep completed."
