#!/bin/bash

# Set visible CUDA devices
export CUDA_VISIBLE_DEVICES=0,7,2,3,4,5,6

# Configuration and paths
AGENT_MODELS_JSON="/playpen-ssd/smerrill/llm_decisions/configs/models.json"
SCRIPT="/playpen-ssd/smerrill/llm_decisions/generate_completion_responses.py"
RESULT_DIR="/playpen-ssd/smerrill/llm_decisions/results"

# Max responses per prompt (used in generation, judging, comparison)
MAX_RESPONSES=20

# Sweep parameters
temps=(0.7 1.0)
top_ps=(0.8 1)
top_ks=(50 100)
reps=(1.2)

# Read agent keys from JSON
agents=($(jq -r 'keys[]' "$AGENT_MODELS_JSON"))

# Loop over param combos
for temp in "${temps[@]}"; do
  for top_p in "${top_ps[@]}"; do
    for top_k in "${top_ks[@]}"; do
      for rep in "${reps[@]}"; do
        for agent in "${agents[@]}"; do
          model_path=$(jq -r --arg agent "$agent" '.[$agent]' "$AGENT_MODELS_JSON")

          output_file="${RESULT_DIR}/${agent}_T${temp}_P${top_p}_K${top_k}_R${rep}_responses.json"

          echo "🔄 Running agent: $agent | T=$temp P=$top_p K=$top_k R=$rep"
          accelerate launch --main_process_port 0 --num_processes 1 "$SCRIPT" \
            --model_path "$model_path" \
            --speaker "$agent" \
            --output_file "$output_file" \
            --temperature "$temp" \
            --top_p "$top_p" \
            --top_k "$top_k" \
            --repetition_penalty "$rep" \
            --max_prompts "$MAX_RESPONSES"
        done
      done
    done
  done
done

# Merge and clean
python /playpen-ssd/smerrill/llm_decisions/tools/merge_agent_responses.py

# Judge and compare for each merged file
for merged_file in "${RESULT_DIR}"/test_responses_T*.json; do
  echo "⚖️  Judging: $merged_file"
  accelerate launch --main_process_port 0 --num_processes 1 /playpen-ssd/smerrill/llm_decisions/judge_completion_response.py \
    --data_file "$merged_file" \
    --overwrite \
    --max_responses "$MAX_RESPONSES"

  echo "📊 Pairwise comparison: $merged_file"
  accelerate launch --main_process_port 0 --num_processes 1 /playpen-ssd/smerrill/llm_decisions/pairwise_comparison.py \
    --data_file "$merged_file" \
    --overwrite \
    --max_responses "$MAX_RESPONSES"
done
