#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,7,2,3,4,5,6

AGENT_MODELS_JSON="/playpen-ssd/smerrill/llm_decisions/configs/models.json"
SCRIPT="/playpen-ssd/smerrill/llm_decisions/generate_trait_responses.py"
JUDGE_SCRIPT="/playpen-ssd/smerrill/llm_decisions/judge_trait_responses.py"
RESULTS_DIR="/playpen-ssd/smerrill/llm_decisions/alignment_results"
DATASET_DIR="/playpen-ssd/smerrill/dataset"

# Parameter grid
temperatures=(0.7)
top_ps=(0.8)
top_ks=(50 100)
repetition_penalties=(1.2)

traits=(belief memory personality)
agents=($(jq -r 'keys[]' "$AGENT_MODELS_JSON"))

# Store generated files for judging phase
declare -a generated_files

for agent in "${agents[@]}"; do
  model_path=$(jq -r --arg agent "$agent" '.[$agent]' "$AGENT_MODELS_JSON")
  echo "Starting inference for agent: $agent"
  echo "Using model: $model_path"

  for temperature in "${temperatures[@]}"; do
    for top_p in "${top_ps[@]}"; do
      for top_k in "${top_ks[@]}"; do
        for rp in "${repetition_penalties[@]}"; do

          config_suffix="T${temperature}_P${top_p}_K${top_k}_R${rp}_agent_${agent}"
          echo "Running config: $config_suffix"

          for trait in "${traits[@]}"; do
            base_file="${DATASET_DIR}/${trait}_results.json"
            output_file="${RESULTS_DIR}/${trait}_results_${config_suffix}.json"

            # 1. Check if output file exists
            if [ ! -f "$output_file" ]; then
              echo "Output file does not exist. Creating from base for $trait with config $config_suffix"
              cp "$base_file" "$output_file"
            else
              echo "Output file already exists: $output_file"
            fi

            # 2. Generate in-place
            accelerate launch --main_process_port 0 --num_processes 1 "$SCRIPT" \
              --model-path "$model_path" \
              -i "$output_file" \
              -o "$output_file" \
              --temperature "$temperature" \
              --top_p "$top_p" \
              --top_k "$top_k" \
              --repetition_penalty "$rp"

            if [ $? -ne 0 ]; then
              echo "Generation failed for agent: $agent, trait: $trait, config: $config_suffix"
              exit 1
            fi

            # 3. Track file for judging later
            generated_files+=("$output_file:$trait")
          done
        done
      done
    done
  done
done

# === Judging Phase ===
echo "Starting judging phase for ${#generated_files[@]} files..."

for entry in "${generated_files[@]}"; do
  IFS=':' read -r file trait <<< "$entry"
  echo "Judging file: $file for trait: $trait"

  accelerate launch --main_process_port 0 --num_processes 1 "$JUDGE_SCRIPT" \
    --data_file "$file" \
    --model_name "meta-llama/Meta-Llama-3-70B-Instruct" \
    --evaluation_type "$trait" \
    --output_key "evaluation"

  if [ $? -ne 0 ]; then
    echo "Judging failed for $file"
    exit 1
  fi
done

echo "All generation and judging completed successfully."
