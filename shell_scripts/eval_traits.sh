#!/bin/bash
set -uo pipefail  # safer error handling; avoid using `-e` for sweeping

# === Paths and Config ===
AGENT_MODELS_JSON="/playpen-ssd/smerrill/llm_decisions/configs/models.json"
SCRIPT="/playpen-ssd/smerrill/llm_decisions/generate_trait_responses.py"
JUDGE_SCRIPT="/playpen-ssd/smerrill/llm_decisions/judge_trait_responses.py"
COMPARE_SCRIPT="/playpen-ssd/smerrill/llm_decisions/pairwise_comparison_traits.py"
RESULTS_DIR="/playpen-ssd/smerrill/llm_decisions/alignment_results"
DATASET_DIR="/playpen-ssd/smerrill/dataset"

# === Sweep Parameters ===
temps=(1.0 0.4 0.9)
top_ps=(0.95 0.6 0.85)
top_ks=(50 20 80)
reps=(1.1 1.3 1.15)

traits=(belief memory personality)
agents=($(jq -r 'keys[]' "$AGENT_MODELS_JSON"))

declare -a generated_files
declare -a failed_jobs
declare -A generated_files_map

# === Generation Phase ===
for agent in "${agents[@]}"; do
  model_path=$(jq -r --arg agent "$agent" '.[$agent]' "$AGENT_MODELS_JSON")
  echo "🧠 Starting inference for agent: $agent"
  echo "🔗 Model path: $model_path"

  for i in "${!temps[@]}"; do
    temperature="${temps[$i]}"
    top_p="${top_ps[$i]}"
    top_k="${top_ks[$i]}"
    rp="${reps[$i]}"
    config_suffix="T${temperature}_P${top_p}_K${top_k}_R${rp}"
    echo "⚙️ Running config: $config_suffix"

    for trait in "${traits[@]}"; do
      base_file="${DATASET_DIR}/${trait}_results.json"
      output_file="${RESULTS_DIR}/${trait}_results_${config_suffix}.json"

      if [ ! -f "$output_file" ]; then
        echo "📄 Creating base file for $trait with config $config_suffix"
        cp "$base_file" "$output_file"
      else
        echo "✅ Output file exists: $output_file"
      fi

      echo "🚀 Generating response for $agent / $trait..."

      if ! accelerate launch --main_process_port 0 --num_processes 1 "$SCRIPT" \
              --model-path "$model_path" \
              -i "$output_file" \
              -o "$output_file" \
              --temperature "$temperature" \
              --top_p "$top_p" \
              --top_k "$top_k" \
              --repetition_penalty "$rp"; then
        echo "[❌ ERROR] Generation failed: $agent | $trait | $config_suffix"
        failed_jobs+=("generate:$agent:$trait:$config_suffix")
        continue
      fi

      generated_files+=("$output_file:$trait")
      generated_files_map["$output_file:$trait"]=1
    done
  done
done

# === Judging Phase ===
echo "🧪 Starting judging phase for ${#generated_files_map[@]} files..."

for entry in "${!generated_files_map[@]}"; do
  IFS=':' read -r file trait <<< "$entry"
  echo "⚖️ Judging file: $file for trait: $trait"

  echo "🔀 Pairwise comparison..."
  if ! accelerate launch --main_process_port 0 --num_processes 1 "$COMPARE_SCRIPT" \
    --data_file "$file" \
    --judge_model "meta-llama/Meta-Llama-3-70B-Instruct" \
    --evaluation_type "$trait"; then
    echo "[❌ ERROR] Comparison failed: $file"
    failed_jobs+=("compare:$file:$trait")
    continue
  fi

  echo "🧑‍⚖️ Judging responses..."
  if ! accelerate launch --main_process_port 0 --num_processes 1 "$JUDGE_SCRIPT" \
    --data_file "$file" \
    --model_name "meta-llama/Meta-Llama-3-70B-Instruct" \
    --evaluation_type "$trait" \
    --output_key "evaluation"; then
    echo "[❌ ERROR] Judging failed: $file"
    failed_jobs+=("judge:$file:$trait")
    continue
  fi
done
