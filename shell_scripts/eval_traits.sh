#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,7,2,3,4,5,6

AGENT_MODELS_JSON="/playpen-ssd/smerrill/llm_decisions/configs/models.json"
SCRIPT="/playpen-ssd/smerrill/llm_decisions/generate_trait_responses.py"
JUDGE_SCRIPT="/playpen-ssd/smerrill/llm_decisions/judge_trait_responses.py"

# Parameter grid
temperatures=(0.7)
top_ps=(0.8)
top_ks=(50 100)
repetition_penalties=(1.2)

traits=(belief memory personality)
agents=($(jq -r 'keys[]' "$AGENT_MODELS_JSON"))

for agent in "${agents[@]}"; do
  model_path=$(jq -r --arg agent "$agent" '.[$agent]' "$AGENT_MODELS_JSON")

  echo "Starting inference for agent: $agent"
  echo "Using model: $model_path"

  for temperature in "${temperatures[@]}"; do
    for top_p in "${top_ps[@]}"; do
      for top_k in "${top_ks[@]}"; do
        for rp in "${repetition_penalties[@]}"; do

          config_suffix="T${temperature}_P${top_p}_K${top_k}_R${rp}"
          echo "Running config: $config_suffix"

          for trait in "${traits[@]}"; do
            input_file="/playpen-ssd/smerrill/dataset/${trait}_results.json"
            output_file="/playpen-ssd/smerrill/llm_decisions/alignment_results/${trait}_results_${config_suffix}.json"

            # Generation
            accelerate launch --main_process_port 0 --num_processes 1 "$SCRIPT" \
              --model-path "$model_path" \
              -i "$input_file" \
              -o "$output_file" \
              --temperature "$temperature" \
              --top_p "$top_p" \
              --top_k "$top_k" \
              --repetition_penalty "$rp"

            if [ $? -ne 0 ]; then
              echo "Generation failed for agent: $agent, trait: $trait, config: $config_suffix"
              exit 1
            fi

            # Judging
            accelerate launch --main_process_port 0 --num_processes 1 "$JUDGE_SCRIPT" \
              --data_file "$output_file" \
              --model_name "meta-llama/Meta-Llama-3-70B-Instruct" \
              --evaluation_type "$trait" \
              --output_key "evaluation"

            if [ $? -ne 0 ]; then
              echo "Judging failed for $output_file"
              exit 1
            fi

          done
        done
      done
    done
  done
done
