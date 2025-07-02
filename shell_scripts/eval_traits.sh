#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,7,2,3,4,5,6
AGENT_MODELS_JSON="/playpen-ssd/smerrill/llm_decisions/configs/models.json"
SCRIPT=/playpen-ssd/smerrill/llm_decisions/generate_trait_responses.py
JUDGE_SCRIPT=/playpen-ssd/smerrill/llm_decisions/judge_trait_alignment.py


agents=($(jq -r 'keys[]' "$AGENT_MODELS_JSON"))

for agent in "${agents[@]}"; do
  model_path=$(jq -r --arg agent "$agent" '.[$agent]' "$AGENT_MODELS_JSON")

  echo "Starting training for agent: $agent"
  echo "Using model: $model_path"

accelerate launch --main_process_port 0 --num_processes 1 "$SCRIPT" --model-path "$model_path" -i /playpen-ssd/smerrill/llm_decisions/results/belief_results.json -o /playpen-ssd/smerrill/llm_decisions/results/belief_results.json
accelerate launch --main_process_port 0 --num_processes 1 "$SCRIPT" --model-path "$model_path" -i /playpen-ssd/smerrill/llm_decisions/results/memory_results.json -o /playpen-ssd/smerrill/llm_decisions/results/memory_results.json
accelerate launch --main_process_port 0 --num_processes 1 "$SCRIPT" --model-path "$model_path" -i /playpen-ssd/smerrill/llm_decisions/results/personality_results.json -o /playpen-ssd/smerrill/llm_decisions/results/personality_results.json

  if [ $? -ne 0 ]; then
    echo "Eval failed for agent: $agent. Exiting..."
    exit 1
  fi
done

# Run judging
accelerate launch --main_process_port 0 --num_processes 1 "$JUDGE_SCRIPT" --data_file /playpen-ssd/smerrill/llm_decisions/results/belief_results.json --model_name meta-llama/Meta-Llama-3-70B-Instruct --evaluation_type belief --output_key evaluation
accelerate launch --main_process_port 0 --num_processes 1 "$JUDGE_SCRIPT" --data_file /playpen-ssd/smerrill/llm_decisions/results/memory_results.json --model_name meta-llama/Meta-Llama-3-70B-Instruct --evaluation_type memory --output_key evaluation
accelerate launch --main_process_port 0 --num_processes 1 "$JUDGE_SCRIPT" --data_file /playpen-ssd/smerrill/llm_decisions/results/personality_results.json --model_name meta-llama/Meta-Llama-3-70B-Instruct --evaluation_type personality --output_key evaluation
