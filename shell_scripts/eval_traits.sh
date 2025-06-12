#!/bin/bash

MODELS=(
  /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/judyle_16/merged
  /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/ellenosborne_16/merged
  /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/grahampaige_16/merged
  /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/katrinacallsen_16/merged
  /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/kateacuff_16/merged
  /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/jonnoalcaro_16/merged
)

for MODEL in "${MODELS[@]}"; do
  echo "Starting training for agent: $MODEL"

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 generate_agent_responses.py --model-path "$MODEL" -i /playpen-ssd/smerrill/llm_decisions/results/belief_results.json -o /playpen-ssd/smerrill/llm_decisions/results/belief_results.json
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 generate_agent_responses.py --model-path "$MODEL" -i /playpen-ssd/smerrill/llm_decisions/results/memory_results.json -o /playpen-ssd/smerrill/llm_decisions/results/memory_results.json
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 generate_agent_responses.py --model-path "$MODEL" -i /playpen-ssd/smerrill/llm_decisions/results/personality_results.json -o /playpen-ssd/smerrill/llm_decisions/results/personality_results.json

  if [ $? -ne 0 ]; then
    echo "Eval failed for agent: $MODEL. Exiting..."
    exit 1
  fi
done

# Run judging
accelerate launch --num_processes 1 judge_response_alignment.py --data_file /playpen-ssd/smerrill/llm_decisions/results/belief_results.json --model_name meta-llama/Meta-Llama-3-70B-Instruct --evaluation_type belief --output_key belief_eval
accelerate launch --num_processes 1 judge_response_alignment.py --data_file /playpen-ssd/smerrill/llm_decisions/results/memory_results.json --model_name meta-llama/Meta-Llama-3-70B-Instruct --evaluation_type memory --output_key memory_eval
accelerate launch --num_processes 1 judge_response_alignment.py --data_file /playpen-ssd/smerrill/llm_decisions/results/personality_results.json --model_name meta-llama/Meta-Llama-3-70B-Instruct --evaluation_type personality --output_key personality_eval
