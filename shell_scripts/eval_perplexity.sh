#!/bin/bash

MODELS=(
/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/ellenosborne_16_0.05/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/davidoberg_16_0.1/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/grahampaige_16_0.05/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/katrinacallsen_16_0.1/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/judyle_8_0.1/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/judyle_8_0.05/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/katrinacallsen_8_0.1/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/davidoberg_16_0.05/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/grahampaige_8_0.05/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/judyle_16_0.05/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/katrinacallsen_8_0.05/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/jonnoalcaro_16_0.05/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/katrinacallsen_16_0.05/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/davidoberg_8_0.05/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/grahampaige_16_0.1/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/kateacuff_16_0.1/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/ellenosborne_16_0.1/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/kateacuff_8_0.05/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/davidoberg_8_0.1/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/kateacuff_8_0.1/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/grahampaige_8_0.1/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/ellenosborne_8_0.05/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/ellenosborne_8_0.1/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/jonnoalcaro_16_0.1/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/kateacuff_16_0.05/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/jonnoalcaro_8_0.1/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/jonnoalcaro_8_0.05/merged
 /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/baseline
)

SCRIPT=/playpen-ssd/smerrill/llm_decisions/evaluate_agent_perplexity.py

for MODEL in "${MODELS[@]}"; do
  # Get the parent directory (one above "merged")
  PARENT_DIR=$(dirname "$MODEL")
  RESULT_FILE="$PARENT_DIR/perplexity_results.csv"

  if [ -f "$RESULT_FILE" ]; then
    echo "Skipping $MODEL as results already exist at $RESULT_FILE"
    continue
  fi

  echo "Starting training for agent: $MODEL"
  CUDA_VISIBLE_DEVICES=0,7,2,3,4,5,6 accelerate launch --num_processes 7 "$SCRIPT" --merged_path "$MODEL" --wandb_run_name "$MODEL"
  
  if [ $? -ne 0 ]; then
    echo "Eval failed for agent: $MODEL. Exiting..."
    exit 1
  fi
done
