#!/bin/bash
GPUS="0,1,2,5,6,7"

# Set up result directories
BASE_DIR="/playpen-ssd/smerrill/llm_decisions"
RESULTS_DIR="$BASE_DIR/results"
TODAY='2025-07-29'
TODAYS_RESULT_DIR="$RESULTS_DIR/$TODAY"

mkdir -p "$BASE_DIR/completion_results"
mkdir -p "$BASE_DIR/alignment_results"
mkdir -p "$BASE_DIR/monologue_results"
mkdir -p "$TODAYS_RESULT_DIR"

# Define all script paths
scripts=(
    #"$BASE_DIR/shell_scripts/train_all_agents.sh"
    #"$BASE_DIR/shell_scripts/eval_perplexity.sh"
    #"$BASE_DIR/shell_scripts/eval_votes.sh"
    "$BASE_DIR/shell_scripts/eval_model_completions.sh"
    "$BASE_DIR/shell_scripts/eval_model_monologues.sh"
    "$BASE_DIR/shell_scripts/eval_traits.sh"
)

# Run each script, continue even if one fails
for script in "${scripts[@]}"; do
    echo "Running: $script"
    if CUDA_VISIBLE_DEVICES=$GPUS bash "$script"; then
        echo "Finished: $script successfully"
    else
        echo "Error running: $script (continuing to next)"
    fi
    echo "-----------------------------"
done

# Move results into dated results directory
mkdir -p "$TODAYS_RESULT_DIR"

mv "$BASE_DIR/completion_results" "$TODAYS_RESULT_DIR/"
mv "$BASE_DIR/alignment_results" "$TODAYS_RESULT_DIR/"
mv "$BASE_DIR/monologue_results" "$TODAYS_RESULT_DIR/"

# Copy models.json into the results folder
cp "$BASE_DIR/configs/models.json" "$TODAYS_RESULT_DIR/models.json"
python /playpen-ssd/smerrill/llm_decisions/tools/generate_overview.py --output_dir /playpen-ssd/smerrill/llm_decisions/results/2025-07-29


echo "All scripts attempted. Results saved to: $TODAYS_RESULT_DIR"
