#!/bin/bash

# Define all script paths
scripts=(
    "/playpen-ssd/smerrill/llm_decisions/shell_scripts/train_all_agents.sh"
    "/playpen-ssd/smerrill/llm_decisions/shell_scripts/eval_perplexity.sh"
    "/playpen-ssd/smerrill/llm_decisions/shell_scripts/eval_votes.sh"
    "/playpen-ssd/smerrill/llm_decisions/shell_scripts/eval_model_completions.sh"
    "/playpen-ssd/smerrill/llm_decisions/shell_scripts/eval_model_monologues.sh"
    "/playpen-ssd/smerrill/llm_decisions/shell_scripts/eval_traits.sh"
)

# Run each script, continue even if one fails
for script in "${scripts[@]}"; do
    echo "Running: $script"
    if bash "$script"; then
        echo "Finished: $script successfully"
    else
        echo "Error running: $script (continuing to next)"
    fi
    echo "-----------------------------"
done

echo "All scripts attempted."
