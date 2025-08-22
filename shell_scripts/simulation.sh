#!/bin/bash

# Path to the agenda JSON file
AGENDA_FILE="/playpen-ssd/smerrill/llm_decisions/configs/agenda.json"

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "[ERROR] jq is not installed. Please install jq to parse JSON."
    exit 1
fi

# Check if agenda file exists
if [ ! -f "$AGENDA_FILE" ]; then
    echo "[ERROR] Agenda file not found at $AGENDA_FILE"
    exit 1
fi

# Sweep parameters
temps=(1.0 0.4 0.9 0.7)
top_ps=(0.95 0.6 0.85 0.8)
top_ks=(50 20 80 100)
reps=(1.1 1.3 1.15 1.2)

# Iterate over agenda items
ITEM_COUNT=$(jq length "$AGENDA_FILE")
echo "[INFO] Starting simulation for $ITEM_COUNT agenda items"

for i in $(seq 0 $((ITEM_COUNT-1)))
do
    agenda_item=$(jq -r ".[$i].agenda_item" "$AGENDA_FILE")
    vote_prompt=$(jq -r ".[$i].vote_prompt" "$AGENDA_FILE")

    # Extract the agenda number (e.g., "3.1" from "Agenda Item No. 3.1: ...")
    agenda_number=$(echo "$agenda_item" | sed -n 's/.*Agenda Item No\. \([0-9.]\+\):.*/\1/p')

    # Sweep over parameter combinations
    for j in "${!temps[@]}"; do
        temp="${temps[$j]}"
        top_p="${top_ps[$j]}"
        top_k="${top_ks[$j]}"
        rep="${reps[$j]}"

        # Define save directory including param combo
        save_dir="/playpen-ssd/smerrill/llm_decisions/results_simulation/${agenda_number}/T${temp}_P${top_p}_K${top_k}_R${rep}"

        # Skip if results already exist
        if [ -d "$save_dir" ]; then
            echo "[INFO] Skipping item $((i+1)) ($agenda_number) – results exist for temp=${temp}, top_p=${top_p}, top_k=${top_k}, rep=${rep}"
            echo "--------------------------------------------------"
            continue
        fi

        echo "[INFO] Processing item $((i+1)) of $ITEM_COUNT"
        echo "[INFO] Agenda Item: $agenda_item"
        echo "[INFO] Vote Prompt: $vote_prompt"
        echo "[INFO] Save Directory: $save_dir"
        echo "[INFO] Params: temperature=${temp}, top_p=${top_p}, top_k=${top_k}, repetition_penalty=${rep}"

        # Run the simulation command
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        accelerate launch --num_processes=1 /playpen-ssd/smerrill/llm_decisions/simulate.py \
            --base_model meta-llama/Meta-Llama-3-70B-Instruct \
            --config /playpen-ssd/smerrill/llm_decisions/configs/models.json \
            --agenda_item "$agenda_item" \
            --vote_prompt "$vote_prompt" \
            --save_dir "$save_dir" \
            --temperature "$temp" \
            --top_p "$top_p" \
            --top_k "$top_k" \
            --repetition_penalty "$rep"

        echo "[INFO] Finished processing item $((i+1)) with param combo $((j+1))"
        echo "--------------------------------------------------"
    done
done

echo "[INFO] All agenda items and parameter sweeps processed."
