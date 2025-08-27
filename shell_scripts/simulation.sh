#!/bin/bash
# =========================
# Parameter Sweep + Repeats Script
# =========================

# Sweep parameters
temps=(1.0 0.4 0.9 0.7)
top_ps=(0.95 0.6 0.85 0.8)
top_ks=(50 20 80 100)
reps=(1.1 1.3 1.15 1.2)

# Number of agenda items
START_IDX=0
END_IDX=5

# Number of repeats per agenda item
REPEATS=5

# Iterate over agenda items
for i in $(seq $START_IDX $END_IDX)
do
    # Repeat each agenda item multiple times
    for run_id in $(seq 1 $REPEATS)
    do
        # Sweep over parameter combinations
        for j in "${!temps[@]}"; do
            temp="${temps[$j]}"
            top_p="${top_ps[$j]}"
            top_k="${top_ks[$j]}"
            rep="${reps[$j]}"

            # Define save directory including agenda number, run ID, and param combo
            save_dir="/playpen-ssd/smerrill/llm_decisions/results_simulation/AgendaItem_${i}/Run${run_id}/T${temp}_P${top_p}_K${top_k}_R${rep}"

            # Skip if results already exist
            if [ -d "$save_dir" ]; then
                echo "[INFO] Skipping item $i, run $run_id – results exist for temp=${temp}, top_p=${top_p}, top_k=${top_k}, rep=${rep}"
                echo "--------------------------------------------------"
                continue
            fi

            echo "[INFO] Processing Agenda Item Index: $i, Run: $run_id"
            echo "[INFO] Save Directory: $save_dir"
            echo "[INFO] Params: temperature=${temp}, top_p=${top_p}, top_k=${top_k}, repetition_penalty=${rep}"

            # Run the simulation command
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
            accelerate launch --num_processes=1 /playpen-ssd/smerrill/llm_decisions/simulate.py \
                --base_model meta-llama/Meta-Llama-3-70B-Instruct \
                --config /playpen-ssd/smerrill/llm_decisions/configs/models.json \
                --agenda_item "$i" \
                --save_dir "$save_dir" \
                --temperature "$temp" \
                --top_p "$top_p" \
                --top_k "$top_k" \
                --repetition_penalty "$rep"

            echo "[INFO] Finished Agenda Item $i, Run $run_id, param combo $((j+1))"
            echo "--------------------------------------------------"
        done
    done
done

echo "[INFO] All agenda items, repeats, and parameter sweeps processed."
