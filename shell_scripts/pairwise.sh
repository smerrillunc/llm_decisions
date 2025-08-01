#!/bin/bash

# --- Configuration ---
# Set visible CUDA devices. Adjust as needed for your system.
#sexport CUDA_VISIBLE_DEVICES=0,1,2,5,6,7

# Maximum number of responses to process per file
MAX_RESPONSES=20

# Base directory for your data files
DATA_BASE_DIR="/playpen-ssd/smerrill/llm_decisions/results"

# Path to your Python evaluation script
PYTHON_SCRIPT="/playpen-ssd/smerrill/llm_decisions/pairwise_comparison.py"

# --- Main Logic ---

echo "Starting evaluation process for all JSON files in ${DATA_BASE_DIR}..."
echo "Using CUDA Devices: ${CUDA_VISIBLE_DEVICES}"
echo "Processing up to ${MAX_RESPONSES} responses per file."

# Loop through each .json file in the specified directory
for data_file in "${DATA_BASE_DIR}"/test_responses_*.json; do
  # Check if any files were found (in case glob matches nothing)
  if [ -e "$data_file" ]; then
    echo "" # Add a newline for better readability between runs
    echo "----------------------------------------------------"
    echo "Processing file: $(basename "$data_file")"
    echo "Full path: $data_file"
    echo "----------------------------------------------------"

    # Run the Python script with accelerate launch
    # --main_process_port 0: Specifies the port for the main process (can be changed if needed)
    # --num_processes 1: Runs the script using a single process (as per your example)
    CUDA_VISIBLE_DEVICES=0,1,2,5,6,7 accelerate launch --main_process_port 0 --num_processes 1 "$PYTHON_SCRIPT" \
      --data_file "$data_file" \
      --overwrite \
      --max_responses "$MAX_RESPONSES"

    # Check the exit status of the last command
    if [ $? -eq 0 ]; then
      echo "Successfully processed $(basename "$data_file")"
    else
      echo "Error processing $(basename "$data_file"). Check logs above."
      # Optionally, you can add `exit 1` here to stop the script on the first error
    fi
  else
    echo "No JSON files found matching pattern 'test_responses_*.json' in ${DATA_BASE_DIR}."
    break # Exit the loop if no files are found
  fi
done

echo ""
echo "All evaluation processes completed."

