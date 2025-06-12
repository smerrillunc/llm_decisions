# LLM Decisions Pipeline

This repository contains scripts and utilities for training, merging, evaluating, and analyzing large language models (LLMs) for personalized conversational agents. The workflow supports multi-GPU training, quantized inference, and downstream analysis of model outputs.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Scripts Overview](#scripts-overview)
  - [Training: `LLM_Train.py`](#training-llm_trainpy)
  - [Merging Adapters: `merge_peft.py`](#merging-adapters-merge_peftpy)
  - [Perplexity Evaluation: `LLM_Test.py`](#perplexity-evaluation-llm_testpy)
  - [Monologue Analysis: `run_monologue_analysis.py`](#monologue-analysis-run_monologue_analysispy)
  - [Question Review CLI: `question_CLI.py`](#question-review-cli-question_clip)
  - [Automated Question Answering: `askQuestions.py`](#automated-question-answering-askquestionspy)
  - [Alignment Evaluation: `evaluate_alignment.py`](#alignment-evaluation-evaluate_alignmentpy)
  - [Train All Agents: `train_all_agents.sh`](#train-all-agents-train_all_agentssh)
- [Data Structure](#data-structure)
- [Tips and Troubleshooting](#tips-and-troubleshooting)

---

## Environment Setup

1. **Install dependencies** (Python 3.10+ recommended):

    ```bash
    pip install torch transformers accelerate peft trl datasets wandb evaluate bitsandbytes
    ```

2. **Set up CUDA and GPUs** as needed for your hardware.

3. **(Optional) Configure Accelerate**:

    ```bash
    accelerate config
    ```

---

## Scripts Overview

### Training: `LLM_Train.py`

Train a LoRA/QLoRA model for a specific agent using multi-GPU and FSDP.

**Example command:**
```bash
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
torchrun --nproc_per_node=8 llm_decisions/LLM_Train.py --config llm_decisions/configs/llamma_3_70b.yaml --agent_name kateacuff
```

- **Arguments**: See `ScriptArguments` in the script for all options (e.g., `--agent_name`, `--dataset_path`, `--wandb_project`).
- **Output**: Trained model and logs are saved to the specified output directory.

---

### Merging Adapters: `merge_peft.py`

Merge LoRA/PEFT adapters into the base model for efficient inference.

**Example command:**
```bash
python llm_decisions/merge_peft.py
```

- **Edit** the `output_dirs` list in the script to include the directories you want to merge.
- **Output**: Merged models are saved in a `merged` subdirectory within each agent's output directory.

---

### Perplexity Evaluation: `LLM_Test.py`

Compute perplexity for each agent's test set using the merged model.

**Example command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 llm_decisions/LLM_Test.py --merged_path /path/to/agent/merged
```

- **Arguments**: `--merged_path` (required), `--wandb_project`, etc.
- **Output**: Perplexity results are saved as `perplexity_results.csv` in the merged model directory.

**Note:**  
- Only use `--num_processes 1` for inference.  
- Do **not** use FSDP or multi-process for evaluation with quantized models.

---

### Monologue Analysis: `run_monologue_analysis.py`

Extract personality, beliefs, and memory cues from monologue data using the LLM.

**Example command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 llm_decisions/run_monologue_analysis.py --model /path/to/merged --input /path/to/monologues.pkl --output_dir ./results --single_speaker ellenosborne
```

- **Arguments**: See `parse_args()` in the script.
- **Output**: JSON files for personality, belief, and memory results in the output directory.

---

### Question Review CLI: `question_CLI.py`

Interactive CLI to review and edit generated questions for each speaker.

**Example command:**
```bash
python llm_decisions/question_CLI.py --file ./results/personality_results.json --speaker ellenosborne
```

- **Arguments**: `--file` (required), `--speaker` (optional).
- **Output**: Edits are saved back to the JSON file.

---

### Automated Question Answering: `askQuestions.py`

Generate model responses to the generated questions for each speaker.

**Example command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 llm_decisions/askQuestions.py --model-path /path/to/merged --input-file ./results/belief_results.json --output-file ./results/belief_results.json
```

- **Arguments**: `--model-path`, `--input-file`, `--output-file`.
- **Output**: Updates the JSON file with model responses.

---

### Alignment Evaluation: `evaluate_alignment.py`

Use a judge LLM to score the alignment of model responses with speaker beliefs.

**Example command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 llm_decisions/evaluate_alignment.py --data_file ./results/belief_results.json --model_name meta-llama/Meta-Llama-3-70B-Instruct
```

- **Arguments**: `--data_file`, `--model_name`, `--speaker` (optional), `--overwrite`.
- **Output**: Adds an `evaluation` field to each entry in the JSON file.

---

### Train All Agents: `train_all_agents.sh`

Automate training for all agents in sequence.

**Example command:**
```bash
bash llm_decisions/train_all_agents.sh
```

- **Edit** the `AGENTS` array and paths in the script as needed.
- **Output**: Trains and saves models for each agent.

---

## Data Structure

- **Training/Test Data**: `.npy` files per agent in the dataset directory.
- **Monologues**: Pickled dict of speaker monologues.
- **Results**: JSON files for each analysis step, CSV for perplexity.

---

## Tips and Troubleshooting

- **OOM Errors**: Free up GPU memory, use fewer GPUs, or reduce batch size/sequence length.
- **bitsandbytes Quantization**: Quantized models require CUDA; do not offload to CPU.
- **Inference**: Always use `--num_processes 1` for evaluation/inference scripts.
- **Device Mismatch**: For sharded models, always move input tensors to the device of the model's input embedding layer.
- **WandB**: Set `WANDB_API_KEY` and `WANDB_PROJECT` as needed for experiment tracking.

---