# LLM Decisions Pipeline

This repository provides a modular, multi-stage pipeline for training, merging, evaluating, and analyzing large language models (LLMs) as personalized conversational agents. The workflow supports multi-GPU training, quantized inference, and downstream analysis of model outputs, including belief/personality extraction, question generation, and alignment evaluation. All major steps are orchestrated via shell scripts for reproducibility and ease of use.

---

## Table of Contents
- [Environment Setup](#environment-setup)
- [Pipeline Overview](#pipeline-overview)
- [Detailed Workflow and Script Descriptions](#detailed-workflow-and-script-descriptions)
  - [1. Training and Merging](#1-training-and-merging)
  - [2. Perplexity Evaluation](#2-perplexity-evaluation)
  - [3. Trait/Belief/Memory Extraction and Question Generation](#3-traitbeliefmemory-extraction-and-question-generation)
  - [4. Agent Response Generation](#4-agent-response-generation)
  - [5. Judging and Alignment Evaluation](#5-judging-and-alignment-evaluation)
  - [6. Vote Dataset Creation and Evaluation](#6-vote-dataset-creation-and-evaluation)
  - [7. Completion Response Generation and Judging](#7-completion-response-generation-and-judging)
  - [8. Manual Review and Utilities](#8-manual-review-and-utilities)
- [Shell Scripts: Orchestration](#shell-scripts-orchestration)
- [Tips and Troubleshooting](#tips-and-troubleshooting)

---

## Environment Setup

1. **Install dependencies** (Python 3.10+ recommended):
   - See `requirements.txt` for Python package requirements.
2. **Set up CUDA and GPUs** as needed for your hardware.
3. **Configure Accelerate**:
   - Run `accelerate config` and set up your distributed environment as needed.
4. **Set environment variables** as needed for FSDP, CUDA, and memory management (see shell scripts for examples).

---

## Pipeline Overview

The pipeline is designed to:
1. **Train** LoRA/QLoRA adapters for each agent using multi-GPU FSDP.
2. **Merge** adapters into the base model for efficient inference.
3. **Evaluate** perplexity of each agent model on held-out data.
4. **Extract** beliefs, memories, and personality traits from agent monologues.
5. **Generate** probing questions for each agent based on extracted traits.
6. **Generate** agent responses to these questions using the merged LLMs.
7. **Judge** the alignment of agent responses using a separate judge LLM.
8. **Create and evaluate** a yes/no vote dataset for agent models.
9. **Generate and judge** completion responses for further analysis.
10. **(Optional) Review and edit** generated questions via CLI.

All steps are orchestrated via shell scripts in `llm_decisions/shell_scripts/`, which ensure correct order and environment setup.

---

## Detailed Workflow and Script Descriptions

### 1. Training and Merging

#### `train_agent_llm.py`
- **Purpose:** Trains a LoRA/QLoRA adapter for each agent using multi-GPU FSDP. Handles dataset loading, tokenizer/model setup, LoRA config, and training loop.
- **Inputs:** Agent name, config YAML, dataset path, LoRA factors/dropout.
- **Outputs:** Trained LoRA adapter and checkpoints in `trained_models/`.
- **How to run:**
  - Use `shell_scripts/train_all_agents.sh` to train all agents with their respective hyperparameters.
  - The script also calls the merge step below after each agent is trained.

#### `merge_lora_adapters.py`
- **Purpose:** Merges LoRA/PEFT adapters into the base model for efficient inference. Produces a "merged" model directory for downstream evaluation and inference.
- **Inputs:** Output directory from training.
- **Outputs:** Merged model in `merged/` subdirectory.
- **How to run:**
  - Called automatically from `train_all_agents.sh` after each agent is trained.

### 2. Perplexity Evaluation

#### `evaluate_agent_perplexity.py`
- **Purpose:** Computes perplexity for each agent's test set using the merged model. Uses Accelerate for distributed evaluation.
- **Inputs:** Merged model path, agent name.
- **Outputs:** Perplexity results CSV per agent.
- **How to run:**
  - Use `shell_scripts/eval_perplexity.sh` to evaluate all agent models in batch.

### 3. Trait/Belief/Memory Extraction and Question Generation

#### `extract_agent_traits.py`
- **Purpose:** Extracts beliefs, memories, and personality traits from agent monologues using the LLM. Generates summaries and candidate probing questions.
- **Inputs:** Pickle of agent monologues.
- **Outputs:** JSON profiles and questions.
- **How to run:**
  - Run directly with appropriate arguments (see script for details).

#### `build_agent_profiles.py`
- **Purpose:** Generates detailed personality profiles and interview questions for each agent from their monologues.
- **Inputs:** Pickle of agent monologues.
- **Outputs:** JSON profiles and questions.
- **How to run:**
  - Run directly with appropriate arguments.

#### `make_votes_dataset.py`
- **Purpose:** Creates a dataset of yes/no vote questions for training or analysis, using the LLM to generate questions from context.
- **Inputs:** CSV of vote contexts.
- **Outputs:** Pickle/JSON dataset of questions.
- **How to run:**
  - Run directly with appropriate arguments.

### 4. Agent Response Generation

#### `generate_trait_responses.py`
- **Purpose:** Generates agent responses to probing questions (belief, memory, personality) using the merged LLMs. Handles inference and response post-processing.
- **Inputs:** Merged model path, input JSON of questions.
- **Outputs:** JSON with generated responses.
- **How to run:**
  - Use `shell_scripts/eval_traits.sh` to run response generation for all agents and all trait types.

#### `generate_completion_responses.py`
- **Purpose:** Generates agent responses to completion prompts for further analysis and pairwise comparison.
- **Inputs:** Merged model path, speaker name, output file.
- **Outputs:** JSON with generated completion responses.
- **How to run:**
  - Use `shell_scripts/eval_model_completions.sh` to run for all agents.

#### `merge_agent_responses.py`
- **Purpose:** Merges per-agent response files into a single JSON for downstream judging and analysis.
- **Inputs:** Per-agent response JSON files.
- **Outputs:** Merged response JSON.
- **How to run:**
  - Called from `eval_model_completions.sh` after all agent responses are generated.

### 5. Judging and Alignment Evaluation

#### `judge_trait_alignment.py`
- **Purpose:** Judges the alignment of agent responses to their profiles/questions using a separate LLM. Produces alignment scores and explanations for belief, memory, and personality.
- **Inputs:** JSON with agent responses.
- **Outputs:** JSON with alignment scores.
- **How to run:**
  - Use `shell_scripts/eval_traits.sh` to run judging for all agents and all trait types.

#### `judge_completion_response.py`
- **Purpose:** Judges the quality, plausibility, and alignment of agent completion responses using a separate LLM. Produces scores and explanations.
- **Inputs:** Merged response JSON.
- **Outputs:** JSON with judgment results.
- **How to run:**
  - Called from `eval_model_completions.sh` after merging responses.

#### `pairwise_comparison.py`
- **Purpose:** Performs pairwise comparison of agent completion responses for further analysis.
- **Inputs:** Merged response JSON.
- **Outputs:** JSON with pairwise comparison results.
- **How to run:**
  - Called from `eval_model_completions.sh` after judging responses.

### 6. Vote Dataset Creation and Evaluation

#### `make_votes_dataset.py` (see above)
- **Purpose:** Creates a dataset of yes/no vote questions.

#### `evaluate_votes.py`
- **Purpose:** Evaluates agent models on yes/no vote prediction tasks. Loads each agent's model and predicts answers to vote questions.
- **Inputs:** JSON of reviewed questions, agent model paths.
- **Outputs:** JSON with predictions and accuracy summary.
- **How to run:**
  - Use `shell_scripts/eval_votes.sh` to run vote evaluation for all agents.

### 7. Completion Response Generation and Judging

#### `generate_completion_responses.py` (see above)
#### `merge_agent_responses.py` (see above)
#### `judge_completion_response.py` (see above)
#### `pairwise_comparison.py` (see above)
- **How to run:**
  - Use `shell_scripts/eval_model_completions.sh` to run the full completion response and judging pipeline.

### 8. Manual Review and Utilities

#### `review_questions_cli.py`
- **Purpose:** Interactive CLI to review and edit generated questions for each agent. Allows manual correction and curation of probing questions.
- **Inputs:** JSON of questions.
- **Outputs:** Updated JSON after review.
- **How to run:**
  - Run directly with appropriate arguments.

---

## Shell Scripts: Orchestration

All major steps are orchestrated via shell scripts in `llm_decisions/shell_scripts/`. These scripts:
- Set up environment variables (CUDA, FSDP, memory management, etc.)
- Select GPUs to use
- Run the appropriate Python scripts in the correct order
- Batch process all agents and hyperparameter settings as needed

**Key shell scripts:**
- `train_all_agents.sh`: Trains and merges LoRA adapters for all agents and hyperparameter settings.
- `eval_perplexity.sh`: Evaluates perplexity for all agent models.
- `eval_traits.sh`: Runs trait response generation and alignment judging for all agents.
- `eval_votes.sh`: Runs vote prediction evaluation for all agent models.
- `eval_model_completions.sh`: Runs completion response generation, merging, judging, and pairwise comparison for all agents.

**Usage:**
- Run each shell script in order as needed for your workflow. The scripts are self-documenting and can be modified to change which agents, models, or datasets are processed.

---

## Tips and Troubleshooting

- **Model Loading:** Loading Llama 70B (full precision or quantized) can take 10–30+ minutes, especially on first load. Ensure your model files are on a fast local SSD and you have sufficient CPU RAM.
- **GPU Selection:** Use `CUDA_VISIBLE_DEVICES` in the shell scripts to select which GPUs to use. Make sure no other jobs are using the same GPUs.
- **Distributed Training:** For full-precision Llama 70B, FSDP or DeepSpeed is required. For quantized LoRA/QLoRA, single-GPU is possible.
- **Environment Variables:** Some scripts require `ACCELERATE_USE_FSDP=1` and `FSDP_CPU_RAM_EFFICIENT_LOADING=1` for efficient sharding and memory usage.
- **Editing Questions:** Use the CLI tool to review and edit probing questions before running evaluations.
- **Logs and Outputs:** All results, logs, and outputs are saved in the `trained_models/` and `results/` directories for later analysis.
- **Script Names:** Always check the shell scripts for the current script names and order of execution, as these reflect the latest workflow.

---

For more details on each script, see the comments at the top of each Python file or the shell scripts in `llm_decisions/shell_scripts/`.