# LLM Decisions Pipeline

This repository provides a modular pipeline for training, merging, evaluating, and analyzing large language models (LLMs) as personalized conversational agents. The workflow supports multi-GPU training, quantized inference, and downstream analysis of model outputs, including belief/personality extraction, question generation, and alignment evaluation.

---

## Table of Contents
- [Environment Setup](#environment-setup)
- [Workflow Overview](#workflow-overview)
- [Scripts and Shell Files](#scripts-and-shell-files)
  - [Training: train_agent_llm.py](#training-train_agent_llmpy)
  - [Merging Adapters: merge_lora_adapters.py](#merging-adapters-merge_lora_adapterspy)
  - [Perplexity Evaluation: evaluate_agent_perplexity.py](#perplexity-evaluation-evaluate_agent_perplexitypy)
  - [Votes Dataset Creation: make_votes_dataset.py](#votes-dataset-creation-make_votes_datasetpy)
  - [Agent Profile Building: build_agent_profiles.py](#agent-profile-building-build_agent_profilespy)
  - [Agent Response Generation: generate_agent_responses.py](#agent-response-generation-generate_agent_responsespy)
  - [Alignment Judging: judge_response_alignment.py](#alignment-judging-judge_response_alignmentpy)
  - [Votes Evaluation: evaluate_votes.py](#votes-evaluation-evaluate_votespy)
  - [Question Review CLI: review_questions_cli.py](#question-review-cli-review_questions_clip)
- [Shell Scripts](#shell-scripts)
- [Tips and Troubleshooting](#tips-and-troubleshooting)

---

## Environment Setup

1. **Install dependencies** (Python 3.10+ recommended):
   - See `requirements.txt` for Python package requirements.
2. **Set up CUDA and GPUs** as needed for your hardware.
3. **Configure Accelerate**:
   - Run `accelerate config` and set up your distributed environment as needed.

---

## Workflow Overview

1. **Train** a personalized LLM for each agent using LoRA/QLoRA adapters.
2. **Merge** adapters into the base model for efficient inference.
3. **Evaluate** perplexity of each agent model on held-out data.
4. **Extract** beliefs, memories, and personality traits from agent monologues using the LLM.
5. **Generate** probing questions for each agent based on extracted traits.
6. **Ask** the agent LLM these questions and record responses.
7. **Judge** the alignment of agent responses using a separate judge LLM.
8. **(Optional) Review** and edit generated questions via CLI.

---

## Scripts and Shell Files

All scripts can be run using the provided shell scripts in `llm_decisions/shell_scripts/`. These shell scripts handle environment variables, GPU selection, and batch processing for you.

### Training: `train_agent_llm.py`
Train a LoRA/QLoRA model for a specific agent using multi-GPU and FSDP. Handles dataset loading, tokenizer/model setup, LoRA config, and training loop.
- **Shell script:** `shell_scripts/train_all_agents.sh`
- **Inputs:** Agent name, config YAML, dataset path
- **Outputs:** Trained LoRA adapter and checkpoints in `trained_models/`

### Merging Adapters: `merge_lora_adapters.py`
Merge LoRA/PEFT adapters into the base model for efficient inference. Produces a "merged" model directory for downstream evaluation and inference.
- **Shell script:** Called from `train_all_agents.sh` after training
- **Inputs:** Output directory from training
- **Outputs:** Merged model in `merged/` subdirectory

### Perplexity Evaluation: `evaluate_agent_perplexity.py`
Compute perplexity for each agent's test set using the merged model. Uses Accelerate for distributed evaluation.
- **Shell script:** `shell_scripts/eval_perplexity.sh`
- **Inputs:** Merged model path, agent name
- **Outputs:** Perplexity results CSV per agent

### Votes Dataset Creation: `make_votes_dataset.py`
Create a dataset of yes/no vote questions for training or analysis, using the LLM to generate questions from context.
- **Run directly:**
  ```bash
  python llm_decisions/make_votes_dataset.py --model <model_name> --csv_path <votes.csv> --output_path <output.pkl>
  ```
- **Inputs:** CSV of vote contexts
- **Outputs:** Pickle/JSON dataset of questions

### Agent Profile Building: `build_agent_profiles.py`
Generate detailed personality profiles and interview questions for each agent from their monologues.
- **Run directly:**
  ```bash
  python llm_decisions/build_agent_profiles.py --input <monologues.pkl> --output_dir <results_dir>
  ```
- **Inputs:** Pickle of agent monologues
- **Outputs:** JSON profiles and questions

### Agent Response Generation: `generate_agent_responses.py`
Generate agent responses to probing questions using the merged LLMs. Handles inference and response post-processing.
- **Shell script:** `shell_scripts/eval_traits.sh`
- **Inputs:** Merged model path, input JSON of questions
- **Outputs:** JSON with generated responses

### Alignment Judging: `judge_response_alignment.py`
Judge the alignment of agent responses to their profiles/questions using a separate LLM. Produces alignment scores and explanations.
- **Shell script:** `shell_scripts/eval_traits.sh` (final step)
- **Inputs:** JSON with agent responses
- **Outputs:** JSON with alignment scores

### Votes Evaluation: `evaluate_votes.py`
Evaluate agent models on yes/no vote prediction tasks. Loads each agent's model and predicts answers to vote questions.
- **Shell script:** `shell_scripts/eval_votes.sh`
- **Inputs:** JSON of reviewed questions, agent model paths
- **Outputs:** JSON with predictions and accuracy summary

### Question Review CLI: `review_questions_cli.py`
Interactive CLI to review and edit generated questions for each agent. Allows manual correction and curation of probing questions.
- **Run directly:**
  ```bash
  python llm_decisions/tools/review_questions_cli.py --file <questions.json> --speaker <agent_name>
  ```
- **Inputs:** JSON of questions
- **Outputs:** Updated JSON after review

---

## Shell Scripts

- **train_all_agents.sh**: Trains and merges LoRA adapters for all agents and hyperparameter settings.
- **eval_perplexity.sh**: Evaluates perplexity for all agent models.
- **eval_traits.sh**: Runs response generation and alignment judging for all agents.
- **eval_votes.sh**: Runs vote prediction evaluation for all agent models.

Each shell script sets up the environment, selects GPUs, and runs the appropriate Python scripts in batch mode. You can modify these scripts to change which agents, models, or datasets are processed.

---

## Tips and Troubleshooting

- **Model Loading:** Loading Llama 70B (full precision or quantized) can take 10–30+ minutes, especially on first load. Ensure your model files are on a fast local SSD and you have sufficient CPU RAM.
- **GPU Selection:** Use `CUDA_VISIBLE_DEVICES` in the shell scripts to select which GPUs to use. Make sure no other jobs are using the same GPUs.
- **Distributed Training:** For full-precision Llama 70B, FSDP or DeepSpeed is required. For quantized LoRA/QLoRA, single-GPU is possible.
- **Environment Variables:** Some scripts require `ACCELERATE_USE_FSDP=1` and `FSDP_CPU_RAM_EFFICIENT_LOADING=1` for efficient sharding and memory usage.
- **Editing Questions:** Use the CLI tool to review and edit probing questions before running evaluations.
- **Logs and Outputs:** All results, logs, and outputs are saved in the `trained_models/` and `results/` directories for later analysis.

---

For more details on each script, see the comments at the top of each Python file or the shell scripts in `llm_decisions/shell_scripts/`.