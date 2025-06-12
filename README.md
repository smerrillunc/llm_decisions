# LLM Decisions Pipeline

This repository provides a modular pipeline for training, merging, evaluating, and analyzing large language models (LLMs) as personalized conversational agents. The workflow supports multi-GPU training, quantized inference, and downstream analysis of model outputs, including belief/personality extraction, question generation, and alignment evaluation.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Workflow Overview](#workflow-overview)
- [Scripts Overview](#scripts-overview)
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

## Scripts Overview

### 1. Training: `train_agent_llm.py` (was `LLM_Train.py`)

Train a LoRA/QLoRA model for a specific agent using multi-GPU and FSDP.

**Example command:**
```bash
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
accelerate launch --num_processes 4 train_agent_llm.py --config configs/llamma_3_70b.yaml --agent_name kateacuff
```

---

### 2. Merging Adapters: `merge_lora_adapters.py` (was `merge_peft.py`)

Merge LoRA/PEFT adapters into the base model for efficient inference.

**Example command:**
```bash
python llm_decisions/tools/merge_lora_adapters.py
```

---

### 3. Perplexity Evaluation: `evaluate_agent_perplexity.py` (was `LLM_Test.py`)

Compute perplexity for each agent's test set using the merged model.

**Example command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 llm_decisions/evaluate_agent_perplexity.py --merged_path /path/to/agent/merged
```

---

### 4. Extract Agent Traits: `extract_agent_traits.py` (was `run_monologue_analysis.py`)

Extract beliefs, memories, and personality traits from agent monologues using the LLM.

**Example command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 llm_decisions/extract_agent_traits.py --model /path/to/merged --input /path/to/monologues.pkl --output_dir ./results --single_speaker ellenosborne
```

---

### 5. Generate Agent Responses: `generate_agent_responses.py` (was `askQuestions.py`)

Ask the agent LLM probing questions (generated from traits) and record responses.

**Example command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 llm_decisions/generate_agent_responses.py --model-path /path/to/merged --input-file ./results/belief_results.json --output-file ./results/belief_results.json
```

---

### 6. Judge Response Alignment: `judge_response_alignment.py` (was `evaluate_alignment.py`)

Use a judge LLM to score the alignment of agent responses with their extracted beliefs.

**Example command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 llm_decisions/judge_response_alignment.py --data_file ./results/belief_results.json --model_name meta-llama/Meta-Llama-3-70B-Instruct
```

---

### 7. Question Review CLI: `review_questions_cli.py` (was `question_CLI.py`)

Interactive CLI to review and edit generated questions for each speaker.

**Example command:**
```bash
python llm_decisions/tools/review_questions_cli.py --file ./results/personality_results.json --speaker ellenosborne
```

---

### 8. Batch Scripts

- **Train All Agents:**  
  `train_all_agents.sh`  
  Automate training for all agents in sequence.

- **Evaluate All Agents:**  
  `evaluate_all_agents.sh` (was `test_all_agents.sh`)  
  Batch evaluation for all agents' perplexity.

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
