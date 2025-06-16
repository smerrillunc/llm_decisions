# LLM Decisions Pipeline

This repository provides a modular pipeline for training, merging, evaluating, and analyzing large language models (LLMs) as personalized conversational agents. The workflow supports multi-GPU training, quantized inference, and downstream analysis of model outputs, including belief/personality extraction, question generation, and alignment evaluation.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Workflow Overview](#workflow-overview)
- [Scripts Overview](#scripts-overview)
  - [Training: train_agent_llm.py](#1-training-train_agent_llmpy)
  - [Merging Adapters: merge_lora_adapters.py](#2-merging-adapters-merge_lora_adapterspy)
  - [Perplexity Evaluation: evaluate_agent_perplexity.py](#3-perplexity-evaluation-evaluate_agent_perplexitypy)
  - [Votes Evaluation/Alignment](#votes-evaluationalignment)
  - [Extract Agent Traits: extract_agent_traits.py](#4-extract-agent-traits-extract_agent_traitspy)
  - [Generate Agent Responses: generate_agent_responses.py](#5-generate-agent-responses-generate_agent_responsespy)
  - [Judge Response Alignment: judge_response_alignment.py](#6-judge-response-alignment-judge_response_alignmentpy)
  - [Question Review CLI: review_questions_cli.py](#7-question-review-cli-review_questions_cli...
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

### 1. Training: `train_agent_llm.py`

Train a LoRA/QLoRA model for a specific agent using multi-GPU and FSDP.

**Example command:**
```bash
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
accelerate launch --num_processes 4 train_agent_llm.py --config configs/llamma_3_70b.yaml --agent_name kateacuff
```

---

### 2. Merging Adapters: `merge_lora_adapters.py`

Merge LoRA/PEFT adapters into the base model for efficient inference.

**Example command:**
```bash
python llm_decisions/tools/merge_lora_adapters.py
```

---

### 3. Perplexity Evaluation: `evaluate_agent_perplexity.py`

Compute perplexity for each agent's test set using the merged model.

**Example command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 llm_decisions/evaluate_agent_perplexity.py --merged_path /path/to/agent/merged
```

---

### 4. Votes Evaluation/Alignment

This section includes scripts for creating a votes dataset, reviewing votes, and evaluating votes on agent responses.

---

#### 4.1 Create Votes Dataset: `make_votes_dataset.py`

Create a dataset of votes for training or analysis.

**Example command:**
```bash
python llm_decisions/tools/make_votes_dataset.py --input ./results/alignment_results.json --output ./results/votes_dataset.json
```

---

#### 4.2 Review Votes CLI: `review_votes_cli.py`

Interactive CLI to review and edit votes on agent responses.

**Example command:**
```bash
python llm_decisions/tools/review_votes_cli.py --file ./results/votes_results.json --speaker ellenosborne
```

---

#### 4.3 Evaluate Votes: `evaluate_votes.py`

Evaluate the votes on agent responses to probing questions.

**Example command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 llm_decisions/evaluate_votes.py --input ./results/alignment_results.json --output ./results/votes_evaluation.json
```

---

### 5. Personality, Beliefs and Memory Evaluation

#### 5.1 Extract Agent Traits: `extract_agent_traits.py`

Extract beliefs, memories, and personality traits from agent monologues using the LLM.

**Example command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 llm_decisions/extract_agent_traits.py --model /path/to/merged --input /path/to/monologues.pkl --output_dir ./results --single_speaker ellenosborne
```

---

#### 5.2. Generate Agent Responses: `generate_agent_responses.py`

Ask the agent LLM probing questions (generated from traits) and record responses.

**Example command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 llm_decisions/generate_agent_responses.py --model-path /path/to/merged --input-file ./results/belief_results.json --output-file ./results/belief_results.json
```

---

### 5.3 Question Review CLI: `review_questions_cli.py`

Interactive CLI to review and edit generated questions for each speaker.

**Example command:**
```bash
python llm_decisions/tools/review_questions_cli.py --file ./results/personality_results.json --speaker ellenosborne
```

---

#### 5.4. Judge Response Alignment: `judge_response_alignment.py`

Use a judge LLM to score the alignment of agent responses with their extracted beliefs.

**Example command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 llm_decisions/judge_response_alignment.py --data_file ./results/belief_results.json --model_name meta-llama/Meta-Llama-3-70B-Instruct
```

---

### 6. Batch Scripts

- **Train All Agents:**  
  `train_all_agents.sh`  
  Automate training for all agents in sequence.

- **Evaluate All Agents:**  
  `evaluate_all_agents.sh` (was `test_all_agents.sh`)  
  Batch evaluation for all agents' perplexity.

---