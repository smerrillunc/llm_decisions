#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,7,2,3,4,5,6 
AGENT_MODELS_JSON="/playpen-ssd/smerrill/llm_decisions/configs/models.json"
SCRIPT=/playpen-ssd/smerrill/llm_decisions/evaluate_votes.py

accelerate launch --main_process_port 0 --num_processes 1 "$SCRIPT" --reviewed_json /playpen-ssd/smerrill/dataset/reviewed_questions.json --output /playpen-ssd/smerrill/llm_decisions/results/reviewed_questions.json --agent_models "$AGENT_MODELS_JSON"