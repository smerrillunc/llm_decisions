#!/usr/bin/env python
# coding: utf-8

import os
import torch
import wandb
import evaluate
from dataclasses import dataclass, field
from transformers import AutoTokenizer, HfArgumentParser
from peft import AutoPeftModelForCausalLM

from utils import train_test_split, compute_perplexity, compute_metrics

@dataclass
class ScriptArguments:
    model_path: str = field(
        default=None, metadata={"help": "Path to saved model and tokenizer"}
    )
    data_path: str = field(
        default="/work/users/s/m/smerrill/Albemarle/dataset",
        metadata={"help": "Base path for datasets"}
    )
    wandb_project: str = field(
        default="LLM_Decisions",
        metadata={"help": "Wandb project name for logging (optional)"}
    )
    wandb_run_name: str = field(
        default=None,
        metadata={"help": "Wandb run name (optional)"}
    )

def main():
    # Parse CLI args into an instance of ScriptArguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Initialize wandb if project specified
    if script_args.wandb_project:
        wandb.init(project=script_args.wandb_project, name=script_args.wandb_run_name)

    # Load model
    model = AutoPeftModelForCausalLM.from_pretrained(
        script_args.model_path,
        torch_dtype=torch.float16,
        quantization_config={"load_in_4bit": True},
        device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)

    # Evaluation datasets
    datasets = [
        'kateacuff',
        'ellenosborne',
        'grahampaige',
        'judyle',
        'katrinacallsen',
        'davidoberg',
        'jonnoalcaro'
    ]

    print("Loading Metrics")
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    for dataset in datasets:
        print(f'Model: {script_args.model_path}, Dataset: {dataset}')
        _, test_data, train_completion_data = train_test_split(dataset)

        print("Computing Train Perplexity")
        perplexity_train, generated_texts, reference_texts = compute_perplexity(
            model,
            train_completion_data,
            tokenizer,
            max_length=1024,
            verbose=False
        )

        bleu_score, rouge_score, bertscore_result, avg_bertscore_f1 = compute_metrics(
            generated_texts, reference_texts, bleu, rouge, bertscore
        )

        print("Computing Test Perplexity")
        perplexity_test = compute_perplexity(
            model,
            test_data, 
            tokenizer,
            max_length=1024,
            verbose=False
        )

        print(f"Train PPL: {perplexity_train:.2f}, Test PPL: {perplexity_test:.2f}")
        print(f"BLEU: {bleu_score}, ROUGE: {rouge_score}, BERTScore-F1: {avg_bertscore_f1:.4f}")


if __name__ == "__main__":
    main()
