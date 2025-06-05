#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import logging
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"

import random
from typing import List, Tuple

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from peft import PeftModel

from trl import SFTTrainer
import wandb
from collections import Counter
import re
import torch
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_test_split(member: str, test_size: float = 0.2, seed: int = 42, data_path: str = '/work/users/s/m/smerrill/Albemarle/dataset') -> Tuple[List[dict], List[dict]]:
    """
    Splits the dataset into training and test sets. Synthetic data is always added to the training set.

    Parameters:
    - member: The name identifier for the board member.
    - test_size: Proportion of the real (non-synthetic) data to include in the test split.
    - seed: Random seed for reproducibility.
    - data_path: Base directory for the dataset files.

    Returns:
    - A tuple (train_data, test_data)
    """
    real_data, synth_data = [], []

    if member == 'kateacuff':
        real_data = np.load(os.path.join(data_path, 'kateacuff_train.npy'))
        synth_data = np.load(os.path.join(data_path, 'synth_kateacuff.npy'))
        test_data = np.load(os.path.join(data_path, 'kateacuff_test.npy'), allow_pickle=True)
        train_completion_data = np.load(os.path.join(data_path, 'kateacuff_train_completion.npy'), allow_pickle=True)

        
    elif member == 'ellenosborne':
        real_data = np.load(os.path.join(data_path, 'ellenosborne_train.npy'))
        synth_data = np.load(os.path.join(data_path, 'synth_ellenosborne.npy'))
        test_data = np.load(os.path.join(data_path, 'ellenosborne_test.npy'), allow_pickle=True)
        train_completion_data = np.load(os.path.join(data_path, 'ellenosborne_train_completion.npy'), allow_pickle=True)
        
    elif member == 'grahampaige':
        real_data = np.load(os.path.join(data_path, 'grahampaige_train.npy'))
        synth_data = np.load(os.path.join(data_path, 'synth_grahampaige.npy'))
        test_data = np.load(os.path.join(data_path, 'grahampaige_test.npy'), allow_pickle=True)
        train_completion_data = np.load(os.path.join(data_path, 'grahampaige_train_completion.npy'), allow_pickle=True)                             
        
    elif member == 'judyle':
        real_data = np.load(os.path.join(data_path, 'judyle_train.npy'))
        synth_data = np.load(os.path.join(data_path, 'synth_judyle.npy'))
        test_data = np.load(os.path.join(data_path, 'judyle_test.npy'), allow_pickle=True)
        train_completion_data = np.load(os.path.join(data_path, 'judyle_train_completion.npy'), allow_pickle=True)
        
    elif member == 'katrinacallsen':
        real_data = np.load(os.path.join(data_path, 'katrinacallsen_train.npy'))
        test_data = np.load(os.path.join(data_path, 'katrinacallsen_test.npy'), allow_pickle=True)
        train_completion_data = np.load(os.path.join(data_path, 'katrinacallsen_train_completion.npy'), allow_pickle=True)
        
    elif member == 'davidoberg':
        real_data = np.load(os.path.join(data_path, 'davidoberg_train.npy'))
        test_data = np.load(os.path.join(data_path, 'davidoberg_test.npy'), allow_pickle=True)
        train_completion_data = np.load(os.path.join(data_path, 'davidoberg_train_completion.npy'), allow_pickle=True)
        
    elif member == 'jonnoalcaro':
        real_data = np.load(os.path.join(data_path, 'jonnoalcaro_train.npy'))
        test_data = np.load(os.path.join(data_path, 'jonnoalcaro_test.npy'), allow_pickle=True)
        train_completion_data = np.load(os.path.join(data_path, 'jonnoalcaro_train_completion.npy'), allow_pickle=True)
        
    else:
        raise ValueError(f"Unknown member: {member}")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be a float between 0 and 1.")

    train_data = list(real_data) + list(synth_data)
    return train_data, test_data, train_completion_data


def generate_texts(model, tokenizer, dataset, max_new_tokens=128, temperature=1.5, top_p=0.9):
    generated_texts = []
    reference_texts = []

    for example in tqdm(dataset):
        input_text = example['prompt']

        inputs = tokenizer(
            input_text,
            return_tensors="pt"
        ).to(model.device)

        input_length = inputs.input_ids.shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            top_p=top_p
        )

        generated_tokens = outputs[0]
        new_tokens = generated_tokens[input_length:]

        decoded_generation = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(decoded_generation)
        generated_texts.append(decoded_generation.strip())
        reference_texts.append(example['completion'].strip())

    return generated_texts, reference_texts



def parse_args():
    parser = argparse.ArgumentParser(description="Train an agent language model.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to test (e.g., paige, acuff, etc.)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to saved model and tokenizer",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/work/users/s/m/smerrill/Albemarle/dataset",
        help="Base path for datasets",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="LLM_Decisions",
        help="Wandb project name for logging (optional)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize wandb if project specified
    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
    agent_name = args.dataset_name.replace(' ', '').lower()

    # Dataset preparation
    _, test_data, train_data = train_test_split(agent_name, data_path=args.data_path)


    print("--------------------")
    print("Loading Model from: ", args.model_path)
    print("--------------------")

    max_seq_length = 1000
    # Load tokenizer from training path (with added tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Load base model (not merged)
    model, _ = FastLanguageModel.from_pretrained(
        model_name = model_name,  # or whatever your original base was
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    # Resize base model to match tokenizer
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # Now load the adapter
    model = PeftModel.from_pretrained(model, args.model_path)


    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    print("--------------------")
    print("Getting Train Generations")
    print("--------------------")
    train_generations, train_references = generate_texts(model, tokenizer, train_data, max_new_tokens=128, temperature=1.5, top_p=0.9)

    print("--------------------")
    save_path = os.path.join(args.model_path, f'{agent_name}_train_generations.npy')
    print(f"Saving Train Generations to {save_path}")
    print("--------------------")
    np.save(save_path, train_generations)

    print("--------------------")
    print("Getting Test Generations")
    print("--------------------")
    test_generations, test_references = generate_texts(model, tokenizer, test_data, max_new_tokens=128, temperature=1.5, top_p=0.9)
        
    print("--------------------")
    save_path = os.path.join(args.model_path, f'{agent_name}_test_generations.npy')
    print(f"Saving Test Generations to {save_path}")
    print("--------------------")

    np.save(save_path, test_generations)

    print("Script Complete")

    # Finish wandb run
    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
