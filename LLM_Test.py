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


def load_chat_dataset(input_path: str) -> List[dict]:
    """
    Load a chat-style message dataset from a JSON or JSONL file.

    Args:
        input_path: Path to the dataset file.

    Returns:
        List of message dicts.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        if input_path.endswith(".jsonl"):
            data = [json.loads(line) for line in f]
        else:  # Assume .json
            data = json.load(f)

    logger.info(f"Loaded dataset from {input_path} with {len(data)} examples")
    return data


def train_test_split(
    member: str,
    test_size: float = 0.2,
    seed: int = 42,
    data_path: str = "/work/users/s/m/smerrill/Albemarle/dataset",
) -> Tuple[List[dict], List[dict]]:
    """
    Splits the dataset into training and test sets, including synthetic data in training.

    Args:
        member: Board member name to select dataset files.
        test_size: Fraction of real data for testing (between 0 and 1).
        seed: Random seed for reproducibility.
        data_path: Base path for dataset files.

    Returns:
        Tuple of (train_data, test_data).
    """
    real_data, synth_data = [], []

    member_files = {
        "kateacuff": ("kateacuff.txt", "synth_kateacuff.txt"),
        "ellenosborne": ("ellenosborne.txt", "synth_ellenosborne.txt"),
        "grahampaige": ("grahampaige.txt", "synth_grahampaige.txt"),
        "judyle": ("judyle.txt", "synth_judyle.txt"),
        "katrinacallsen": ("katrinacallsen.txt", None),
        "davidoberg": ("davidoberg.txt", None),
        "jonnoalcaro": ("jonnoalcaro.txt", None),
    }

    if member not in member_files:
        raise ValueError(f"Unknown member: {member}")

    real_file, synth_file = member_files[member]

    real_data = load_chat_dataset(os.path.join(data_path, real_file))
    if synth_file:
        synth_data = load_chat_dataset(os.path.join(data_path, synth_file))

    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")

    random.seed(seed)
    shuffled_real = real_data.copy()
    random.shuffle(shuffled_real)

    split_index = int(len(shuffled_real) * (1 - test_size))
    train_data = shuffled_real[:split_index] + synth_data
    test_data = shuffled_real[split_index:]

    return train_data, test_data


def create_formatted_dataset(train_data, target_speaker):
    def format_chat(prompt_text, target_speaker):
        lines = prompt_text.split('\n')
        text = ''
        current_block_type = None
        block_lines = []

        def flush_block():
            nonlocal text, block_lines, current_block_type
            if not block_lines:
                return
            if current_block_type == 'assistant':
                text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                text += "<|start_header_id|>user<|end_header_id|>\n\n"
            text += '\n'.join(block_lines) + '<|eot_id|>\n\n'
            block_lines.clear()

        for line in lines:
            if not line.strip():
                continue
            speaker_name = line.split(':')[0].strip()
            block_type = 'assistant' if speaker_name == target_speaker else 'user'

            if block_type != current_block_type:
                flush_block()
                current_block_type = block_type

            block_lines.append(line)

        flush_block()
        return text

    def format_chat_with_response(data_item, target_speaker):
        prompt_text = data_item['prompt']
        response_text = data_item['response']

        formatted_prompt = format_chat(prompt_text, target_speaker)
        prompt_only = formatted_prompt + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        response_only = f"{response_text}<|eot_id|>"
        prompt_response = prompt_only + response_only

        return prompt_only, response_only, prompt_response

    dataset_input = []
    for item in train_data:
        prompt_only, response_only, prompt_response = format_chat_with_response(item, target_speaker)
        dataset_input.append({
            "prompt": prompt_only,
            "response": response_only,
            "text": prompt_response
        })

    formatted_dataset = Dataset.from_list(dataset_input)
    return formatted_dataset


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
        reference_texts.append(example['response'].strip())

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
    train_data, test_data = train_test_split(agent_name, data_path=args.data_path)
    train_data = create_formatted_dataset(train_data, agent_name)
    test_data = create_formatted_dataset(test_data, agent_name)

    print("--------------------")
    print("Train Format")
    print(train_data['text'][0])
    print("--------------------")

    print("--------------------")
    print("Test Format")
    print(test_data['text'][0])
    print("--------------------")

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
