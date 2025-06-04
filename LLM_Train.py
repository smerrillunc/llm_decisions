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
from trl import SFTTrainer
import wandb
from collections import Counter
import re
import torch

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


def create_formatted_dataset(train_data, target_speaker, system_message=None):
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

    def format_chat_with_response(data_item, target_speaker, system_message):
        prompt_text = data_item['prompt']
        response_text = data_item['response']
                
        formatted_prompt = format_chat(prompt_text, target_speaker)
        
        if system_message:
            system_message = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system_message + '<|eot_id|>\n\n'
            prompt_only = system_message + formatted_prompt + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            prompt_only = "<|begin_of_text|>" + formatted_prompt + "<|start_header_id|>assistant<|end_header_id|>\n\n"
            
        response_only = f"{response_text}<|eot_id|>"
        prompt_response = prompt_only + response_only

        return prompt_only, response_only, prompt_response

    dataset_input = []
    for item in train_data:
        prompt_only, response_only, prompt_response = format_chat_with_response(item, target_speaker, system_message)
        dataset_input.append({
            "prompt": prompt_only,
            "response": response_only,
            "text": prompt_response
        })

    formatted_dataset = Dataset.from_list(dataset_input)
    return formatted_dataset
    


def parse_args():
    parser = argparse.ArgumentParser(description="Train an agent language model.")
    parser.add_argument(
        "--agent_name",
        type=str,
        required=True,
        help="Name of the agent to train (e.g., paige, acuff, etc.)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train for",
    )

    parser.add_argument(
        "--factors",
        type=int,
        default=4,
        help="LORA Factors",
    )
    
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="LLM Max Sequence length",
    )
    parser.add_argument(
        "--weight_decay",
        type=int,
        default=0.2,
        help="weight_decay",
    )

    parser.add_argument(
        "--learning_rate",
        type=int,
        default=1e-5,
        help="learning_rate",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/work/users/s/m/smerrill/Albemarle/dataset",
        help="Base path for datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/work/users/s/m/smerrill/Albemarle/trained_models",
        help="Directory to save the trained model and checkpoints",
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

def extract_speakers(text):
    return re.findall(r"^(?:speaker \d+|[a-zA-Z0-9_]+):", text, flags=re.MULTILINE)


def main():
    args = parse_args()

    # Initialize wandb if project specified
    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
    agent_name = args.agent_name.replace(' ', '').lower()
    output_dir = os.path.join(args.output_dir , args.wandb_run_name)

    # Dataset preparation
    train_data, _ = train_test_split(agent_name, data_path=args.data_path)
    train_data = create_formatted_dataset(train_data, agent_name)

    print("--------------------")
    print("Train Format")
    print(train_data['text'][0])
    print("--------------------")

    print("--------------------")
    print("Adding Special Tokens to Tokenizer")
    speaker_counter = Counter()
    for sample in train_data:
        speakers = extract_speakers(sample["text"])
        speaker_counter.update(speakers)

    speaker_tokens = list(speaker_counter.keys())

    # Add special tokens
    special_tokens = {
        "additional_special_tokens": speaker_tokens + [
            "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"
        ]
    }
    print("--------------------")


    print("--------------------")
    print("Loading Model")
    # Model Loding
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        device_map=None, 
        load_in_4bit=True,
    )

    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.factors,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )


    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_data,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            num_train_epochs = args.epochs,
            learning_rate = args.learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = args.weight_decay,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
            report_to = "wandb", # Use this for WandB etc
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    print("--------------------")
    print("Training Start")
    trainer_stats = trainer.train()

    print("--------------------")
    print("Training finished!")
    print("--------------------")
    
    model.save_pretrained(output_dir)  # Local saving
    print(f"Model saved to {output_dir}")
    
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")

    # Finish wandb run
    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
