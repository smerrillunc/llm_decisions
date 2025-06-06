#!/usr/bin/env python
# coding: utf-8
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

import argparse
import json
import logging
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"
os.environ["PT_DISABLE_DTORCH"] = "1"

import random
from typing import List, Tuple

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

from trl import SFTTrainer
import wandb
from collections import Counter
import re
import torch
import numpy as np
from accelerate import Accelerator
from transformers import DataCollatorForSeq2Seq

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

    accelerator = Accelerator()

    # Initialize wandb if project specified
    if args.wandb_project:
        if accelerator.is_main_process:
            wandb.init(project=args.wandb_project, name=args.wandb_run_name)
            
    model_name = "unsloth/Llama-3.3-70B-Instruct"
    agent_name = args.agent_name.replace(' ', '').lower()
    output_dir = os.path.join(args.output_dir , args.wandb_run_name)

    # Dataset preparation
    train_data, test_data, train_completion_data = train_test_split('judyle')

    tmp = [{"text": text} for text in train_data]
    train_data = Dataset.from_list(tmp)
    
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
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        device_map='auto', 
        load_in_4bit=False,
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

    # Instantiate the data collator first
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    # Prepare model, dataset, and collator all together
    model, train_data, data_collator = accelerator.prepare(model, train_data, data_collator)


    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_data,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        data_collator = data_collator,  # Use prepared collator here!
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            #deepspeed="/work/users/s/m/smerrill/LLM/ds_config.json",
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
