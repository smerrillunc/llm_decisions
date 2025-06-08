#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import json
import logging
import re
import random
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from accelerate import Accelerator
from trl import SFTTrainer
import wandb

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

# --- Environment configuration ---
os.environ.update({
    "NCCL_P2P_DISABLE": "1",
    "NCCL_DEBUG": "INFO",
    "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
    "CUDA_LAUNCH_BLOCKING": "1",
    "TORCH_USE_CUDA_DSA": "1",
    "TORCH_DISTRIBUTED_USE_DTENSOR": "0",
    "TORCH_DIST_DDP_SHARDING": "0",
    "ACCELERATE_USE_TP": "false",
    "PYTORCH_ENABLE_DISTRIBUTED": "1",
    "TORCH_CPP_LOG_LEVEL": "INFO",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TORCH_COMPILE_DISABLE": "1",
    "TORCHINDUCTOR_DISABLE": "1",
})

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an agent language model.")
    parser.add_argument("--agent_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--factors", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=850)
    parser.add_argument("--weight_decay", type=float, default=0.2)  # Fixed type
    parser.add_argument("--learning_rate", type=float, default=1e-5)  # Fixed type
    parser.add_argument("--data_path", type=str, default="/playpen-ssd/smerrill/dataset")
    parser.add_argument("--output_dir", type=str, default="/playpen-ssd/smerrill/trained_models")
    parser.add_argument("--wandb_project", type=str, default="LLM_Decisions")
    parser.add_argument("--wandb_run_name", type=str, default='test')
    return parser.parse_args()


def train_test_split(member: str, data_path: str = '/playpen-ssd/smerrill/dataset') -> Tuple[List[dict], List[dict], List[dict]]:
    def load_data(file_name):
        path = os.path.join(data_path, file_name)
        return np.load(path, allow_pickle=True) if os.path.exists(path) else []

    real_data = load_data(f"{member}_train.npy")
    synth_data = load_data(f"synth_{member}.npy")
    test_data = load_data(f"{member}_test.npy")
    train_completion_data = load_data(f"{member}_train_completion.npy")

    return list(real_data) + list(synth_data), list(test_data), list(train_completion_data)


def extract_speakers(text):
    return re.findall(r"^(?:speaker \d+|[a-zA-Z0-9_]+):", text, flags=re.MULTILINE)


def main():
    args = parse_args()
    accelerator = Accelerator()

    if args.wandb_project and accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    model_name = "unsloth/Llama-3.3-70B-Instruct"
    agent_name = args.agent_name.replace(' ', '').lower()
    output_dir = os.path.join(args.output_dir, args.wandb_run_name)

    train_data, test_data, train_completion_data = train_test_split(agent_name, args.data_path)
    train_data = Dataset.from_list([{"text": text} for text in train_data])

    # Speaker token extraction
    speaker_counter = Counter()
    for sample in train_data:
        speaker_counter.update(extract_speakers(sample["text"]))
    speaker_tokens = list(speaker_counter.keys())

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({
        "additional_special_tokens": speaker_tokens + ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
    })
    tokenizer = accelerator.prepare(tokenizer)
    accelerator.wait_for_everyone()

    # Model loading
    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        load_in_4bit=False,
        device_map=None,
    )

    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.factors,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    model.gradient_checkpointing_enable()

    model = accelerator.prepare(model)
    train_data = accelerator.prepare(train_data)
    data_collator = accelerator.prepare(data_collator)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=data_collator,
        dataset_num_proc=1,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="wandb" if args.wandb_project else [],
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # Train
    print("--------------------\nTraining Start\n--------------------")
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    accelerator.free_memory()
    trainer.train()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model and tokenizer saved to {output_dir}")

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
