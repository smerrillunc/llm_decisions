#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import logging
import os
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


def convert_to_chat_format(data: List[dict]) -> List[dict]:
    """
    Convert dataset entries to chat format with alternating roles user/assistant.

    Args:
        data: List of dicts with keys 'prompt' and 'response'.

    Returns:
        List of dicts with 'role' and 'content'.
    """
    result = []
    for item in data:
        result.append({"role": "user", "content": item["prompt"]})
        result.append({"role": "assistant", "content": item["response"]})
    return result


def combine_conversations(dataset: List[dict]) -> Dataset:
    """
    Combine messages into conversation pairs (user + assistant).

    Args:
        dataset: List of dicts with keys 'role' and 'content'.

    Returns:
        Huggingface Dataset with combined conversations.
    """
    new_data = []
    i = 0
    while i < len(dataset):
        if i + 1 < len(dataset):
            convo = [
                {"role": dataset[i]["role"], "content": dataset[i]["content"]},
                {"role": dataset[i + 1]["role"], "content": dataset[i + 1]["content"]},
            ]
            new_data.append({"conversations": convo})
            i += 2
        else:
            convo = [{"role": dataset[i]["role"], "content": dataset[i]["content"]}]
            new_data.append({"conversations": convo})
            i += 1
    return Dataset.from_list(new_data)


def replace_system_message(example: dict) -> dict:
    """
    Replace system message in text if format allows.

    Args:
        example: Single dataset example with "text" field.
        custom_system_message: The system message string to replace with.

    Returns:
        Modified example dict.
    """
    parts = example["text"].split("<|eot_id|>", 1)
    if len(parts) == 2:
        new_text = custom_system_message + parts[1]
    else:
        new_text = example["text"]
    return {"text": new_text}


def formatting_prompts_func(examples: dict) -> dict:
    """
    Apply chat template formatting to conversations.

    Args:
        examples: Batch of examples from the dataset.
        tokenizer: Tokenizer instance with apply_chat_template method.

    Returns:
        Dictionary with formatted "text".
    """
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix(tokenizer.bos_token)
        for convo in convos
    ]
    return {"text": texts}


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



def main():
    args = parse_args()
    global tokenizer, custom_system_message

    # Initialize wandb if project specified
    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.3-70B-Instruct",
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
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



    agent_name = args.agent_name.replace(' ', '').lower()
    output_dir = os.path.join(args.output_dir , args.wandb_run_name)
    message = f"""You are {args.agent_name}. Fill in the next response to the conversation by continuing the dialogue naturally. Only write what {agent_name} would say next — do not write for other speakers.

Example:
user:  
{agent_name}: Good evening, everyone. Let's get started with the agenda.  
Speaker_1: Thanks, {agent_name}. I just had a quick question about the minutes from last time.  
assistant:  
{agent_name}: Sure, go ahead with your question."""

    custom_system_message = f"<|start_header_id|>system<|end_header_id|>\n\n{message}\n<|eot_id|>"

    # Dataset preparation
    train_data, test_data = train_test_split(agent_name, data_path=args.data_path)

    train_data = convert_to_chat_format(train_data)
    test_data = convert_to_chat_format(test_data)

    train_data = combine_conversations(train_data)
    test_data = combine_conversations(test_data)


    train_data = train_data.map(formatting_prompts_func, batched=True)
    test_data = test_data.map(formatting_prompts_func, batched=True)

    train_data = train_data.map(replace_system_message)
    test_data = test_data.map(replace_system_message)

    print("--------------------")
    print("Train Format")
    print(test_data['text'][0])
    print("--------------------")
    
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
            per_device_train_batch_size = 2,
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

    trainer_stats = trainer.train()

    print("Training finished!")
    model.save_pretrained(output_dir)  # Local saving
    print(f"Model saved to {output_dir}")

    # Finish wandb run
    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
