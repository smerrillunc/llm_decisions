import os
import re
from accelerate import Accelerator

import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from utils import get_dataset, train_on_responses_only

filters = {
    'meta-llama/Meta-Llama-3-70B-Instruct': ["<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"],
    'Qwen/Qwen3-235B-A22B-Instruct-2507': ['<|im_start|>\n', '<|im_end|>\n'],
    'Qwen/Qwen3-30B-A3B-Instruct-2507': ['<|im_start|>\n', '<|im_end|>\n'],
    'openai/gpt-oss-120b': ['<|start|>user<|message|>', '<|start|>assistant<|message|>'],
    'openai/gpt-oss-20b': ['<|start|>user<|message|>', '<|start|>assistant<|message|>']
}

def clean_text(text: str) -> str:
    # remove <think> blocks and normalize newlines
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    text = re.sub(r'\n{2,}', '\n', text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def get_args():
    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--agent_name", type=str, default="judyle")
    parser.add_argument("--sys_message", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, default="/playpen/smerrill/dataset")
    parser.add_argument("--save_dir", type=str, default="/playpen/smerrill/trained_models")
    parser.add_argument("--wandb_project", type=str, default="LLM_Decisions")
    parser.add_argument("--config", type=str, default="LLM_Decisions")
    return parser.parse_args()


def training_function(script_args):
    # -------------------
    # Set CUDA device based on local rank
    # -------------------

    # -------------------
    # Tokenizer
    # -------------------

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=True)
    tokenizer.model_max_length = script_args.max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    # -------------------
    # Dataset
    # -------------------
    train_path = os.path.join(script_args.dataset_path, "train_dataset.json")
    test_path = os.path.join(script_args.dataset_path, "test_dataset.json")
    train_examples, _ = get_dataset(train_path, test_path, script_args.agent_name, script_args.sys_message)
    train_examples = [tokenizer.apply_chat_template(x, tokenize=False) for x in train_examples]
    train_data = Dataset.from_list([{"text": clean_text(x)} for x in train_examples])

    # -------------------
    # Model
        # -------------------
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        device_map='cpu',
        use_cache=False,
    )

    peft_config = LoraConfig(
        lora_alpha=args.factors*2,
        lora_dropout=script_args.dropout,
        r=script_args.factors,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    model = get_peft_model(model, peft_config)
    # -------------------
    # TrainingArguments
    # -------------------
    training_args = TrainingArguments(
        output_dir=script_args.save_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=300,
        learning_rate=script_args.lr,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.1,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
        fp16=True,
        group_by_length=True,
        fsdp_min_num_params=0,
    )

    # -------------------
    # Trainer
    # -------------------
    # accelerate is now implemented inside of trainer so no need to do this anymore
    #model = accelerator.prepare(model)
    #train_data = accelerator.prepare(train_data)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part=filters[script_args.model_name][0],
        response_part=filters[script_args.model_name][1]
    )

    # -------------------
    # Train
    # -------------------
    trainer.train()

    # Save model only from rank 0
    if torch.distributed.get_rank() == 0:
        trainer.save_model()
        log_history = trainer.state.log_history
        df = pd.DataFrame(log_history)
        results_path = os.path.join(training_args.output_dir, "train_eval_log.csv")
        df.to_csv(results_path, index=False)
        print(f"\nSaved train/eval logs to {results_path}")


if __name__ == "__main__":
    args = get_args()
    set_seed(42)
    training_function(args)
