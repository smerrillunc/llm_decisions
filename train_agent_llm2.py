import os
from dataclasses import dataclass, field
from datasets import (Dataset, IterableDataset,)
import torch
from transformers import AutoTokenizer, TrainingArguments
from transformers import TrainingArguments

from trl.commands.cli_utils import  TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    set_seed,

)

from peft import LoraConfig
import numpy as np
import pandas as pd
from trl import (
   SFTTrainer)

import wandb
from utils import get_dataset, preprocess_test_data, compute_perplexity_metrics, train_on_responses_only

import evaluate
from accelerate import Accelerator
import re
import argparse

# Comment in if you want to use the Llama 3 instruct template but make sure to add modules_to_save
# LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

# Anthropic/Vicuna like template without the need for special tokens
#LLAMA_3_CHAT_TEMPLATE = (
#    "{% for message in messages %}"
#        "{% if message['role'] == 'system' %}"
#            "{{ message['content'] }}"
#        "{% elif message['role'] == 'user' %}"
#            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
#        "{% elif message['role'] == 'assistant' %}"
#            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
#        "{% endif %}"
#    "{% endfor %}"
#    "{% if add_generation_prompt %}"
#    "{{ '\n\nAssistant: ' }}"
#    "{% endif %}"
#)
filters = {'meta-llama/Meta-Llama-3-70B-Instruct':["<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"],
        'Qwen/Qwen3-235B-A22B-Instruct-2507':['<|im_start|>\n', '<|im_end|>\n'],
        'openai/gpt-oss-120b ':['<|start|>user<|message|>', '<|start|>assistant<|message|>']
        }

# REMOVES THINKING BLOCK FOR QWWEN
def clean_text(text: str) -> str:
    # remove the whole <think>...</think> block
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    # collapse multiple newlines into one
    text = re.sub(r'\n{2,}', '\n', text)
    # strip spaces at line ends
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def get_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help="Model Name"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2048, help="The maximum sequence length for SFT Trainer"
    )
    parser.add_argument(
        "--factors", type=int, default=32, help="Lora factors"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.05, help="Lora dropout rate"
    )
    parser.add_argument(
        "--agent_name", type=str, default="judyle", help="Name of agent to train"
    )
    parser.add_argument(
        "--sys_message", type=int, default=1, help="Include System message/context card during training"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="/playpen-ssd/smerrill/dataset", help="Dataset path"
    )
    parser.add_argument(
        "--save_dir", type=str, default="/playpen-ssd/smerrill/trained_models", help="Trained model save path"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="LLM_Decisions", help="Wandb project name"
    )

    parser.add_argument(
        "--config", type=str, default="LLM_Decisions", help="Wandb project name"
    )

    return parser.parse_args()


def training_function(script_args, accelerator):
    #training_args.output_dir = script_args.save_dir
    #training_args.num_train_epochs = script_args.epochs
    #training_args.learning_rate = script_args.lr

    training_args = TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        logging_steps=0.2,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="none"
    )

    ################
    # Dataset
    ################
    #train_data, eval_data, train_completion_data = train_test_split(
    #    script_args.agent_name, data_path=script_args.dataset_path
    #)
    
    #train_data = Dataset.from_list([{"text": text} for text in train_data])
    #eval_data = preprocess_test_data(eval_data)
    

    ################
    # Model & Tokenizer
    ################

    # Tokenizer        
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
        
    train_path = '/playpen/smerrill/dataset/train_dataset.json'
    test_path = '/playpen/smerrill/dataset/test_dataset.json'

    train_examples, _ = get_dataset(train_path, test_path, script_args.agent_name, script_args.sys_message)
    train_examples = [tokenizer.apply_chat_template(x, tokenize=False) for x in train_examples]
    train_data = Dataset.from_list([{"text": clean_text(text)} for text in train_examples])

    # Model    
    #torch_dtype = torch.bfloat16
    #quant_storage_dtype = torch.bfloat16

    #quantization_config = BitsAndBytesConfig(
    #        load_in_4bit=True,
    #        bnb_4bit_use_double_quant=True,
    #        bnb_4bit_quant_type="nf4",
    #        bnb_4bit_compute_dtype=torch_dtype,
    #        bnb_4bit_quant_storage=quant_storage_dtype,
    #    )
    
    # Whichever config we want
    #quantization_config = BitsAndBytesConfig(
    #    load_in_8bit=True,
    #    llm_int8_threshold=6.0,  # Optional, can tweak based on model
    #    llm_int8_has_fp16_weight=False,  # Optional, depends on your model and setup
    #)

    #max_memory = {i: "35GiB" for i in range(torch.cuda.device_count())}
    #max_memory["cpu"] = "200GiB"

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        #quantization_config=quantization_config,
        attn_implementation="sdpa", # use sdpa, alternatively use "flash_attention_2"
        torch_dtype=torch.bfloat16,  
        #max_memory=max_memory,
        device_map=None,  # Automatically set device map for multi-GPU
        use_cache=False,  # this is needed for gradient checkpointing
    )
    

    # --- Accelerate: prepare model and tokenizer ---
    #model, train_data, eval_data = accelerator.prepare(model, train_data, eval_data)

    ################
    # PEFT
    ################

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    if script_args.model_name == 'openai/gpt-oss-120b':
        peft_config = LoraConfig(
        lora_alpha=2*script_args.factors,
        lora_dropout=script_args.dropout,
        r=script_args.factors,
        bias="none",
        target_modules="all-linear",
        target_parameters=[
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
        task_type="CAUSAL_LM",
        # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    )

    else:
       peft_config = LoraConfig(
        lora_alpha=2*script_args.factors,
        lora_dropout=script_args.dropout,
        r=script_args.factors,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        #eval_dataset=eval_data,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        #compute_metrics=compute_perplexity_metrics
    )
    
    # Now done in SFTConfig Directly
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part=filters[script_args.model_name][0],
        response_part=filters[script_args.model_name][1]
    )

    if accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    accelerator.wait_for_everyone()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

    if accelerator.is_main_process:
        log_history = trainer.state.log_history
        df = pd.DataFrame(log_history)
        results_path = os.path.join(training_args.output_dir, "train_eval_log.csv")
        df.to_csv(results_path, index=False)
        print(f"\nSaved train/eval logs to {results_path}")

    return model, tokenizer


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 accelerate launch --num_processes 4 train_agent_llm.py --agent_name judyle --config "/playpen-ssd/smerrill/llm_decisions/configs/llamma_3_70b.yaml"
    #parser = TrlParser((ScriptArguments, TrainingArguments))
    #script_args, training_args = parser.parse_args_and_config()
    script_args, training_args = None, None
    script_args = get_args()

    accelerator = Accelerator()
    if accelerator.is_main_process:
        wandb_run_name = f"{script_args.agent_name}_{script_args.factors}"
        wandb.init(project=script_args.wandb_project, name=wandb_run_name)

    set_seed(42)


    model, tokenizer = training_function(script_args, accelerator)

    if accelerator.is_main_process:
        print(f"\nSaved all results to {os.path.join(training_args.output_dir, 'evaluation_results.csv')}")
        wandb.finish()
