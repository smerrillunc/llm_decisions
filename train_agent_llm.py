import os
from dataclasses import dataclass, field
from datasets import (Dataset, IterableDataset,)
import torch
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import  TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    set_seed,

)

from trl import setup_chat_format
from peft import LoraConfig
import numpy as np
import pandas as pd
from trl import (
   SFTTrainer)

import wandb
from utils import train_test_split, train_on_responses_only, preprocess_test_data, compute_perplexity_metrics

import evaluate
from accelerate import Accelerator

# Comment in if you want to use the Llama 3 instruct template but make sure to add modules_to_save
# LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

# Anthropic/Vicuna like template without the need for special tokens
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

@dataclass
class ScriptArguments:
   
    model_name: str = field(
        default="meta-llama/Meta-Llama-3-70B-Instruct", metadata={"help": "Model Name"}
    )

    max_seq_length: int = field(
        default=850, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )

    factors: int = field(
        default=16, metadata={"help": "Lora factors"}
    )

    agent_name: str = field(
        default='judyle', metadata={"help": "Name of agent to train"}
    )
    
    dataset_path: str = field(
        default='/playpen-ssd/smerrill/dataset', metadata={"help": "Dataset path"}
    )
    save_dir: str = field(
        default='/playpen-ssd/smerrill/trained_models', metadata={"help": "Trained model save path"}
    )
    
    wandb_project: str = field(
        default='LLM_Decisions', metadata={"help": "Wandb project name"}
    )

    wandb_run_name: str = field(
        default='test', metadata={"help": "Wandb run name"}
    )


def training_function(script_args, training_args, accelerator):
    output_name = f"{script_args.agent_name}_{script_args.factors}"
    output_dir = os.path.join(script_args.save_dir, script_args.model_name, output_name)
    training_args.output_dir = output_dir

    ################
    # Dataset
    ################
    train_data, eval_data, train_completion_data = train_test_split(
        script_args.agent_name, data_path=script_args.dataset_path
    )
    
    train_data = Dataset.from_list([{"text": text} for text in train_data])
    eval_data = preprocess_test_data(eval_data)
    
    ################
    # Model & Tokenizer
    ################

    # Tokenizer        
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    
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

    max_memory = {i: "35GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "200GiB"

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        #quantization_config=quantization_config,
        attn_implementation="sdpa", # use sdpa, alternatively use "flash_attention_2"
        #torch_dtype=quant_storage_dtype,
        max_memory=max_memory,
        device_map=None,  # Automatically set device map for multi-GPU
        use_cache=False if training_args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # --- Accelerate: prepare model and tokenizer ---
    #model, train_data, eval_data = accelerator.prepare(model, train_data, eval_data)

    ################
    # PEFT
    ################

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
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
        eval_dataset=eval_data,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        compute_metrics=compute_perplexity_metrics
    )
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
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
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    accelerator = Accelerator()
    if accelerator.is_main_process:
        wandb.init(project=script_args.wandb_project, name=script_args.wandb_run_name)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    set_seed(training_args.seed)

    model, tokenizer = training_function(script_args, training_args, accelerator)

    if accelerator.is_main_process:
        print(f"\nSaved all results to {os.path.join(training_args.output_dir, 'evaluation_results.csv')}")
        wandb.finish()
