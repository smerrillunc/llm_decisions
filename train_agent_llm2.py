import os
from dataclasses import dataclass, field
from datasets import (Dataset, IterableDataset,)
import torch
from transformers import AutoTokenizer, TrainingArguments
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model

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
        'Qwen/Qwen3-30B-A3B-Instruct-2507':['<|im_start|>\n', '<|im_end|>\n'],
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
    import os
    import pandas as pd
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from accelerate.utils import DataLoaderConfiguration
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)

    accelerator = Accelerator(dataloader_config=dataloader_config)
    ################
    # Training arguments
    ################
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
        fp16=True,
        bf16=False,
        group_by_length=True,
        report_to="none"
    )

    ################
    # Dataset and tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token


    train_path = '/playpen-ssd/smerrill/dataset/train_dataset.json'
    test_path = '/playpen-ssd/smerrill/dataset/test_dataset.json'

    train_examples, _ = get_dataset(train_path, test_path, script_args.agent_name, script_args.sys_message)
    train_examples = [tokenizer.apply_chat_template(x, tokenize=False) for x in train_examples]
    train_data = Dataset.from_list([{"text": clean_text(text)} for text in train_examples])

    ################
    # Load base model in bfloat16
    ################
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        dtype=torch.float16,
        device_map=None,
        use_cache=False,
    )

    from peft import get_peft_model, LoraConfig

    peft_config = LoraConfig(
        lora_alpha=16,                           # Scaling factor for LoRA
        lora_dropout=0.05,                       # Add slight dropout for regularization
        r=64,                                    # Rank of the LoRA update matrices
        bias="none",                             # No bias reparameterization
        task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Target modules for LoRA
    )

    model = get_peft_model(model, peft_config)
    tokenizer.model_max_length = 2048
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=300,
        learning_rate=1e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.1,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        # FSDP and distributed configs:
        fsdp="full_shard",
        fsdp_min_num_params=0,
        fsdp_config={
            "fsdp_sharding_strategy": "FULL_SHARD",
            "fsdp_offload_params": True,
            "fsdp_backward_prefetch": "BACKWARD_PRE",
            "fsdp_use_orig_params": True,
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "activation_checkpointing": True,
        },
        fsdp_transformer_layer_cls_to_wrap=None,  # Only needed for CLASS_BASED_WRAP
        accelerator_config=None,
        parallelism_config=None,
    )


    trainer = SFTTrainer(
        model = model,
        #tokenizer = tokenizer,
        train_dataset = train_data,
        #dataset_text_field = "text",
        #max_seq_length = max_seq_length,
        #dataset_num_proc = 2,
        #packing = False, # Can make training 5x faster for short sequences.
        args = training_args,
    )


    # If you have your filtering logic for responses:
    trainer = train_on_responses_only(
        trainer,
        instruction_part=filters[script_args.model_name][0],
        response_part=filters[script_args.model_name][1]
    )

    ################
    # Training
    ################

    trainer.train()
    accelerator.wait_for_everyone()


    trainer.save_model()

    ################
    # Logging
    ################
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
    #local_rank = int(os.environ["LOCAL_RANK"])
    #torch.cuda.set_device(local_rank)

    script_args = get_args()

    #accelerator = Accelerator()
    #if accelerator.is_main_process:
    #    wandb_run_name = f"{script_args.agent_name}_{script_args.factors}"
    #    wandb.init(project=script_args.wandb_project, name=wandb_run_name)

    set_seed(42)


    model, tokenizer = training_function(script_args, None)

    #if accelerator.is_main_process:
    #    print(f"\nSaved all results to {os.path.join(training_args.output_dir, 'evaluation_results.csv')}")
    #    wandb.finish()
