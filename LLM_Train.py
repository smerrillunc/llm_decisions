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
        set_seed,

)
from trl import setup_chat_format
from peft import LoraConfig
import numpy as np
import pandas as pd
from trl import (
   SFTTrainer)

import wandb
from utils import train_test_split, compute_perplexity, compute_metrics, train_on_responses_only
import evaluate

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



# ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=4 ./scripts/run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml

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


def training_function(script_args, training_args):
    output_name = f"{script_args.agent_name}_{script_args.factors}"
    output_dir = os.path.join(script_args.save_dir, script_args.model_name, output_name)
    training_args.output_dir = output_dir  # if it's a class like TrainingArguments

    ################
    # Dataset
    ################
    train_data, test_data, train_completion_data = train_test_split(script_args.agent_name, data_path=script_args.dataset_path)
    train_data = Dataset.from_list([{"text": text} for text in train_data])

    ################
    # Model & Tokenizer
    ################

    # Tokenizer        
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    
    # Model    
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        attn_implementation="sdpa", # use sdpa, alternatively use "flash_attention_2"
        torch_dtype=quant_storage_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

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
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(training_args.output_dir,safe_serialization=True, max_shard_size="2GB")
    return model, tokenizer
    
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()    
    
    wandb.init(project=script_args.wandb_project, name=script_args.wandb_run_name)

    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)
  
    # launch training
    print("Training Complete")
    model, tokenizer = training_function(script_args, training_args)
    
    print("Running Evaluation")
    datasets = [
    'kateacuff',
    'ellenosborne',
    'grahampaige',
    'judyle',
    'katrinacallsen',
    'davidoberg',
    'jonnoalcaro'
    ]

    print("Loading Metrics")
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    results = []

    for dataset in datasets:
        print(f'Model: {script_args.model_path}, Dataset: {dataset}')
        _, test_data, train_completion_data = train_test_split(dataset)

        print("Computing Train Perplexity")
        perplexity_train, generated_texts, reference_texts = compute_perplexity(
            model,
            train_completion_data,
            tokenizer,
            max_length=1024,
            verbose=False
        )

        bleu_score, rouge_score, bertscore_result, avg_bertscore_f1 = compute_metrics(
            generated_texts, reference_texts, bleu, rouge, bertscore
        )

        print("Computing Test Perplexity")
        perplexity_test = compute_perplexity(
            model,
            test_data, 
            tokenizer,
            max_length=1024,
            verbose=False
        )

        print(f"Train PPL: {perplexity_train:.2f}, Test PPL: {perplexity_test:.2f}")
        print(f"BLEU: {bleu_score}, ROUGE: {rouge_score}, BERTScore-F1: {avg_bertscore_f1:.4f}")

        # Append result row
        results.append({
            "model": script_args.model_path,
            "dataset": str(dataset),
            "train_perplexity": perplexity_train,
            "test_perplexity": perplexity_test,
            "bleu_score": bleu_score["bleu"],
            "rouge_score": rouge_score["rougeL"],
            "bertscore_f1": avg_bertscore_f1
        })

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    
    df.to_csv(os.path.join(training_args.output_dir, "evaluation_results.csv"), index=False)
    print(f"\nSaved all results to {os.path.join(training_args.output_dir, 'evaluation_results.csv')}")

    
    wandb.finish()
