import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.8"

from dataclasses import dataclass, field
from datasets import (Dataset, IterableDataset,)
import torch
import gc
from transformers import AutoTokenizer, TrainingArguments
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    set_seed,
    DataCollatorWithPadding
)
from torch.utils.data import DataLoader

from peft import AutoPeftModelForCausalLM
from transformers import HfArgumentParser

from peft import LoraConfig
import numpy as np
import pandas as pd

import wandb
from utils import train_test_split, compute_metrics, train_on_responses_only
import evaluate
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object

def compute_perplexity_on_dataset_accelerate(model, tokenizer, dataset, accelerator, max_length=1024, batch_size=1):
    import math
    import torch
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import Dataset as TorchDataset

    class PromptCompletionDataset(TorchDataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            return item['prompt'] + item['completion']

    eval_dataset = PromptCompletionDataset(dataset)
    dataloader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda batch: tokenizer(
            batch, 
            padding="longest",  # Better padding control
            truncation=True, 
            max_length=max_length,
            return_tensors="pt"
        )
    )
    dataloader = accelerator.prepare(dataloader)  # Critical for distributed setup
    
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]#.to(accelerator.device)
            attention_mask = batch["attention_mask"]#.to(accelerator.device)
            labels = input_ids.clone()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            # Handle padding and gathering properly
            loss = accelerator.pad_across_processes(loss, dim=0)
            gathered_loss = accelerator.gather(loss)
            losses.append(gathered_loss.cpu())
    
    all_losses = torch.cat(losses)
    mean_loss = all_losses.mean().item()
    perplexity = math.exp(mean_loss)
    return perplexity

@dataclass
class ScriptArguments:
   
    merged_path: str = field(
        default='None', metadata={"help": "Merged model path"}
    )

    dataset_path: str = field(
        default='/playpen-ssd/smerrill/dataset', metadata={"help": "Dataset path"}
    )
    
    wandb_project: str = field(
        default='LLM_Decisions', metadata={"help": "Wandb project name"}
    )

    wandb_run_name: str = field(
        default='test', metadata={"help": "Wandb run name"}
    )

    
if __name__ == "__main__":
    # To run this script with accelerate, use:
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 evaluate_agent_perplexity.py --merged_path /path/to/merged --wandb_run_name myrun
    parser = HfArgumentParser(ScriptArguments)
    script_args, = parser.parse_args_into_dataclasses()

    torch.cuda.empty_cache()
    gc.collect()
    
    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb.init(project=script_args.wandb_project, name=script_args.wandb_run_name)

    torch_dtype = torch.bfloat16
    #quant_storage_dtype = torch.bfloat16
    #quantization_config = BitsAndBytesConfig(
    #        load_in_4bit=True,
    #        bnb_4bit_use_double_quant=True,
    #        bnb_4bit_quant_type="nf4",
    #        bnb_4bit_compute_dtype=torch_dtype,
    #        bnb_4bit_quant_storage=quant_storage_dtype,
    #        llm_int8_enable_fp32_cpu_offload=True#
    #
    # #   )
    
    path = script_args.merged_path
    print(path)
    path = path.replace('/merged', '')
    print(f"Loading tokenizer from {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)    
    if tokenizer.pad_token is None:
        print("Tokenizer has no pad_token, setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in full precision (bf16) and let Accelerate handle device placement
    try:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.merged_path,
            torch_dtype=torch.bfloat16,  # or torch.float16 if you prefer
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    except RuntimeError as e:
        print("[ERROR] Failed to load model on CUDA. Exiting.")
        raise e

    # Let Accelerate handle device placement
    model = accelerator.prepare(model)


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

    print("Running Perplexity Evaluation on Each Dataset (Accelerate)")
    results = []
    for dataset in datasets:
        torch.cuda.empty_cache()
        gc.collect()
        if accelerator.is_main_process:
            print(f'Computing Perplexity for Dataset: {dataset}')
        _, test_data, train_completion_data = train_test_split(dataset)
        
        ppl = compute_perplexity_on_dataset_accelerate(
            model, tokenizer, test_data, accelerator, max_length=1024, batch_size=1)
        if accelerator.is_main_process:
            print(f"Perplexity for {dataset}: {ppl:.2f}")
            results.append({'model': path, "dataset": dataset, "perplexity": ppl})

    # Save results as DataFrame (only on main process)
    if accelerator.is_main_process:
        df = pd.DataFrame(results)
        save_path = os.path.join(path, "perplexity_results.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved perplexity results to {save_path}")

    wandb.finish()