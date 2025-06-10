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

from peft import LoraConfig
import numpy as np
import pandas as pd

import wandb
from utils import train_test_split, compute_perplexity_on_dataset_accelerate, compute_metrics, train_on_responses_only
import evaluate
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object


@dataclass
class ScriptArguments:
   
    merged_path: str = field(
        default=None, metadata={"help": "Model Name"}
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
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 accelerate launch --num_processes 1 LLM_Test.py --merged_path /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/ellenosborne_16/merged  --wandb_run_name /playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/ellenosborne_16/merged
    
    torch.cuda.empty_cache()
    gc.collect()
    
    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb.init(project=ScriptArguments.wandb_project, name=ScriptArguments.wandb_run_name)

    max_memory = {
    0: "40GiB",
    1: "40GiB",
    2: "40GiB",
    3: "40GiB",
    4: "40GiB",
    5: "40GiB",
    6: "40GiB",
    7: "40GiB",  
    "cpu": "100GiB"
}
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
            llm_int8_enable_fp32_cpu_offload=True

        )
    
    tokenizer = AutoTokenizer.from_pretrained(ScriptArguments.merged_path.split('/merged')[0])

    # Try to load model with device_map="auto" and lower max_memory per GPU
    try:
        max_memory = {i: "25GiB" for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "100GiB"
        model = AutoModelForCausalLM.from_pretrained(
            ScriptArguments.merged_path,
            quantization_config=quantization_config,
            torch_dtype=quant_storage_dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            device_map="auto",
        )
    except RuntimeError as e:
        print("[WARNING] OOM on GPU, loading model on CPU only.")
        model = AutoModelForCausalLM.from_pretrained(
            ScriptArguments.merged_path,
            quantization_config=quantization_config,
            torch_dtype=quant_storage_dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
        )


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

    # no need to wrap model with PeftModel since we are not training
    #model = accelerator.prepare(model)

    print("Running Perplexity Evaluation on Each Dataset (Accelerate)")
    results = []
    for dataset in datasets:
        torch.cuda.empty_cache()
        gc.collect()
        print(f'Computing Perplexity for Dataset: {dataset}')
        _, test_data, train_completion_data = train_test_split(dataset)
        ppl = compute_perplexity_on_dataset_accelerate(
            model, tokenizer, train_completion_data, accelerator, max_length=1024, batch_size=1
        )
        if accelerator.is_main_process:
            print(f"Perplexity for {dataset}: {ppl:.2f}")
            results.append({'model': ScriptArguments.merged_path, "dataset": dataset, "perplexity": ppl})

    # Save results as DataFrame (only on main process)
    if accelerator.is_main_process:
        df = pd.DataFrame(results)
        save_path = os.path.join(ScriptArguments.merged_path, "perplexity_results.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved perplexity results to {save_path}")

    wandb.finish()