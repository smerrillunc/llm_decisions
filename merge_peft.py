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
import evaluate
from peft import AutoPeftModelForCausalLM
import gc


def merge(output_dir):
    adapter_config_path = os.path.join(output_dir, "adapter_config.json")
    merged_output_dir = os.path.join(output_dir, "merged")

    if not os.path.exists(adapter_config_path):
        print(f"No adapter config found at {adapter_config_path}.")
        print("This model is likely already merged. Skipping merge.")
        return

    print("Loading PEFT model to CPU")
    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    print("Merging LoRA adapter into base model")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {merged_output_dir}")
    merged_model.save_pretrained(merged_output_dir, safe_serialization=True, max_shard_size="2GB")

    print("Merge complete. Cleaning up.")
    del merged_model
    gc.collect()
    

if __name__ == "__main__":
    output_dirs = [#'/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/ellenosborne_16',
                   '/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/grahampaige_16',
                   '/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/judyle_16',
                   '/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/kateacuff_16',
                   '/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/katrinacallsen_16',
                   '/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/davidoberg_16',
                   ]
    
    for output_dir in output_dirs:
        print(f"attempting to merge model: {output_dir}")
        merge(output_dir)