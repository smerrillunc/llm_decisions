import os
import argparse
import gc
import torch
from peft import AutoPeftModelForCausalLM


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
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory to merge"
    )

    args = parser.parse_args()

    output_dir = args.outputdir
    print(f"Attempting to merge model: {output_dir}")
    merge(output_dir)