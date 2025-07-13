import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse
import pickle
import json
import os
from typing import Dict, List


# --- Prompt Template ---
REVERSE_PROMPT_TEMPLATE = """You are given a monologue your task is to write the *Original Question* that could have elicited a response like the one below. Make the prompt clear, natural, and open-ended enough to produce a detailed answer.

Monologue:
\"\"\"{monologue}\"\"\"

Original Question:"""


# --- CLI Args ---
def parse_args():
    parser = argparse.ArgumentParser(description="Generate prompts that could elicit monologues.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help="Huggingface model path")
    parser.add_argument("--input", type=str, default="/playpen-ssd/smerrill/dataset/monologues.pkl", help="Path to input .pkl")
    parser.add_argument("--output_dir", type=str, default="./monologue_results", help="Output directory")
    parser.add_argument("--max_per_agent", type=int, default=20, help="Max monologues to process per agent")
    return parser.parse_args()


# --- Model Setup ---
def setup_model(model_name):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model with multi-GPU support...")
    max_memory = {i: "30GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "200GiB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder="offload",
        offload_state_dict=True,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )

    print("Creating pipeline...")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=150,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
    return pipe


def generate_reverse_prompts(pipe, monologues: Dict[str, List[Dict[str, str]]], max_per_agent: int) -> Dict[str, List[Dict[str, str]]]:
    results = {}

    for agent, entries in monologues.items():
        print(f"\n🧠 --- Processing agent: {agent} ---")
        agent_results = []

        for idx, entry in enumerate(entries[:max_per_agent]):
            monologue = entry["completion"] if isinstance(entry, dict) else entry
            print(f"\n📄 Monologue {idx + 1}/{min(len(entries), max_per_agent)} (length {len(monologue)} chars):")
            print("-" * 80)
            print(monologue[:500] + ("..." if len(monologue) > 500 else ""))
            print("-" * 80)

            prompt_input = REVERSE_PROMPT_TEMPLATE.format(monologue=monologue)
            output = pipe(prompt_input, max_new_tokens=150)[0]["generated_text"]
            reverse_prompt = output[len(prompt_input):].strip()

            print(f"\n📝 Generated Prompt:\n>>> {reverse_prompt}")
            print("=" * 80)

            agent_results.append({
                "prompt": reverse_prompt,
                "completion": monologue
            })

        results[agent] = agent_results
        print(f"\n✅ Done with agent '{agent}': {len(agent_results)} prompts generated.")

    return results



# --- Main ---
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"📂 Loading monologues from: {args.input}")
    with open(args.input, "rb") as f:
        monologues = pickle.load(f)

    pipe = setup_model(args.model)

    print("\n🚀 Starting reverse prompt generation...")
    reversed_dataset = generate_reverse_prompts(pipe, monologues, max_per_agent=args.max_per_agent)

    output_path = os.path.join(args.output_dir, "reverse_prompt_monologues.json")
    with open(output_path, "w") as f:
        json.dump(reversed_dataset, f, indent=2)

    print(f"\n✅ All done. Results saved to {output_path}")
