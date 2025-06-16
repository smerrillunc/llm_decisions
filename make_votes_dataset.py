import os
import re
import argparse
import pandas as pd
import torch
import pickle
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from accelerate import init_empty_weights, Accelerator


AGENT_NAME_MAP = {
    "acuff": "kateacuff",
    "dr. acuff": "kateacuff",
    "mrs. acuff": "kateacuff",
    "osborne": "ellenosborne",
    "ms. osborne": "ellenosborne",
    "paige": "grahampaige",
    "mr. paige": "grahampaige",
    "callsen": "katrinacallsen",
    "ms. callsen": "katrinacallsen",
    "oberg": "davidoberg",
    "mr. oberg": "davidoberg",
    "alcaro": "jonnoalcaro",
    "mr. alcaro": "jonnoalcaro",
}

TARGET_AGENTS = set(AGENT_NAME_MAP.values())


def standardize_name(name):
    name = name.strip().lower().rstrip(".")
    name = re.sub(r"^(mr|ms|mrs|dr|chair)\.?\s+", "", name)
    return AGENT_NAME_MAP.get(name)


def extract_names(name_field):
    if pd.isna(name_field):
        return []
    text = str(name_field).lower()
    text = re.sub(r'\band\b', ',', text)
    text = re.sub(r'[.;]', '', text)
    return [n.strip() for n in text.split(',') if n.strip()]


def generate_question(context, text_gen, max_tokens=40, temperature=0.7):
    prompt = f"Turn the following statement into a formal yes/no question for a school board vote:\n\n\"{context.strip()}\"\n\nQuestion:"
    try:
        result = text_gen(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
        return result[0]['generated_text'].split("Question:")[-1].strip()
    except Exception as e:
        print(f"[ERROR] Failed to generate question for context: {context[:50]}...: {e}")
        return context


def load_model_pipeline(model_name):
    print(f"[INFO] Loading model '{model_name}' using accelerate...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    with init_empty_weights():
        _ = AutoModelForCausalLM.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype="auto",
        max_memory={i: "35GiB" for i in range(torch.cuda.device_count())}
    )

    return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")


def build_dataset(csv_path, text_gen):
    print(f"[INFO] Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    agent_dataset = defaultdict(list)

    print(f"[INFO] Processing {len(df)} rows...\n")

    for idx, row in df.iterrows():
        context = str(row.get("Context", "")).strip()
        if not context:
            continue

        print(f"\n--- Row {idx+1} ---")
        print(f"[CONTEXT] {context}")

        question = generate_question(context, text_gen)
        print(f"[QUESTION] {question}")

        for vote_col, vote_label in [("Ayes", "Aye"), ("Nayes", "Naye")]:
            names = extract_names(row.get(vote_col, ""))
            for raw_name in names:
                agent_key = standardize_name(raw_name)
                if agent_key in TARGET_AGENTS:
                    agent_dataset[agent_key].append((question, vote_label))
                    print(f"[ASSIGN] {agent_key}: {vote_label}")

    return dict(agent_dataset)


def save_dataset(dataset, output_path, json_out=False):
    print(f"[INFO] Saving dataset to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    if json_out:
        import json
        json_path = os.path.splitext(output_path)[0] + ".json"
        print(f"[INFO] Also saving JSON to: {json_path}")
        with open(json_path, "w") as jf:
            json.dump(dataset, jf, indent=2)


def main():
    #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6  accelerate launch --num_processes 1 process_votes.py
    parser = argparse.ArgumentParser(description="Build agent vote dataset from board meeting votes.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help="HuggingFace model name or path")
    parser.add_argument("--csv_path", type=str, default="/playpen-ssd/smerrill/dataset/votes.csv", help="Path to votes.csv file")
    parser.add_argument("--output_path", type=str, default="agent_vote_dataset.pkl", help="Output .pkl file path")
    parser.add_argument("--json_out", action="store_true", help="Also save the dataset as JSON")

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    text_gen = load_model_pipeline(args.model)
    dataset = build_dataset(args.csv_path, text_gen)
    save_dataset(dataset, args.output_path, args.json_out)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
