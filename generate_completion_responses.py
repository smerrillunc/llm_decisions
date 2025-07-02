import os
import json
import argparse
from typing import Dict, List
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map


def load_model_and_tokenizer(model_path: str):
    tokenizer_path = model_path.replace('/merged', '')
    print(model_path)
    print(tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"🔄 Loading model from {model_path} using `accelerate`...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return model, tokenizer


def ask_model(input_text: str, model, tokenizer, speaker='kateacuff', max_new_tokens=150,
              temperature=1.0, top_p=0.95, seed=None) -> str:
    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if speaker + ":" in decoded:
        response = decoded.split(f"{speaker}:")[-1].strip()
    else:
        response = decoded.strip()

    response_lines = response.split("\n")
    unique_lines = []
    seen = set()
    for line in response_lines:
        line_clean = line.strip()
        if line_clean not in seen:
            seen.add(line_clean)
            unique_lines.append(line_clean)
    response = " ".join(unique_lines).strip()

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned LLM on question dataset.")
    parser.add_argument('--model_path', '-m', required=True, help='Path to the fine-tuned model directory or HF hub ID.')
    parser.add_argument('--speaker', '-s', required=True, help='Speaker.')
    parser.add_argument('--output_file', '-o', default='test_responses.json', help='File to save generated responses.')
    parser.add_argument('--max_prompts', '-n', type=int, default=None, help='Maximum number of prompts to process.')
    args = parser.parse_args()

    from utils import train_test_split
    results = []

    print(f"🔄 Loading data for speaker: {args.speaker}")
    
    train_data, test_data, train_completion_data = train_test_split(args.speaker, data_path='/playpen-ssd/smerrill/dataset')
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    num_prompts = args.max_prompts if args.max_prompts is not None else len(test_data)
    for i in tqdm(range(num_prompts)):
        prompt = test_data[i]['prompt']
        completion = test_data[i]['completion']

        responses = []
        for j in range(3):
            response = ask_model(prompt, model, tokenizer, speaker=args.speaker,
                                 max_new_tokens=150, temperature=1.0, top_p=0.95, seed=torch.seed())
            responses.append(response)

        print("Prompt:", prompt)
        for j, r in enumerate(responses):
            print(f"Response {j+1}:", r)
        print()

        results.append({
            "prompt": prompt,
            "true_completion": completion,
            "model_responses": responses
        })

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(results)} items with 3 responses each to {args.output_file}")
