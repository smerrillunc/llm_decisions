import os
import json
import argparse
from typing import Dict, List
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from utils import train_test_split, add_system_message

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
              temperature=1.0, top_p=0.95, top_k=50, repetition_penalty=1.0, seed=None) -> str:
    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty
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
    parser.add_argument('--num_completions', '-nc', type=int, default=3, help='Number of completions per prompt.')

    # 🔧 Added generation parameters
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (creativity).')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p nucleus sampling.')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling.')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty.')
    parser.add_argument("--cot", action="store_true", help="Use CoT Reasoning.")

    args = parser.parse_args()
    results = []

    print(f"🔄 Loading data for speaker: {args.speaker}")
    train_data, test_data, train_completion_data = train_test_split(args.speaker, data_path='/playpen-ssd/smerrill/dataset')
    system_message = f"You are a school board member named '{args.speaker}' participating in a collaborative board discussion. Please read through the conversation and think step by step about about how '{args.speaker}' would think. Then, write what '{args.speaker}' would say next in the conversation."

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    num_prompts = args.max_prompts if args.max_prompts is not None else len(test_data)
    num_prompts = min(num_prompts, len(test_data))
    
    for i in tqdm(range(num_prompts)):
        prompt = test_data[i]['prompt']
        completion = test_data[i]['completion']

        if args.cot:
            prompt = add_system_message(prompt, system_message)
        responses = []
        for j in range(args.num_completions):
            response = ask_model(
                prompt,
                model,
                tokenizer,
                speaker=args.speaker,
                max_new_tokens=150,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                seed=torch.seed()
            )
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

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:  # This avoids errors if output_file is just a filename (no dir)
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(results)} items with {args.num_completions} responses each to {args.output_file}")
