import os
import json
import argparse
from typing import Dict, List
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import wrap_prompt

def load_model_and_tokenizer(model_path: str):
    tokenizer_path = model_path.replace('/merged', '')
    print(f"Model path: {model_path}")
    print(f"Tokenizer path: {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return model, tokenizer


def ask_model(input_text: str, model, tokenizer, speaker='agent', max_new_tokens=150,
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
    parser = argparse.ArgumentParser(description="Run inference on reverse-generated prompts.")
    parser.add_argument('--model_path', '-m', required=True, help='Path to the fine-tuned model.')
    parser.add_argument('--prompts_file', '-p', required=True, help='Path to JSON file with reverse prompts.')
    parser.add_argument('--speaker', '-s', required=True, help='Agent name (for speaker tag cleanup).')
    parser.add_argument('--output_file', '-o', default='responses_from_prompts.json', help='Output file to save results.')
    parser.add_argument('--max_prompts', '-n', type=int, default=None, help='Max number of prompts to process.')
    parser.add_argument('--num_completions', '-nc', type=int, default=3, help='Number of completions per prompt.')

    # Generation args
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    args = parser.parse_args()

    print(f"🔄 Loading reverse prompts from {args.prompts_file}")
    with open(args.prompts_file, 'r') as f:
        data = json.load(f)

    prompts = data[args.speaker]  # replace some_key with actual key

    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    print(f"Loaded {len(prompts)} prompts for speaker {args.speaker}")

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    results = []
    for i, item in enumerate(tqdm(prompts)):
        prompt = wrap_prompt(item['prompt'], agent_name=args.speaker)
        monologue = item['completion']

        print(f"\n--- [{i + 1}/{len(prompts)}] ---")
        print("Prompt:", prompt)
        print("Original Monologue Snippet:", monologue[:150].replace("\n", " "), "...")

        responses = []
        for j in range(args.num_completions):
            response = ask_model(
                prompt, model, tokenizer,
                speaker=args.speaker,
                max_new_tokens=250,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                seed=torch.seed()
            )
            print(f"Response {j + 1}:", response)
            responses.append(response)

        results.append({
            "agent": args.speaker,
            "monologue": monologue,
            "prompt": prompt,
            "model_responses": responses
        })

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:  # This avoids errors if output_file is just a filename (no dir)
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nFinished. Saved {len(results)} examples to {args.output_file}")
