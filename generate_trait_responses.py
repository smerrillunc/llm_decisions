import os
import json
import argparse
from typing import Dict, List
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map


def wrap_prompt(prompt, agent_name):
    system_message = (
        "<|begin_of_text|><|system|>\n\n"
        "You are a school board member and will be asked a question. "
        "Please think step by step and explain your reasoning in our response.<|eot_id|>\n\n"
    )

    
    user_message = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"unknownspeaker: {prompt}<|eot_id|>\n\n"
    )

    assistant_header = f"<|start_header_id|>assistant<|end_header_id|>\n\n{agent_name}:"

    return system_message + user_message + assistant_header


def load_model_and_tokenizer(model_path: str):
    tokenizer_path = model_path.replace('/merged', '')
    print(model_path)
    print(tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"Loading model from {model_path} using `accelerate`...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return model, tokenizer


def ask_model(
    question: str,
    model,
    tokenizer,
    speaker='ellenosborne',
    max_new_tokens=500,
    temperature=1.0,
    top_p=1.0,
    top_k=50,
    repetition_penalty=1.0
) -> str:
    input_text = wrap_prompt(question, speaker)
    print(f"[DEBUG] Input prompt:\n{input_text}\n{'-'*40}")

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
    print(f"[DEBUG] Raw decoded output:\n{decoded}\n{'-'*40}")

    if speaker + ":" in decoded:
        response = decoded.split(f"{speaker}:")[-1].strip()
    else:
        response = decoded.strip()

    response = response.replace(f"{speaker}:", "").strip()
    response_lines = response.split("\n")
    unique_lines = []
    seen = set()
    for line in response_lines:
        line_clean = line.strip()
        if line_clean not in seen:
            seen.add(line_clean)
            unique_lines.append(line_clean)
    response = " ".join(unique_lines).strip()

    print(f"[DEBUG] Final cleaned response:\n{response}\n{'='*40}")
    return response


def run_inference_on_data(
    model_path: str,
    input_file: str,
    output_file: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float
):
    speaker = model_path.split('/')[-2].split('_')[0]

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()

    try:
        entries = data[speaker]
    except:
        entries = []
        print("no entries found for speaker:", speaker)
    print(f"\nProcessing entries for: {speaker}")
    
    # Limit to at most 25 questions
    entries = entries[:25]

    for entry in tqdm(entries, desc=f"{speaker}", leave=False):
        question = entry.get("question", "").strip()
        if not question:
            continue

        try:
            response = ask_model(
                question,
                model,
                tokenizer,
                speaker=speaker,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
            entry["response"] = response
        except Exception as e:
            print(f"Error generating response for question: {question[:60]}... — {e}")
            entry["response"] = "ERROR"
        
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nResponses written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned LLM on question dataset.")
    parser.add_argument('--model-path', '-m', required=True, help='Path to the fine-tuned model directory or HF hub ID.')
    parser.add_argument('--input-file', '-i', required=True, help='Path to the input JSON data.')
    parser.add_argument('--output-file', '-o', required=True, help='Path to save the updated JSON with responses.')

    # New generation parameters
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (default: 1.0)')
    parser.add_argument('--top_p', type=float, default=1.0, help='Nucleus sampling probability (default: 1.0)')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling cutoff (default: 50)')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty (default: 1.0)')

    args = parser.parse_args()

    run_inference_on_data(
        args.model_path,
        args.input_file,
        args.output_file,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty
    )
