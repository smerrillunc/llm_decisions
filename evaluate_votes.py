import argparse
import json
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.8"

from collections import defaultdict
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from accelerate import init_empty_weights
import warnings
import re
from utils import wrap_prompt

def load_question_votes(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def load_model_pipeline(model_path):
    print(f"[INFO] Loading model: {model_path}")
    base_path = model_path.split('/merged')[0]
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with init_empty_weights():
            _ = AutoModelForCausalLM.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype="auto",
        max_memory={i: "35GiB" for i in range(torch.cuda.device_count())}
    )

    return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")


def interpret_answer(generated_text):
    text = generated_text.lower()

    # Patterns that imply an Aye vote
    patterns_aye = [
        r'\bi (move|recommend|propose)\b.*\bapprove\b',
        r'\bi (move|recommend|propose)\b.*\badopt\b',
        r'\bi (move|recommend|propose)\b.*\bauthorize\b',
        r'\bi (support|am in favor|would support|do support|shall support)\b',
        r'\bi vote (yes|aye|in favor)\b',
        r'\b(i )?vote (yes|aye|in favor)\b',
        r'\byes\b',
        r'\baye\b',
        r'\bsecond\b',
        r'\bi agree\b',
        r'\bapprove\b',
        r'\bsupport\b',
        r'\bit should be approved\b',
        r'\bthis is acceptable\b',
        r'\bi (move|propose)\b.*\bresolution\b.*\b(adopt|approve|authorize)\b'
    ]

    # Patterns that imply a Naye vote
    patterns_naye = [
        r'\bi (move|recommend|propose)\b.*\breject\b',
        r'\bi oppose\b',
        r'\boppose\b',
        r'\bvote against\b',
        r'\bi vote no\b',
        r'\bno\b',
        r'\bnaye\b',
        r'\bdisagree\b',
        r'\breject\b',
        r'\bi do not support\b',
        r'\bi (cannot|won\'t|will not|don\'t) support\b',
        r'\bi am not in favor\b',
        r'\bi’m not in favor\b',
        r'\bi oppose this\b',
        r'\bshould not be approved\b'
    ]

    # First pass: Aye detection
    for pattern in patterns_aye:
        if re.search(pattern, text):
            return "Aye"

    # Second pass: Naye detection
    for pattern in patterns_naye:
        if re.search(pattern, text):
            return "Naye"

    # Optional: disqualify procedural motion-only phrases
    if re.search(r'\bi move that\b', text) and not re.search(r'\b(approve|support|yes|aye|favor)\b', text):
        return "Unknown"

    return "Unknown"


def evaluate_models(reviewed_json_path, agent_models, output_path=None):
    data = load_question_votes(reviewed_json_path)
    predictions = defaultdict(list)
    accuracy = {}

    for agent, model_path in agent_models.items():
        print(f"\n=== Evaluating model for agent: {agent} ===")
        try:
            pipe = load_model_pipeline(model_path)
        except Exception as e:
            print(f"[ERROR] Could not load model for {agent}: {e}")
            continue

        correct = 0
        total = 0

        for question, agent_votes in data.items():
            if agent not in agent_votes:
                continue

            true_vote = agent_votes[agent]
            prompt = wrap_prompt(question, agent)

            try:
                # Generate with repetition penalty and deterministic output
                response = pipe(
                    prompt,
                    max_new_tokens=100,
                    do_sample=False,
                    repetition_penalty=1.2
                )[0]["generated_text"]

                # Extract generated completion by removing prompt prefix
                if response.startswith(prompt):
                    completion = response[len(prompt):].strip()
                else:
                    # Fallback if prefix not found
                    completion = response.strip()

                pred_vote = interpret_answer(completion)
                is_correct = pred_vote == true_vote

                # Logging details
                print(f"\n[QUESTION]     {question}")
                print(f"[PROMPT]       {repr(prompt)}")
                print(f"[RAW OUTPUT]   {repr(response)}")
                print(f"[COMPLETION]   {repr(completion)}")
                print(f"[PREDICTED]    {pred_vote} | [ACTUAL] {true_vote} {'✅' if is_correct else '❌'}\n")

                predictions[agent].append({
                    "question": question,
                    "true_vote": true_vote,
                    "pred_vote": pred_vote,
                    "completion": completion,
                    "correct": is_correct
                })

                total += 1
                if is_correct:
                    correct += 1

            except Exception as e:
                print(f"  - ERROR running model for question: {e}")
                predictions[agent].append({
                    "question": question,
                    "true_vote": true_vote,
                    "pred_vote": "ERROR",
                    "completion": "",
                    "correct": False
                })

        acc = round(correct / total * 100, 2) if total > 0 else 0
        accuracy[agent] = {"accuracy": acc, "total": total}
        print(f"[SUMMARY] {agent}: {acc}% accuracy on {total} questions")

        # Free GPU memory
        del pipe
        torch.cuda.empty_cache()

    print("\n=== Accuracy Summary ===")
    for agent, stats in accuracy.items():
        print(f"{agent}: {stats['accuracy']}% ({stats['total']} questions)")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"\n[✔] Saved detailed predictions to: {output_path}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent-specific LLMs on vote prediction.")
    parser.add_argument(
        "--reviewed_json", type=str,
        default='/playpen-ssd/smerrill/llm_decisions/reviewed_questions.json',
        help="Path to reviewed_questions.json"
    )
    parser.add_argument(
        "--output", type=str,
        default='/playpen-ssd/smerrill/llm_decisions/reviewed_questions2.json',
        help="Output path for detailed predictions JSON"
    )
    parser.add_argument(
        "--agent_models", type=str,
        required=True,
        help="Path to JSON file mapping agent names to model paths"
    )

    args = parser.parse_args()

    with open(args.agent_models, "r") as f:
        agent_models = json.load(f)

    evaluate_models(args.reviewed_json, agent_models, args.output)


if __name__ == "__main__":
    main()
