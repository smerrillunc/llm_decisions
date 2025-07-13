import argparse
import json
import os
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


def wrap_prompt(prompt, agent_name):
    system_message = (
        "<|begin_of_text|><|system|>\n\n"
        "You are a school board member and will be asked to cast a vote on a question. "
        "Please think step by step and explain your reasoning before casting your vote. "
        "Then respond by explicitly stating your **Final Vote** as either 'Aye' or 'Naye'.<|eot_id|>\n\n"
    )

    in_context_example = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "unknownspeaker: The school board is voting on whether to allocate additional funds to expand the after-school program. "
        "Do you support this motion?<|eot_id|>\n\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{agent_name}: Expanding the after-school program could provide greater access for students who need supervision "
        "and enrichment opportunities. However, we must also consider our current budget constraints and whether we have sufficient staffing.\n\n"
        "That said, the long-term benefits of supporting student development and family needs outweigh the cost.\n\n"
        "Final Vote: Aye.<|eot_id|>\n\n"
    )

    user_message = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"unknownspeaker: {prompt}<|eot_id|>\n\n"
    )

    assistant_header = f"<|start_header_id|>assistant<|end_header_id|>\n\n{agent_name}:"

    return system_message + in_context_example + user_message + assistant_header


def interpret_answer(generated_text):
    text = generated_text.lower()

    patterns_aye = [
        r'\bfinal vote:.*\baye\b',
        r'\bi (support|am in favor|would support|do support|shall support)\b',
        r'\bi vote (yes|aye|in favor)\b',
        r'\byes\b',
        r'\baye\b',
        r'\bsupport\b',
        r'\bapprove\b'
    ]

    patterns_naye = [
        r'\bfinal vote:.*\bnaye\b',
        r'\bi oppose\b',
        r'\boppose\b',
        r'\bvote against\b',
        r'\bi vote no\b',
        r'\bno\b',
        r'\bnaye\b',
        r'\bdisagree\b',
        r'\breject\b',
        r'\bi do not support\b',
        r'\bi am not in favor\b'
    ]

    for pattern in patterns_aye:
        if re.search(pattern, text):
            return "Aye"
    for pattern in patterns_naye:
        if re.search(pattern, text):
            return "Naye"
    return "Unknown"


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

            attempts = 0
            pred_vote = "Unknown"
            max_attempts = 3
            completion = ""

            while attempts < max_attempts and pred_vote == "Unknown":
                attempts += 1
                print(f"\n[ATTEMPT {attempts}] Generating response for: {question}")

                try:
                    response = pipe(
                        prompt,
                        max_new_tokens=500,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.2
                    )[0]["generated_text"]

                    if response.startswith(prompt):
                        completion = response[len(prompt):].strip()
                    else:
                        completion = response.strip()

                    pred_vote = interpret_answer(completion)

                    print(f"[RAW OUTPUT]   {repr(response)}")
                    print(f"[COMPLETION]   {repr(completion)}")
                    print(f"[INTERPRETED]  {pred_vote}")

                except Exception as e:
                    print(f"[ERROR] Generation failed: {e}")
                    break

            is_correct = pred_vote == true_vote
            print(f"[FINAL VOTE]   {pred_vote} | [ACTUAL] {true_vote} {'✅' if is_correct else '❌'}")

            predictions[agent].append({
                "question": question,
                "true_vote": true_vote,
                "pred_vote": pred_vote,
                "completion": completion,
                "correct": is_correct,
                "attempts": attempts
            })

            total += 1
            if is_correct:
                correct += 1

        acc = round(correct / total * 100, 2) if total > 0 else 0
        accuracy[agent] = {"accuracy": acc, "total": total}
        print(f"[SUMMARY] {agent}: {acc}% accuracy on {total} questions")

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
