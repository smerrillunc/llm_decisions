import argparse
import json
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.8"

from collections import defaultdict
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from accelerate import init_empty_weights


AGENT_MODELS = {
    "kateacuff": "/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/kateacuff_32/merged",          
    "ellenosborne": "/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/ellenosborne_32/mergede",
    "grahampaige": "/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/grahampaige_32/merged",
    "katrinacallsen": "/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/katrinacallsen_32/merged",
    "davidoberg": "/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/davidoberg_32/merged",
    "jonnoalcaro": "/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/jonnoalcaro_32/merged",
    "judyle": "/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/judyle_32/merged",
}

AGENT_MODELS = {
    "kateacuff": "/playpen-ssd/smerrill/trained_models_4bit/meta-llama/Meta-Llama-3-70B-Instruct/kateacuff_16/merged",          
    "ellenosborne": "/playpen-ssd/smerrill/trained_models_4bit/meta-llama/Meta-Llama-3-70B-Instruct/ellenosborne_16/mergede",
    "grahampaige": "/playpen-ssd/smerrill/trained_models_4bit/meta-llama/Meta-Llama-3-70B-Instruct/grahampaige_16/merged",
    "katrinacallsen": "/playpen-ssd/smerrill/trained_models_4bit/meta-llama/Meta-Llama-3-70B-Instruct/katrinacallsen_16/merged",
    "davidoberg": "/playpen-ssd/smerrill/trained_models_4bit/meta-llama/Meta-Llama-3-70B-Instruct/davidoberg_16/merged",
    "jonnoalcaro": "/playpen-ssd/smerrill/trained_models_4bit/meta-llama/Meta-Llama-3-70B-Instruct/jonnoalcaro_16/merged",
    "judyle": "/playpen-ssd/smerrill/trained_models_4bit/meta-llama/Meta-Llama-3-70B-Instruct/judyle_16/merged",
}

def load_question_votes(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def load_model_pipeline(model_path):
    print(f"[INFO] Loading model: {model_path}")
    base_path = model_path.split('/merged')[0]
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)

    with init_empty_weights():
        _ = AutoModelForCausalLM.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype="auto",
        max_memory={i: "35GiB" for i in range(torch.cuda.device_count())}
    )

    return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")


def format_prompt(question):
    return f"{question} Please answer with either 'Yes' or 'No'."


def interpret_answer(generated_text):
    text = generated_text.lower()
    if "yes" in text:
        return "Aye"
    elif "no" in text:
        return "Naye"
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
            prompt = format_prompt(question)

            try:
                output = pipe(prompt, max_new_tokens=20, do_sample=False)[0]["generated_text"]
                pred_vote = interpret_answer(output)
                is_correct = pred_vote == true_vote
                predictions[agent].append({
                    "question": question,
                    "true_vote": true_vote,
                    "pred_vote": pred_vote,
                    "correct": is_correct
                })
                total += 1
                if is_correct:
                    correct += 1

                print(f"  - {question[:60]}... | Predicted={pred_vote} | Actual={true_vote} {'✅' if is_correct else '❌'}")
            except Exception as e:
                print(f"  - ERROR running model for question: {e}")
                predictions[agent].append({
                    "question": question,
                    "true_vote": true_vote,
                    "pred_vote": "ERROR",
                    "correct": False
                })

        acc = round(correct / total * 100, 2) if total > 0 else 0
        accuracy[agent] = {"accuracy": acc, "total": total}
        print(f"[SUMMARY] {agent}: {acc}% accuracy on {total} questions")

        # Clean up
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

    args = parser.parse_args()
    evaluate_models(args.reviewed_json, AGENT_MODELS, args.output)


if __name__ == "__main__":
    main()
