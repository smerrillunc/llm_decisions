import json
import os
import argparse
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re


def load_data(path):
    print(f"[INFO] Loading data from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data, path):
    print(f"[INFO] Saving evaluated data to: {path}")

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    try:
        cleaned_data = make_serializable(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        print("[INFO] Save successful.")
    except Exception as e:
        print(f"[ERROR] Saving failed: {e}")


def build_judging_prompt(prompt, monologue, model_response, aspect):
    instructions = {
        "fit": (
            "Evaluate how well the model's response matches the ground truth monologue in terms of meaning, intent, and content. It does not need to be word-for-word, but should be semantically and thematically consistent."
        ),
        "style": (
            "Evaluate how closely the model's response matches the tone, voice, and rhetorical style of the monologue. This includes emotional tone, word choice, cadence, and personality."
        )
    }

    base_prompt = f"""
You are an impartial expert evaluator. {instructions[aspect]}

Scoring guide:
5 = Excellent
4 = Good
3 = Fair
2 = Weak
1 = Poor

Prompt (reverse-engineered from monologue):
\"\"\"{prompt}\"\"\"

Reference Monologue:
\"\"\"{monologue}\"\"\"

Model Response:
\"\"\"{model_response}\"\"\"

Respond ONLY with a JSON object like:
{{"score": 4, "explanation": "..."}}

Now output your JSON Object:
"""

    return base_prompt


def evaluate_data(data, generator, agent_filter=None, overwrite=False, output_key="gpt_judgment", max_responses=None):
    print("[INFO] Starting evaluation loop.")
    aspects = ["fit", "style"]

    for i, entry in enumerate(tqdm(data, desc="Evaluating entries")):
        agent = entry.get("agent", None)
        if agent_filter and agent != agent_filter:
            continue

        prompt = entry.get("prompt", "")
        monologue = entry.get("monologue", "")
        responses = entry.get("model_responses", [])

        if output_key not in entry:
            entry[output_key] = []

        if not overwrite and len(entry[output_key]) >= len(responses) * len(aspects):
            print(f"[INFO] Skipping entry {i} (already evaluated).")
            continue

        for idx, response in enumerate(responses):
            print(f"\n[INFO] Evaluating Response {idx} for Entry {i}")
            for aspect in aspects:
                full_prompt = build_judging_prompt(prompt, monologue, response, aspect)
                result = generator(full_prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]

                score = None
                explanation = None
                try:
                    match = re.findall(r"\{.*?\"score\"\s*:\s*\d+.*?\}", result, re.DOTALL)
                    if match:
                        parsed = json.loads(match[-1])
                        score = parsed.get("score")
                        explanation = parsed.get("explanation", "").strip()
                        print(f"[✓] Score {score} | Aspect: {aspect}")
                    else:
                        explanation = f"Warning: No valid JSON found in model output:\n{result.strip()[:200]}"
                        print(explanation)
                        continue
                except Exception as e:
                    explanation = f"Warning: Failed to parse JSON: {e}\nOutput snippet: {result.strip()[:200]}"
                    print(explanation)
                    continue

                entry[output_key].append({
                    "response_idx": idx,
                    "aspect": aspect,
                    "score": score,
                    "explanation": explanation
                })


def load_judge_model(model_name):
    print(f"[INFO] Loading judge model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return pipeline("text-generation", model=model, return_full_text=False, tokenizer=tokenizer)


def main():
    parser = argparse.ArgumentParser(description="Evaluate monologue generation using a judging model.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to JSON file with prompts/monologues/responses.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help="Judge model.")
    parser.add_argument("--agent", type=str, default=None, help="Only evaluate entries for this agent.")
    parser.add_argument("--output_key", type=str, default="gpt_judgment", help="Key to store results.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite previous judgments.")
    parser.add_argument("--max_responses", type=int, default=None, help="Limit number of responses.")

    args = parser.parse_args()

    print(f"[INFO] Loading input data from {args.data_file}")
    data = load_data(args.data_file)
    generator = load_judge_model(args.model_name)

    evaluate_data(
        data=data,
        generator=generator,
        agent_filter=args.agent,
        overwrite=args.overwrite,
        output_key=args.output_key,
        max_responses=args.max_responses
    )

    save_data(data, args.data_file)
    print("[✅] Judging complete.")


if __name__ == "__main__":
    main()
