import json
import os
import argparse
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re
import traceback


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



def evaluate_data(
    data,
    generator,
    agent_filter=None,
    overwrite=False,
    output_key="gpt_judgment",
    max_responses=None,
    max_retries=3,
    temperature=0.7,
    top_p=0.9
):
    print("[INFO] Beginning evaluation loop.")
    aspects = ["fit", "style"]

    for agent, examples in data.items():
        if agent_filter and agent != agent_filter:
            print(f"[INFO] Skipping agent '{agent}' (filtered).")
            continue

        print(f"[INFO] Evaluating agent: {agent}")
        response_count = 0

        for entry_idx, entry in enumerate(tqdm(examples, desc=f"Processing entries for {agent}")):
            if max_responses is not None and response_count >= max_responses:
                print(f"[INFO] Reached max_responses ({max_responses}) for agent '{agent}'.")
                break

            prompt = entry.get("prompt", "")
            ground_truth = entry.get("true_completion", "")
            responses = entry.get("model_responses", [])
            gpt_response = entry.get("gpt_response", None)

            if output_key not in entry:
                entry[output_key] = []
                existing_evals = set()
            elif not overwrite:
                # Set of response indices already evaluated
                existing_evals = {r["response_idx"] for r in entry[output_key]}
            else:
                entry[output_key] = []
                existing_evals = set()

            # Combine model_responses and gpt_response into a unified evaluation list
            combined_responses = [(i, r) for i, r in enumerate(responses)]

            if gpt_response and (overwrite or "gpt" not in existing_evals):
                combined_responses.append(("gpt", gpt_response))

            for i, response in combined_responses:
                label = f"GPT" if i == "gpt" else f"Response {i}"
                print(f"[DEBUG] Entry {entry_idx}, {label}: {response[:60]}...")

                if not overwrite and i in existing_evals:
                    print(f"[INFO] Skipping {label} (already evaluated).")
                    continue

                print(f"[INFO] Evaluating {label} for entry {entry_idx}")

                for aspect in aspects:
                    for attempt in range(max_retries):
                        print(f"[INFO] Attempt {attempt+1} evaluating aspect '{aspect}'...")
                        full_prompt = build_judging_prompt(prompt, ground_truth, response, aspect)

                        try:
                            result = generator(
                                full_prompt,
                                max_new_tokens=256,
                                do_sample=True,
                                temperature=temperature,
                                top_p=top_p,
                            )[0]["generated_text"]
                        except Exception as e:
                            print(f"[ERROR] Generation failed: {e}")
                            traceback.print_exc()
                            continue

                        try:
                            match = re.findall(r"\{.*?\"score\"\s*:\s*\d+.*?\}", result, re.DOTALL)
                            if match:
                                parsed = json.loads(match[-1])
                                score = parsed.get("score")
                                explanation = parsed.get("explanation", "").strip()

                                print(f"[DEBUG] Parsed JSON: score={score}, explanation={explanation[:60]}")

                                result_entry = {
                                    "response_idx": i,
                                    "aspect": aspect,
                                    "score": score,
                                    "explanation": explanation
                                }
                                entry[output_key].append(result_entry)
                                break  # Exit retry loop on success
                            else:
                                print(f"[WARN] No valid JSON found. Output snippet:\n{result[:200]}")
                        except Exception as e:
                            print(f"[WARN] Failed to parse JSON: {e}\nOutput:\n{result[:200]}")

                    else:
                        print(f"[ERROR] Failed to evaluate aspect '{aspect}' for {label} after {max_retries} retries.")

                response_count += 1  # Count all evaluated responses, including gpt_response



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
    print("Judging complete.")


if __name__ == "__main__":
    main()
