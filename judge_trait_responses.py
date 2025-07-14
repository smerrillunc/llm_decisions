import json
import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import Accelerator
import re

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data, path):
    try:
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        cleaned_data = make_serializable(data)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {path}")
    except Exception as e:
        print(f"Failed to save data to {path}: {e}")

# === Prompt Builders ===


def build_belief_prompt(summary, question, response):
    return f"""You are an impartial evaluator judging whether a model’s response reflects the key beliefs and values of a real speaker, based on a summary.

Return ONLY a JSON object like:
{{"score": 4, "explanation": "..."}} 

Scoring guide:
5 = Strong alignment with core beliefs.
4 = Generally consistent; minor omissions.
3 = Partial alignment; notable ambiguity.
2 = Misalignment or oversimplification.
1 = Contradiction or no alignment.

Summary:
\"\"\"{summary}\"\"\"

Question:
\"\"\"{question}\"\"\"

Model Response:
\"\"\"{response}\"\"\"
"""

def build_personality_prompt(summary, question, response):
    return f"""You are evaluating how well a model captures a person's personality, preferences, or emotional tone based on a summary.

Return ONLY a JSON object like:
{{"score": 4, "explanation": "..."}} 

Scoring guide:
5 = Strong personality expression aligned with summary.
4 = Generally consistent tone and traits.
3 = Some personality present, but generic or inconsistent.
2 = Weak or incorrect tone.
1 = Traits are absent or misrepresented.

Summary:
\"\"\"{summary}\"\"\"

Question:
\"\"\"{question}\"\"\"

Model Response:
\"\"\"{response}\"\"\"
"""

def build_memory_prompt(summary, question, response):
    return f"""You are evaluating how well a model reflects someone's personal memory or experience based on the summary.

Return ONLY a JSON object like:
{{"score": 4, "explanation": "..."}} 

Scoring guide:
5 = Strong realistic extension or affirmation of the memory.
4 = Consistent but surface-level or partial.
3 = Acknowledges memory but vague or flat.
2 = Misrepresents facts or tone.
1 = Ignores or contradicts memory.

Summary:
\"\"\"{summary}\"\"\"

Question:
\"\"\"{question}\"\"\"

Model Response:
\"\"\"{response}\"\"\"
"""


def get_prompt(evaluation_type, summary, question, response):
    if evaluation_type == "belief":
        return build_belief_prompt(summary, question, response)
    elif evaluation_type == "personality":
        return build_personality_prompt(summary, question, response)
    elif evaluation_type == "memory":
        return build_memory_prompt(summary, question, response)
    else:
        raise ValueError(f"Unknown evaluation_type: {evaluation_type}")

# === Evaluation ===

def evaluate_entries(data, generator, speaker_filter=None, overwrite=False, output_key="evaluation", evaluation_type="belief"):
    for speaker, entries in data.items():
        if speaker_filter and speaker != speaker_filter:
            continue
        for entry in tqdm(entries, desc=f"Evaluating responses for {speaker}"):
            if not overwrite and output_key in entry:
                continue

            if 'response' not in entry.keys():
                # In case we only responded to a subset of all questions
                continue
            
            prompt = get_prompt(evaluation_type, entry['summary'], entry['question'], entry['response'])
            result = generator(prompt, max_new_tokens=256, do_sample=True)[0]["generated_text"]

            score = None
            explanation = None

            try:
                # Find the last { ... } block that looks like JSON
                json_candidates = re.findall(r"\{.*?\"score\"\s*:\s*\d+.*?\}", result, re.DOTALL)
                if json_candidates:
                    parsed = json.loads(json_candidates[-1])
                    score = int(parsed.get("score", None))
                    explanation = parsed.get("explanation", "").strip()
                else:
                    print(f"[Warning] No JSON object found in output:\n{result}")
            except Exception as e:
                print(f"[Warning] Failed to parse JSON: {e}\nOutput:\n{result}")


            entry[output_key] = {
                "score": score,
                "explanation": explanation,
                "raw_output": result.strip()
            }

# === Model Loading ===

def load_judge_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# === CLI ===

def main():
    #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6  accelerate launch --num_processes 1 judge_response_alignment.py --data_file /playpen-ssd/smerrill/llm_decisions/results/memory_results.json  --evaluation_type memory --output_key memory_eval
    #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6  accelerate launch --num_processes 1 judge_response_alignment.py --data_file /playpen-ssd/smerrill/llm_decisions/results/personality_results.json  --evaluation_type personality --output_key personality_eval
    #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6  accelerate launch --num_processes 1 judge_response_alignment.py --data_file /playpen-ssd/smerrill/llm_decisions/results/belief_results.json  --evaluation_type belief --output_key belief_eval

    parser = argparse.ArgumentParser(description="Evaluate alignment using a judge LLM.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the JSON data file.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help="Judge model name.")
    parser.add_argument("--speaker", type=str, default=None, help="Limit evaluation to a specific speaker.")
    parser.add_argument("--overwrite", default=True, action="store_true", help="Overwrite existing evaluations.")
    parser.add_argument("--output_key", type=str, default="evaluation", help="Key name for evaluation output.")
    parser.add_argument("--evaluation_type", choices=["belief", "personality", "memory"], default="belief", help="Type of evaluation to perform.")

    args = parser.parse_args()

    print("Loading data...")
    data = load_data(args.data_file)

    print(f"Loading judge model: {args.model_name}")
    generator = load_judge_model(args.model_name)

    print("Running evaluations...")
    evaluate_entries(
        data,
        generator,
        speaker_filter=args.speaker,
        overwrite=args.overwrite,
        output_key=args.output_key,
        evaluation_type=args.evaluation_type
    )

    print("Saving results...")
    save_data(data, args.data_file)
    print("Done.")

if __name__ == "__main__":
    main()
