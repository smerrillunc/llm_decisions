import json
import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model, Accelerator


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data, path):
    try:
        # Ensure everything is serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj)  # fallback to string for unknown types

        cleaned_data = make_serializable(data)

        # Actually write to file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        print(f"Data saved to {path}")

    except Exception as e:
        print(f"Failed to save data to {path}: {e}")


def build_judge_prompt(summary, question, response):
    return f"""You are an impartial evaluator judging whether a model’s response reflects the key beliefs and values of a real speaker, based on a provided summary of that speaker’s statements.

Your task is to give a score from 1 to 5 based on how well the model's response captures the core beliefs, attitudes, and values reflected in the summary — even if some minor details are missing.

Be lenient and generous in scoring: as long as the main position and values are reflected accurately and respectfully, the response can score 4 or 5.

- **5** = Strong alignment; clearly reflects core beliefs and values.
- **4** = Generally aligns with the speaker's values; may miss some nuance but still consistent.
- **3** = Partially aligns; contains some core ideas but also notable gaps or ambiguity.
- **2** = Misalignment or oversimplification of the speaker's position.
- **1** = Poor alignment or contradicts the speaker's expressed views.

Return your evaluation as a JSON object with `score` and `explanation` fields. Be concise and focus only on the most important belief alignment.

Summary:
\"\"\"{summary}\"\"\"

Question asked:
\"\"\"{question}\"\"\"

Model's response:
\"\"\"{response}\"\"\"

Evaluate how well the response reflects the summary. Be generous in your interpretation:
"""

def evaluate_entries(data, generator, speaker_filter=None, overwrite=False, output_key="evaluation"):
    for speaker, entries in data.items():
        if speaker_filter and speaker != speaker_filter:
            continue

        for entry in tqdm(entries, desc=f"Evaluating responses for {speaker}"):
            if not overwrite and output_key in entry:
                continue

            prompt = build_judge_prompt(entry['summary'], entry['question'], entry['response'])
            result = generator(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]

            # Try to extract score and explanation
            score = None
            explanation = None
            for line in result.splitlines():
                if line.lower().startswith("score:"):
                    try:
                        score = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.lower().startswith("explanation:"):
                    explanation = line.split(":", 1)[1].strip()

            entry[output_key] = {
                "score": score,
                "explanation": explanation,
                "raw_output": result.strip()
            }
            
def load_judge_model(model_name):
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_name)

    #device_map = infer_auto_device_map(model, max_memory={i: "20GiB" for i in range(torch.cuda.device_count())})
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype="auto", max_memory={i: "35GiB" for i in range(torch.cuda.device_count())})

    return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

def main():
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 evaluate_alignment.py --data_file /playpen-ssd/smerrill/llm_decisions/results/belief_results.json
    parser = argparse.ArgumentParser(description="Evaluate belief alignment using a judge LLM.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the JSON data file.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct",
                        help="HuggingFace model name or path for the judge LLM.")
    parser.add_argument("--speaker", type=str, default=None, help="Limit evaluation to a specific speaker.")
    parser.add_argument("--overwrite", default=True, action="store_true", help="Overwrite existing evaluations.")
    parser.add_argument("--output_key", type=str, default="evaluation", help="Field name to store evaluation result.")

    args = parser.parse_args()

    print("Loading data...")
    data = load_data(args.data_file)

    print(f"Loading judge model: {args.model_name}")
    generator = load_judge_model(args.model_name)

    print("Running evaluations...")
    evaluate_entries(data, generator, speaker_filter=args.speaker, overwrite=args.overwrite, output_key=args.output_key)

    print("Saving results...")
    save_data(data, args.data_file)
    print("Done.")

if __name__ == "__main__":
    main()
