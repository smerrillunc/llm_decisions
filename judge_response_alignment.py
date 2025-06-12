import json
import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import init_empty_weights, Accelerator

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
    return f"""You are an impartial evaluator judging whether a model’s response reflects the key beliefs and values of a real speaker, based on a provided summary of that speaker’s statements.

Your task is to give a score from 1 to 5 based on how well the model's response captures the core beliefs, attitudes, and values reflected in the summary — even if some minor details are missing.

Be lenient and generous in scoring: as long as the main position and values are reflected accurately and respectfully, the response can score 4 or 5.

5 = Strong alignment; clearly reflects core beliefs and values.
4 = Generally aligns with the speaker's values; may miss some nuance but still consistent.
3 = Partially aligns; contains some core ideas but also notable gaps or ambiguity.
2 = Misalignment or oversimplification of the speaker's position.
1 = Poor alignment or contradicts the speaker's expressed views.

Return your evaluation as a JSON object with score and explanation fields. Be concise and focus only on the most important belief alignment.

Summary:
\"\"\"{summary}\"\"\"

Question asked:
\"\"\"{question}\"\"\"

Model's response:
\"\"\"{response}\"\"\"

Evaluate how well the response reflects the summary. Be generous in your interpretation:
"""

def build_personality_prompt(summary, question, response):
    return f"""You are judging how well a model captures someone’s personality — including their preferences, emotions, tendencies, or values — based on a summary of such traits.

Give a score from 1 to 5 based on how accurately the model’s response reflects the personality described in the summary.

5 = Personality traits are clearly and convincingly expressed.
4 = Generally aligned; some nuance missing but still shows consistent tone or values.
3 = Some traits reflected but inconsistent or generic tone.
2 = Weak personality expression or wrong emotional tone.
1 = Personality traits absent or misrepresented.

Return a JSON object with score and explanation fields.

Summary:
\"\"\"{summary}\"\"\"

Question asked:
\"\"\"{question}\"\"\"

Model's response:
\"\"\"{response}\"\"\"
"""

def build_memory_prompt(summary, question, response):
    return f"""You are judging how well a model reflects a speaker’s personal memory, story, or lived experience based on the summary below.

Score from 1 to 5 how accurately and authentically the model conveys or builds upon this memory.

5 = Response clearly affirms or extends the memory in a realistic and meaningful way.
4 = Generally consistent with the memory, with minor gaps or surface-level restatement.
3 = Acknowledges the memory but is vague, shallow, or lacks emotion.
2 = Misrepresents key facts or tone of the memory.
1 = Ignores or contradicts the memory entirely.

Return a JSON object with score and explanation fields.

Summary:
\"\"\"{summary}\"\"\"

Question asked:
\"\"\"{question}\"\"\"

Model's response:
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

            prompt = get_prompt(evaluation_type, entry['summary'], entry['question'], entry['response'])
            result = generator(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]

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

# === Model Loading ===

def load_judge_model(model_name):
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype="auto",
        max_memory={i: "35GiB" for i in range(torch.cuda.device_count())}
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

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
