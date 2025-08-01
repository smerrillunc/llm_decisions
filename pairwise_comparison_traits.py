import json
import os
import argparse
import torch
import traceback
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
import re

MODEL_CONTEXT_LIMITS = {
    "meta-llama/Meta-Llama-3-70B-Instruct": 8192,
}

def load_data(path):
    print(f"[INFO] Loading data from: {path}")
    return json.load(open(path, "r", encoding="utf-8"))

def save_data(data, path):
    print(f"[INFO] Saving evaluated data to: {path}")
    def serialize(obj):
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)
    json.dump(serialize(data), open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("[INFO] Save successful.")

def safe_truncate(prompt, tokenizer, max_input_tokens):
    tokens = tokenizer(prompt, truncation=True, max_length=max_input_tokens, return_tensors="pt")
    return tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

def build_generation_prompt(question):
    return f"""You are a helpful assistant. Please answer the following question as clearly and accurately as possible:

Question: {question.strip()}

Answer:"""


def build_belief_prompt(summary, question, response_a, response_b):
    return f"""You are an impartial evaluator judging which of two responses better reflects the beliefs and values of a real speaker, based on a summary.

Return ONLY a JSON object like:
{{"winner": "B", "justification": "..."}}

Scoring guide:
- Strong alignment with core beliefs is best.
- Generic, vague, or contradictory responses are worse.

Summary:
\"\"\"{summary}\"\"\"

Question:
\"\"\"{question}\"\"\"

Response A:
\"\"\"{response_a}\"\"\"

Response B:
\"\"\"{response_b}\"\"\"
"""


def build_personality_prompt(summary, question, response_a, response_b):
    return f"""You are evaluating which response better captures a speaker's personality, preferences, or emotional tone based on a summary.

Return ONLY a JSON object like:
{{"winner": "A", "justification": "..."}}

Scoring guide:
- Expressive, emotionally resonant, and consistent with personality traits is best.
- Generic or tone-deviating responses are worse.

Summary:
\"\"\"{summary}\"\"\"

Question:
\"\"\"{question}\"\"\"

Response A:
\"\"\"{response_a}\"\"\"

Response B:
\"\"\"{response_b}\"\"\"
"""


def build_memory_prompt(summary, question, response_a, response_b):
    return f"""You are evaluating which response better reflects the speaker's personal memory or lived experience based on a summary.

Return ONLY a JSON object like:
{{"winner": "A", "justification": "..."}}

Scoring guide:
- Responses that feel personal, specific, and accurate are best.
- Generic or contradicting responses are worse.

Summary:
\"\"\"{summary}\"\"\"

Question:
\"\"\"{question}\"\"\"

Response A:
\"\"\"{response_a}\"\"\"

Response B:
\"\"\"{response_b}\"\"\"
"""


def get_prompt(evaluation_type, summary, question, response_a, response_b):
    if evaluation_type == "belief":
        return build_belief_prompt(summary, question, response_a, response_b)
    elif evaluation_type == "personality":
        return build_personality_prompt(summary, question, response_a, response_b)
    elif evaluation_type == "memory":
        return build_memory_prompt(summary, question, response_a, response_b)
    else:
        raise ValueError(f"Unknown evaluation_type: {evaluation_type}")


def parse_json_block(text, key, debug=False):
    """
    Extracts the largest valid JSON object from text that matches criteria based on key.

    Args:
        text (str): The raw string output possibly containing JSON.
        key (str): Type of JSON expected ('comparison' or other).
        debug (bool): If True, prints debug info.

    Returns:
        dict or None: Parsed JSON object or None if parsing fails.
    """
    # Clean common markdown fences
    clean = text.replace("```json", "").replace("```", "").replace("`", "").strip()
    
    if debug:
        print(f"[DEBUG] Cleaned text for JSON extraction:\n{clean[:500]}...\n")
    
    # Find candidate JSON blocks by balanced braces
    candidates = []
    stack = []
    start_idx = None
    
    for idx, ch in enumerate(clean):
        if ch == '{':
            if not stack:
                start_idx = idx
            stack.append(ch)
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = clean[start_idx:idx+1]
                    candidates.append(candidate)
                    start_idx = None
    
    # Sort candidates by length descending - assume largest valid JSON first
    candidates = sorted(candidates, key=len, reverse=True)
    
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            # Check for keys depending on type
            if key == "comparison":
                if isinstance(obj, dict) and "winner" in obj and "justification" in obj:
                    if debug:
                        print("[DEBUG] Found valid comparison JSON block.")
                    return obj
            else:
                # For other keys, just return the first valid JSON object
                if debug:
                    print("[DEBUG] Found valid JSON block (non-comparison).")
                return obj
        except json.JSONDecodeError as e:
            if debug:
                print(f"[DEBUG] JSONDecodeError on candidate: {e}")
            continue
    
    # As a fallback, try to parse the entire cleaned text
    try:
        obj = json.loads(clean)
        if key == "comparison":
            if isinstance(obj, dict) and "winner" in obj and "justification" in obj:
                if debug:
                    print("[DEBUG] Parsed entire text as valid comparison JSON.")
                return obj
        else:
            if debug:
                print("[DEBUG] Parsed entire text as valid JSON.")
            return obj
    except json.JSONDecodeError as e:
        print(f"[WARN] Failed to parse JSON for key '{key}'. Error: {e}")
        if debug:
            print(f"[DEBUG] Raw text snippet:\n{clean[:500]}")
        return None

    return None

def attempt_generation(pipe, prompt, parse_key, tokenizer, max_input, sample_kwargs, greedy_kwargs, max_retries=3):
    prompt = safe_truncate(prompt, tokenizer, max_input)
    if not prompt.endswith("\n"):
        prompt += "\n"

    for mode, kwargs in [("sample", sample_kwargs), ("greedy", greedy_kwargs)]:
        for attempt in range(1, max_retries + 1):
            print(f"[INFO] Generation attempt {attempt} using mode '{mode}'")
            try:
                out = pipe(prompt, **kwargs)[0]["generated_text"]
            except Exception as e:
                print(f"[ERROR] Generation error ({mode}): {e}")
                traceback.print_exc()
                break

            print(f"[DEBUG] Raw output ({mode}, attempt {attempt}):\n{out}\n{'-'*40}")
            if not out.strip():
                print(f"[WARN] Empty output on {mode}, retrying...")
                continue

            parsed = parse_json_block(out, parse_key)
            if parsed is not None:
                return parsed
            else:
                print(f"[WARN] Failed to parse JSON on attempt {attempt} (mode: {mode})")

    print(f"[ERROR] All attempts failed for both 'sample' and 'greedy' modes.")
    return None

def evaluate(data, judge_pipe, gen_pipe, tokenizer, max_input, sample_limit, overwrite, evaluation_type):
    sample_kwargs = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.9,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": False
    }
    greedy_kwargs = {
        "max_new_tokens": 256,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": False
    }

    for agent, entries in data.items():
        print(f"[INFO] Agent: {agent}")
        for entry in tqdm(entries[:sample_limit], desc=agent):
            summary = entry.get("summary", "")
            question = entry.get("question", "")
            existing_response = entry.get("response", "")

            # Generate GPT response if missing or overwrite
            if overwrite or "gpt_response" not in entry:
                gen_prompt = build_generation_prompt(question)  # assumes you adapted this to just question
                try:
                    out = gen_pipe(
                        safe_truncate(gen_prompt, tokenizer, max_input) + "\n",
                        **sample_kwargs
                    )[0]["generated_text"]
                    entry["gpt_response"] = out.strip()
                except Exception as e:
                    print(f"[ERROR] generator failed: {e}")
                    traceback.print_exc()
                    entry["gpt_response"] = ""

            # Run comparison with chosen evaluation type prompt
            if overwrite or "final_comparison" not in entry:
                cmp_prompt = get_prompt(
                    evaluation_type,
                    summary,
                    question,
                    existing_response,
                    entry.get("gpt_response", "")
                )
                cmp = attempt_generation(judge_pipe, cmp_prompt, "comparison",
                                            tokenizer, max_input, sample_kwargs, greedy_kwargs)
                entry["final_comparison"] = cmp or {
                    "winner": "Unknown",
                    "justification": "parse failure"
                }
def load_model(model_name):
    print(f"[INFO] Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    pipe = pipeline("text-generation", model=mdl, tokenizer=tok, return_full_text=False)
    return pipe, tok

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True)
    p.add_argument("--judge_model", default="meta-llama/Meta-Llama-3-70B-Instruct")
    p.add_argument("--gen_model", default=None)
    p.add_argument("--max_responses", type=int, default=20)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--evaluation_type", choices=["belief", "personality", "memory"], required=True)

    args = p.parse_args()

    data = load_data(args.data_file)
    judge_pipe, tokenizer = load_model(args.judge_model)
    gen_pipe = load_model(args.gen_model)[0] if args.gen_model else judge_pipe
    max_input = MODEL_CONTEXT_LIMITS.get(args.judge_model, 4096) - 512

    evaluate(data, judge_pipe, gen_pipe, tokenizer, max_input,
             args.max_responses, args.overwrite, args.evaluation_type)
    save_data(data, args.data_file)

if __name__ == "__main__":
    main()
