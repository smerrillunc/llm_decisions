import json
import os
import argparse
import torch
import traceback
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

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


def build_generation_prompt(context, persona_key, personas):
    info = personas.get(persona_key,"")
    return f"You are a school board member.  You will be provided a persona and tasked with responding to the context with according to this persona . Persona:\n{info}\n\nContext:\n\"{context}\"\n\nRespond:\n"


def build_comparison_prompt(reference, response_a, response_b):
    return f"""
You are an impartial evaluator comparing two responses (A and B) to a REFERENCE.

Your ONLY goal is to determine which response most closely matches the **content and tone** of the REFERENCE.

DO reward:
- Similarity in tone, vagueness, informality, or stream-of-consciousness phrasing — **only if these are present in the REFERENCE.**
- Alignment with the key points and main ideas expressed in the REFERENCE.

Return ONLY a JSON object in this format:
{{
  "winner": "A" or "B" or "Tie",
  "justification": "1–2 sentences clearly explaining your choice, consistent with the winner."
}}

Reference:
\"\"\"{reference.strip()}\"\"\"

Response A:
\"\"\"{response_a.strip()}\"\"\"

Response B:
\"\"\"{response_b.strip()}\"\"\"

Output JSON Now:

""".strip()

def parse_json_block(text, key):
    # remove markdown fences and backticks
    clean = text.replace("```","").replace("`","").strip()
    # extract all {...} blocks
    cand, stack, start = [], [], None
    for i,ch in enumerate(clean):
        if ch=="{":
            if not stack: start=i
            stack.append(ch)
        elif ch=="}" and stack:
            stack.pop()
            if not stack and start is not None:
                cand.append(clean[start:i+1])
                start=None
    # try longest first
    for block in sorted(cand, key=len, reverse=True):
        try:
            obj = json.loads(block)
            if key=="selection" and "best_index" in obj: return obj
            if key=="comparison" and "winner" in obj: return obj
            if key not in ("selection","comparison"): return obj
        except: pass
    # fallback
    try:
        return json.loads(clean)
    except:
        print(f"[WARN] Failed to parse JSON for {key}. Raw:\n{clean[:500]}\n")
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
                break  # Don't retry on generation failure, just switch to next mode

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

def evaluate(data, judge_pipe, gen_pipe, personas, tokenizer, max_input, sample_limit, overwrite):
    sample_kwargs = {
        "max_new_tokens": 128,
        "do_sample": True,
        "temperature": 0.9,
        "top_p": 0.9,
        "repetition_penalty":1.2,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": False
    }
    greedy_kwargs = {
        "max_new_tokens": 128,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": False
    }

    for agent, entries in data.items():
        print(f"[INFO] Agent: {agent}")
        for entry in tqdm(entries[:sample_limit], desc=agent):
            prompt, gt = entry.get("prompt",""), entry.get("true_completion","")
            raw_cands = entry.get("model_responses",[])
            cands = [c.strip() for c in raw_cands if c.strip()]
            
            # just use the first response item (in the other script we use GPT to select best)
            best = cands[0]

            # gpt response
            if overwrite or "gpt_response" not in entry:
                gen_p = build_generation_prompt(prompt, agent, personas)
                try:
                    out = gen_pipe(
                        safe_truncate(gen_p, tokenizer, max_input)+"\n",
                        max_new_tokens=250,
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        return_full_text=False
                    )[0]["generated_text"]
                    entry["gpt_response"] = out.strip()
                except Exception as e:
                    print(f"[ERROR] generator failed: {e}")
                    traceback.print_exc()
                    entry["gpt_response"] = ""

            # comparison
            if overwrite or "final_comparison" not in entry:
                cmp_p = build_comparison_prompt(gt, best, entry["gpt_response"])
                cmp = attempt_generation(judge_pipe, cmp_p, "comparison",
                                         tokenizer, max_input, sample_kwargs, greedy_kwargs)
                entry["final_comparison"] = cmp or {"winner":"Unknown","justification":"parse failure"}


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
    args = p.parse_args()

    data = load_data(args.data_file)
    judge_pipe, tokenizer = load_model(args.judge_model)
    gen_pipe = load_model(args.gen_model)[0] if args.gen_model else judge_pipe
    max_input = MODEL_CONTEXT_LIMITS.get(args.judge_model, 4096) - 512
    personas = json.load(open("/playpen-ssd/smerrill/llm_decisions/configs/personas.json","r"))

    evaluate(data, judge_pipe, gen_pipe, personas, tokenizer, max_input,
             args.max_responses, args.overwrite)
    save_data(data, args.data_file)

if __name__=="__main__":
    main()
