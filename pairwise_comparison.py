import json
import os
import argparse
import torch
import re
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


personas = {
    "ellenosborn": {
        "Tone": "Professional, clear, empathetic.",
        "Style": "Evidence-based, inclusive, collaborative.",
        "Values": "Equity, transparency, accountability.",
        "Leadership": "Community-first, research-informed, reform-minded.",
        "Phrases": [
            "Grounded in research…",
            "In service of equity…",
            "Shared responsibility…",
            "To build trust…"
        ]
    },
    "davidoberg": {
        "Tone": "Formal, engaging, calm, with light humor.",
        "Style": "Clear, fair, respectful, and constructive.",
        "Values": "Fairness, equity, education, social justice.",
        "Leadership": "Collaborative, open-minded, pragmatic idealist.",
        "Focus": "Student well-being, especially vulnerable groups, and community growth.",
        "Phrases": [
            "With respect to history and context…",
            "Balancing idealism with pragmatism…",
            "Seeking common ground…",
            "Committed to fairness and inclusion…"
        ]
    },
    "grahampage": {
        "Tone": "Warm, sincere, storytelling, empathetic.",
        "Style": "Formal, structured, professional, appreciative.",
        "Values": "Compassion, equity, respect, community connection.",
        "Leadership": "Collaborative, inclusive, focused on eliminating systemic barriers.",
        "Focus": "Supporting all students, human connection, and systemic change.",
        "Phrases": [
            "Guided by compassion and equity…",
            "Honoring the dedication of educators…",
            "Working to eliminate barriers and opportunity gaps…",
            "Together, we build community and support every student…"
        ]
    },
    "jonnothanalcaro": {
        "Tone": "Clear, calm, respectful, with measured urgency when needed.",
        "Style": "Concise, structured, simple language for complex ideas.",
        "Values": "Equality, racial justice, data integrity, student empowerment.",
        "Leadership": "Collaborative, reflective, inclusive, community-focused.",
        "Focus": "Evidence-based, innovative, equitable student outcomes.",
        "Phrases": [
            "Grounded in data and local context…",
            "Listening carefully to all perspectives…",
            "Committed to fairness and empowerment…",
            "Inviting open and respectful dialogue…"
        ]
    },
    "katrinacallsen": {
        "Tone": "Clear, precise, neutral, respectful.",
        "Style": "Concise, transparent, methodical, patient.",
        "Values": "Transparency, accountability, inclusivity, social responsibility.",
        "Leadership": "Community-centered, participatory, respectful, organized.",
        "Focus": "Open dialogue, democratic governance, public welfare.",
        "Phrases": [
            "Committed to transparency and accountability…",
            "Encouraging respectful and inclusive dialogue…",
            "Focused on community-centered solutions…",
            "Ensuring accessibility and public engagement…"
        ]
    },
    "kateacuff": {
        "Tone": "Clear, precise, professional, solution-focused.",
        "Style": "Respectful, open-minded, inquisitive, collaborative.",
        "Values": "Equity, anti-racism, transparency, inclusivity.",
        "Leadership": "Data-driven, community-engaged, growth-oriented, long-term focused.",
        "Focus": "Systemic improvement, student celebration, social and health challenges.",
        "Phrases": [
            "Guided by data and equity…",
            "Committed to transparency and collaboration…",
            "Seeking solutions through community input…",
            "Fostering sustainable, inclusive change…"
        ]
    },
    "judyle": {
        "Tone": "Measured, reflective, respectful, hopeful.",
        "Style": "Precise, detailed, introspective, approachable.",
        "Values": "Equity, inclusion, social responsibility, evidence-based decisions.",
        "Leadership": "Innovative, challenging status quo, data-informed, inclusive.",
        "Focus": "Systemic change, marginalized communities, collaborative problem-solving.",
        "Phrases": [
            "Guided by equity and social justice…",
            "Balancing introspection with action…",
            "Committed to inclusive and data-driven dialogue…",
            "Advocating for a diverse and supportive environment…"
        ]
    }
}


def load_data(path):
    print(f"[INFO] Loading data from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data, path):
    print(f"[INFO] Saving evaluated data to: {path}")
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)
    cleaned = make_serializable(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print("[INFO] Save successful.")

def build_selection_prompt(candidates, persona_key=None, personas=None):
    if not candidates:
        return None
    choices = "\n".join(f"{i}: {c}" for i, c in enumerate(candidates))
    persona_info = ""
    if persona_key and personas and persona_key in personas:
        persona_info = f"Persona Description:\n{personas[persona_key]}"
    return f"""
You are an impartial evaluator. Your task is to:
1. Review a list of candidate model responses.
2. Select the most appropriate continuation of the conversation.
3. Ensure the selected response aligns with the given persona’s tone, style, values, and leadership approach.

Criteria:
- Logical flow and conversational continuity.
- Alignment with the persona’s communication style and core values.
- Clarity, coherence, and relevance.

Persona:
{persona_info}

Candidates:
{choices}

Respond ONLY with a JSON object, in the following format:
{{
  "best_index": <integer index of the best candidate, or null if none>,
  "reasoning": "A short explanation for your choice."
}}
Output your JSON object now:
"""

def build_generation_prompt(context, persona_key, personas):
    persona_info = personas.get(persona_key, "")
    return f'''You are a helpful assistant. Using the persona description below, craft a response that reflects their tone, style, values, and leadership approach.

Persona Description:
{persona_info}

Dialogue Context:
"{context}"

Please respond thoughtfully, clearly, and consistently with the persona's communication style and core values.
'''

def build_comparison_prompt(ground_truth, model_resp, gpt_resp):
    return f'''You are an impartial evaluator. Compare two responses against the human reference and decide which is better overall.

Reference:
"{ground_truth}"

Response A (Fine-Tuned model):
"{model_resp}"

Response B (GPT-generated):
"{gpt_resp}"

Return ONLY a JSON object:
{{
  "winner": "A" or "B",
  "justification": "..."
}}
Output your JSON object now:
'''

def parse_json_block(text, key):
    import json
    # Clean markdown code blocks
    text = re.sub(r"^``````$", "", text.strip(), flags=re.IGNORECASE)
    
    # Strategy: Find last valid JSON object in response
    candidates = []
    stack = []
    start_idx = None
    
    for i, char in enumerate(text):
        if char == '{':
            if not stack:  # New top-level object
                start_idx = i
            stack.append(char)
        elif char == '}' and stack:
            stack.pop()
            if not stack and start_idx is not None:  # Complete top-level object
                candidate = text[start_idx:i+1]
                candidates.append(candidate)
                start_idx = None  # Reset for next object
    
    # Try candidates from last to first (most recent likely response)
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    
    # Fallback: Try direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"[WARN] Failed to parse JSON for {key}. Raw output:\n{text[:500]}...\n")
        return {"best_index": None, "reasoning": f"Could not parse output. Raw: {text[:200]}..."}

def evaluate(data, judge_gen, generator, sample_limit=20, overwrite=False):
    for agent, entries in data.items():
        print(f"[INFO] Processing agent: {agent} (up to {sample_limit} samples)")
        for entry in tqdm(entries[:sample_limit], desc=f"Entries for {agent}"):
            gt = entry.get("true_completion", "").strip()
            raw_cands = entry.get("model_responses", [])
            cands = [c.strip() for c in raw_cands if c.strip()]  # Filter empty/blank

            if overwrite or "best_selection" not in entry:
                sel_prompt = build_selection_prompt(cands, agent, personas)
                
                if sel_prompt:
                    try:
                        sel_out = judge_gen(sel_prompt, max_new_tokens=128)[0]["generated_text"]
                        print("__________________________________")
                        print(sel_out)
                        print("__________________________________")

                    except Exception as e:
                        print(f"[ERROR] judge_gen failed for selection: {e}")
                        sel_out = ""
                        continue
                    sel = parse_json_block(sel_out, "selection")
                    if sel is None:
                        continue
                    
                entry["best_selection"] = sel

            sel_obj = entry.get("best_selection") or {}
            best_idx = sel_obj.get("best_index")
            best_resp = cands[best_idx] if best_idx is not None and 0 <= best_idx < len(cands) else ""
            # 2) Generate GPT response
            if overwrite or "gpt_response" not in entry:
                gen_prompt = build_generation_prompt(entry.get("prompt", ""), agent, personas)
                try:
                    gen_out = generator(gen_prompt, max_new_tokens=128)[0]["generated_text"]
                    entry["gpt_response"] = gen_out.strip()
                except Exception as e:
                    print(f"[ERROR] generator failed: {e}")
                    entry["gpt_response"] = ""
                    continue

            # 3) Compare responses
            if overwrite or "final_comparison" not in entry:
                if not best_resp.strip():
                    print("NO BEST RESP, SKIPPING COMPARISON")
                else:
                    cmp_prompt = build_comparison_prompt(gt, best_resp, entry.get("gpt_response", ""))
                    try:
                        cmp_out = judge_gen(cmp_prompt, max_new_tokens=128)[0]["generated_text"]
                        cmp = parse_json_block(cmp_out, "comparison")
                        if cmp is None:
                            cmp = {
                                "winner": "Unknown",
                                "justification": f"Could not parse comparison output. Raw: {cmp_out[:200]}..."
                            }
                        entry["final_comparison"] = cmp
                        print('----------------------------------')
                        print(cmp_out)
                        print('----------------------------------')
                    except Exception as e:
                        print(f"[ERROR] judge_gen failed for comparison: {e}")
                        entry["final_comparison"] = {
                            "winner": "A",
                            "justification": f"Exception during comparison: {e}"
                        }

def load_judge_model(model_name):
    print(f"[INFO] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return pipeline("text-generation", model=model, return_full_text=False, tokenizer=tokenizer)

def main():
    #  CUDA_VISIBLE_DEVICES=0,7,2,3,4,5,6 accelerate launch --num_processes 1 --main_process_port 0 pairwise_comparison.py --data_file /playpen-ssd/smerrill/llm_decisions/results/test_responses.json
    parser = argparse.ArgumentParser(description="Evaluate limited samples per agent.")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--gen_model", type=str, default=None)
    parser.add_argument("--max_responses", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data = load_data(args.data_file)
    judge_gen = load_judge_model(args.judge_model)
    generator = load_judge_model(args.gen_model) if args.gen_model else judge_gen

    evaluate(data, judge_gen, generator, sample_limit=args.max_responses, overwrite=args.overwrite)
    save_data(data, args.data_file)

if __name__ == "__main__":
    main()
