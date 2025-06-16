import torch
import pickle
import json
import textwrap
import time
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging

# ------------------- Config -------------------

MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"
MAX_MONOLOGUE_CHARS = 5000
MAX_NEW_TOKENS = 500
MAX_FINAL_TOKENS = 700
MAX_MONOLOGUES_PER_SPEAKER = 20

# Suppress HF model loading logs
logging.set_verbosity_error()

# ------------------- Model Loading -------------------

def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        max_memory={i: "35GiB" for i in range(torch.cuda.device_count())}
    )
    return model, tokenizer

# ------------------- Generation -------------------

def generate_text(prompt: str, model, tokenizer, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.7,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------- Preprocessing -------------------

def sample_monologues(monologues: List[str], max_samples: int = MAX_MONOLOGUES_PER_SPEAKER) -> List[str]:
    print(f"Sampling up to {max_samples} longest monologues...")
    sampled = sorted(monologues, key=len, reverse=True)[:max_samples]
    print(f"Sampled {len(sampled)} monologues.")
    return sampled

def chunk_text(monologues: List[str], max_chars: int = MAX_MONOLOGUE_CHARS) -> List[str]:
    print(f"Chunking monologues into max {max_chars}-character chunks...")
    chunks = []
    buffer = ""
    for m in monologues:
        if len(buffer) + len(m) > max_chars:
            chunks.append(buffer.strip())
            buffer = m
        else:
            buffer += "\n\n" + m
    if buffer:
        chunks.append(buffer.strip())
    print(f"Created {len(chunks)} chunks.")
    return chunks

# ------------------- Prompt Templates -------------------

def build_prompt(monologue_chunk: str) -> str:
    return textwrap.dedent(f"""
        Analyze the following monologues and construct a detailed personal profile of the speaker. 
        Describe their personality, interests, communication style, tone, and any notable beliefs or themes. 
        Use clear, analytical language and write in paragraph form.

        Monologues:
        \"\"\"
        {monologue_chunk}
        \"\"\"

        Personal Profile:
    """)

def consolidate_profiles(partial_profiles: List[str], model, tokenizer, max_new_tokens=MAX_FINAL_TOKENS) -> str:
    combined = "\n\n".join(partial_profiles)
    prompt = textwrap.dedent(f"""
        The following are partial analyses of a speaker's personality based on their monologues. 
        Please consolidate these into a single, cohesive personal profile. 
        Avoid repetition, combine overlapping ideas, and ensure the final result reads fluidly and naturally. 
        Focus on clarity, completeness, and insight into the speaker's traits and style.

        Partial Profiles:
        \"\"\"
        {combined}
        \"\"\"

        Final Consolidated Profile:
    """)
    output = generate_text(prompt, model, tokenizer, max_new_tokens=max_new_tokens)
    result_start = output.find("Final Consolidated Profile:")
    return output[result_start + len("Final Consolidated Profile:"):].strip() if result_start != -1 else output.strip()

# ------------------- Profile Generation -------------------

def generate_profile_and_questions(speaker: str, monologues: List[str], model, tokenizer) -> Dict:
    print(f"Generating partial profiles...")
    chunks = chunk_text(monologues)
    partial_profiles = []

    for i, chunk in enumerate(chunks):
        print(f"Generating partial profile {i + 1}/{len(chunks)}...")
        prompt = build_prompt(chunk)

        start_time = time.time()
        try:
            output = generate_text(prompt, model, tokenizer)
        except Exception as e:
            print(f"Error during generation: {e}")
            continue

        duration = time.time() - start_time
        print(f"Completed in {duration:.1f}s")

        profile_start = output.find("Personal Profile:")
        profile_text = output[profile_start + len("Personal Profile:"):].strip() if profile_start != -1 else output
        partial_profiles.append(profile_text)

    if not partial_profiles:
        print("No partial profiles generated.")
        return {"profile": "ERROR: No partial profiles", "questions": []}

    print(f"Consolidating {len(partial_profiles)} partial profiles into final profile...")
    final_profile = consolidate_profiles(partial_profiles, model, tokenizer)
    print(f"Final profile generated.")

    print(f"Generating interview questions...")
    questions = generate_interview_questions(final_profile, model, tokenizer)
    print(f"Generated {len(questions)} questions.")

    return {"profile": final_profile, "questions": questions}



def generate_interview_questions(profile: str, model, tokenizer, max_new_tokens: int = 500) -> List[str]:
    prompt = textwrap.dedent(f"""
        Based on the following personality profile, write a set of 10 interview questions designed to test how well the speaker adheres to this profile in practice.  
        Focus on probing the speaker's personality traits, values, and behaviors that emerge in their communication.
        Profile:
        \"\"\"
        {profile}
        \"\"\"

        Interview Questions:
    """)
    output = generate_text(prompt, model, tokenizer, max_new_tokens=max_new_tokens)
    questions_start = output.find("Interview Questions:")
    if questions_start != -1:
        output = output[questions_start + len("Interview Questions:"):].strip()

    questions = [q.strip("•- \n") for q in output.split("\n") if q.strip()]
    return questions

# ------------------- Main -------------------

def main(monologue_pkl_path: str, output_json_path: str):
    print(f"Loading monologues from: {monologue_pkl_path}")
    with open(monologue_pkl_path, "rb") as f:
        data: Dict[str, List[str]] = pickle.load(f)

    print(f"🚀 Loading model: {MODEL_NAME}")
    model, tokenizer = load_model(MODEL_NAME)
    print(f"Model loaded successfully.")

    results = {}

    for speaker, monologues in data.items():
        print(f"\nProcessing speaker: {speaker} ({len(monologues)} monologues)")
        try:
            sampled = sample_monologues(monologues)
            result = generate_profile_and_questions(speaker, sampled, model, tokenizer)
            results[speaker] = result
        except Exception as e:
            print(f"Error processing {speaker}: {e}")
            results[speaker] = {"profile": "ERROR", "questions": []}

    print(f"\nSaving all profiles and questions to: {output_json_path}")
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done. Profiles and interview questions saved.")



# ------------------- Entrypoint -------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate speaker profiles from monologues.")
    parser.add_argument("--input", type=str, default="/playpen-ssd/smerrill/dataset/monologues.pkl",
                        help="Path to the input pickle file containing agent monologues.")
    parser.add_argument("--output", type=str, default="speaker_profiles.json",
                        help="Path to output JSON file with generated profiles.")

    args = parser.parse_args()
    main(args.input, args.output)
