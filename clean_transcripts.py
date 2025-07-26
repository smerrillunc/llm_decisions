import json
import os
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from tqdm import tqdm


def load_transcripts(path: str):
    """
    Load transcripts from a NumPy file. Expected format: array of dicts with 'speaker' and 'text'.
    """
    return np.load(path, allow_pickle=True)


def sanitize_entries(raw_entries_chunk):
    """
    Convert numpy-loaded entries to pure Python types for JSON serialization.
    """
    return [{k: (v.item() if isinstance(v, np.generic) else v) for k, v in entry.items()} for entry in raw_entries_chunk]

# Prompt template for LLaMA-70B
PROMPT_TEMPLATE = '''
You are a meeting transcript assistant. The following is a raw transcript as a list of entries with possible speaker misattributions or typos. Your task is to:

1. Analyze the transcript and identify each statement's correct speaker name.
2. Correct any name misspellings or misattributions.
3. Output the cleaned transcript in a structured format (e.g., list of dicts) with 'speaker' and 'text'.

Always think step-by-step. Begin by reasoning through any ambiguous names or duplicates, then present the final cleaned transcript.

Transcript:
{raw_transcript}

Cleaned Transcript:
'''


def build_prompt(raw_entries_chunk):
    """
    Build the prompt by injecting a chunk of raw transcript entries.
    """
    sanitized = sanitize_entries(raw_entries_chunk)
    return PROMPT_TEMPLATE.format(raw_transcript=json.dumps(sanitized, indent=2))


def initialize_model(model_name: str = "meta-llama/Meta-Llama-3-70B-Instruct", batch_size=4):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,       # Helps with performance/quality trade-off
        bnb_4bit_quant_type="nf4",            # Recommended for LLaMA-3
        bnb_4bit_compute_dtype=torch.float16  # float16 gives better performance, or bfloat16 for H100
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
        #padding_side='left'   # <--- add this here

    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = model.config.eos_token_id

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        batch_size=batch_size
    )

    pipe.tokenizer = tokenizer
    pipe.max_context = getattr(model.config, 'max_position_embeddings', 4096)
    print(f"Model max context length: {pipe.max_context} tokens (4-bit quantized)")

    return pipe


def clean_transcript(pipe, raw_entries, chunk_size=50, max_retries=3, batch_size=4):
    """
    Clean transcripts using batched inference for improved GPU throughput.
    Splits raw_entries into chunks, builds prompts, and uses batch inference.
    """
    cleaned_entries = []
    total = len(raw_entries)
    idx = 0

    prompt_batches = []
    entry_slices = []
    reserves = []

    # Step 1: Create valid prompt chunks respecting token context limit
    print("🔧 Creating prompt chunks")
    while idx < total:
        print(f"  ➤ Chunk start index: {idx}/{total}")
        end = min(idx + chunk_size, total)
        while end > idx:
            chunk = raw_entries[idx:end]
            prompt = build_prompt(chunk)
            token_len = len(pipe.tokenizer.encode(prompt))
            reserve = max(1, int(token_len * 0.2))
            available = pipe.max_context - token_len
            print(f"    • Trying entries {idx}-{end}, token_len={token_len}, reserve={reserve}, available={available}")
            if available >= reserve or end - idx == 1:
                reserve = min(reserve, available)
                prompt_batches.append(prompt)
                entry_slices.append((idx, end))
                reserves.append(reserve)
                print(f"    ✅ Accepted chunk {idx}-{end} (reserve: {reserve})")
                break
            end -= 1
        if end == idx:
            print(f"    ⚠️ Fallback: single entry {idx} too large, copying raw.")
            cleaned_entries.extend(sanitize_entries(raw_entries[idx:idx+1]))
            idx += 1
        else:
            idx = end

    print(f"✅ Created {len(prompt_batches)} prompt chunks")
    print("🚀 Running Inference")

    # Step 2: Run batch inference in groups
    for i in tqdm(range(0, len(prompt_batches), batch_size), desc="🧠 Batches"):
        batch_prompts = prompt_batches[i:i+batch_size]
        slices = entry_slices[i:i+batch_size]
        batch_reserves = reserves[i:i+batch_size]
        max_reserve = max(batch_reserves)

        print(f"\n📦 Processing batch {i}-{i+len(batch_prompts)} | batch_size={len(batch_prompts)} | max_reserve={max_reserve}")
        for j, (s, e) in enumerate(slices):
            print(f"  └─ Prompt {j} spans entries {s}-{e}")

        success_flags = [False] * len(batch_prompts)
        outputs = None

        for attempt in range(1, max_retries + 1):
            print(f"    🔁 Attempt {attempt}")
            try:
                outputs = pipe(batch_prompts, max_new_tokens=max_reserve, batch_size=batch_size)
                print("    ✅ Inference successful")
                break
            except Exception as e:
                print(f"    ❌ [Batch {i}-{i+len(batch_prompts)}] attempt {attempt} failed: {e}")

        if outputs is None:
            print(f"    ⚠️ Batch {i}-{i+len(batch_prompts)} failed completely. Using fallback.")
            for start, end in slices:
                cleaned_entries.extend(sanitize_entries(raw_entries[start:end]))
            continue

        # Step 3: Parse outputs
        for j, output in enumerate(outputs):
            try:
                print(f"    🔎 Parsing output {j}")
                gen_text = output['generated_text']
                cleaned_json = gen_text.split('Cleaned Transcript:')[-1].strip()
                parsed = _parse_json(cleaned_json)
                cleaned_entries.extend(parsed)
                success_flags[j] = True
                print(f"    ✅ Output {j} parsed successfully")
            except Exception as e:
                print(f"    ❌ [Item {i+j}] parse error: {e}")

        # Fallback for individual failures
        for j, success in enumerate(success_flags):
            if not success:
                start, end = slices[j]
                print(f"    ⚠️ Fallback for item {j} (entries {start}-{end})")
                cleaned_entries.extend(sanitize_entries(raw_entries[start:end]))

    print("🎉 All batches processed.")
    return cleaned_entries



def _parse_json(json_str: str):
    """
    Robust JSON extraction: parse or extract array portion.
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        start, stop = json_str.find('['), json_str.rfind(']')
        if start != -1 and stop != -1:
            try:
                return json.loads(json_str[start:stop+1])
            except json.JSONDecodeError:
                return []
        return []


def process_all(input_path: str, output_path: str, chunk_size: int, max_retries: int, batch_size: int):
    os.makedirs(output_path, exist_ok=True)
    pipe = initialize_model(batch_size=batch_size)
    files = os.listdir(input_path)
    for filename in tqdm(files, desc="Files"):
        print(f"Processing transcript: {filename}")
        raw = load_transcripts(os.path.join(input_path, filename))
        cleaned = clean_transcript(pipe, raw, chunk_size, max_retries, batch_size)
        out_file = os.path.splitext(filename)[0] + '_cleaned.json'
        with open(os.path.join(output_path, out_file), 'w') as f:
            json.dump(cleaned, f, indent=2)
    print("All transcripts processed.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Clean meeting transcripts')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--chunk_size', type=int, default=50)
    parser.add_argument('--max_retries', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()

process_all(args.input, args.output, args.chunk_size, args.max_retries, args.batch_size)
