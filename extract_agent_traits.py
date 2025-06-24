import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from typing import List, Dict
import textwrap
import argparse
import pickle
import json
import os


# -------------- Prompts -----------------
PERSONALITY_FILTER_PROMPT = """Does this excerpt reveal anything about the speaker’s personality (e.g., preferences, tendencies, behavior, emotions, or values)?\nAnswer YES or NO. If YES, summarize briefly.\nExcerpt: \"\"\"{chunk}\"\"\"\nAnswer:"""
BELIEF_FILTER_PROMPT = """Does the following conversation excerpt contain any opinions, beliefs, values, or general stances by the speaker?\nAnswer YES or NO. If YES, explain briefly.\nExcerpt: \"\"\"{chunk}\"\"\"\nAnswer:"""
MEMORY_EXTRACTION_PROMPT = """Does this text contain a personal story or memory?\nIf yes, summarize it as a structured fact.\nIf no, say \"NO\".\nExcerpt: \"\"\"{chunk}\"\"\"\nAnswer:"""

PERSONALITY_QUESTION_GENERATION_PROMPT = """Based on the following summary of someone's preferences, tendencies, behavior, emotions, or values, write a direct question that would reveal whether they genuinely possess these qualities:\n\nSummary: \"\"\"{summary}\"\"\"\n\nQuestion:"""
BELIEF_QUESTION_GENERATION_PROMPT = """Based on the following summary of someone's belief or opinion, generate a concise, direct question that would elicit their stance:\n\nSummary: \"\"\"{summary}\"\"\"\n\nQuestion:"""
MEMORY_QUESTION_GENERATION_PROMPT = """Based on the personal story or memory, generate a concise, direct question that invites the person to reflect on or discuss the memory:\n\nSummary: \"\"\"{summary}\"\"\"\n\nQuestion:"""


# -------------- Setup Argument Parsing -----------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run monologue analysis for personality, beliefs, and memory.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help="Huggingface model name or path")
    parser.add_argument("--input", type=str, default="/playpen-ssd/smerrill/dataset/monologues.pkl", help="Path to monologues.pkl")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    return parser.parse_args()

# -------------- Setup Model & Tokenizer -----------------
def setup_model(model_name):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded.")

    print("Loading model with multi-GPU support and offloading...")
    max_memory = {i: "30GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "200GiB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder="offload",
        offload_state_dict=True,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    print("Model loaded.")

    print("Setting up pipeline...")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
    print("Pipeline ready.")
    return pipe

# -------------- Helper Functions -----------------
def chunk_text(text: str, max_words=200) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)
    print(f"Chunked text into {len(chunks)} pieces.")
    return chunks

def query_llm(pipe, prompt: str) -> str:
    print(f"Querying LLM with prompt snippet:\n{prompt[:150]}...")
    outputs = pipe(prompt, max_new_tokens=256)
    completion = outputs[0]['generated_text'][len(prompt):].strip()
    print(f"LLM response snippet:\n{completion[:150]}...\n")
    return completion

# -------------- Processing Pipeline -----------------
def process_monologues(pipe, monologues: Dict[str, List[str]], prompt_template: str, question_template: str) -> Dict[str, List[Dict]]:
    results = {}

    for speaker, texts in monologues.items():
        print(f"\n--- Processing speaker: {speaker} ---")
        speaker_results = []
        total_chunks = 0
        relevant_chunks = 0
        for text_idx, text in enumerate(texts):
            print(f"Processing text #{text_idx + 1} for {speaker} (length {len(text)} chars)...")
            chunks = chunk_text(text)
            total_chunks += len(chunks)
            for chunk_idx, chunk in enumerate(chunks):
                print(f" Processing chunk #{chunk_idx + 1} / {len(chunks)}")
                prompt = prompt_template.format(chunk=chunk)
                response = query_llm(pipe, prompt)
                if response.strip().upper().startswith("YES"):
                    summary = response.strip()[3:].strip()
                    print(f"  -> YES detected. Summary: {summary[:100]}")
                    item = {"chunk": chunk, "summary": summary}
                    q_prompt = question_template.format(summary=summary)
                    question = query_llm(pipe, q_prompt)
                    item["question"] = question
                    speaker_results.append(item)
                    relevant_chunks += 1
                else:
                    print("  -> NO or irrelevant.")
                    continue
                
                if len(speaker_results) >= 25:
                    print("Reached 25 relevant chunks, stopping early.")
                    break

        results[speaker] = speaker_results
        print(f"Finished speaker {speaker}: {relevant_chunks} relevant chunks out of {total_chunks} total.")

    return results

# ----------- Main --------------
if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 1 extract_agent_traits.py
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading monologues from pickle file...")
    with open(args.input, "rb") as f:
        monologues = pickle.load(f)

    pipe = setup_model(args.model)

    print("Starting Personality Alignment filtering...")
    personality_results = process_monologues(pipe, monologues, PERSONALITY_FILTER_PROMPT, PERSONALITY_QUESTION_GENERATION_PROMPT)
    with open(os.path.join(args.output_dir, "personality_results.json"), "w") as f:
        json.dump(personality_results, f, indent=2)
    print("Saved personality_results.json")

    print("\nStarting Beliefs/Values extraction...")
    belief_results = process_monologues(pipe, monologues, BELIEF_FILTER_PROMPT, BELIEF_QUESTION_GENERATION_PROMPT)
    with open(os.path.join(args.output_dir, "belief_results.json"), "w") as f:
        json.dump(belief_results, f, indent=2)
    print("Saved belief_results.json")

    print("\nStarting Memory/Episodic Recall extraction...")
    memory_results = process_monologues(pipe, monologues, MEMORY_EXTRACTION_PROMPT, MEMORY_QUESTION_GENERATION_PROMPT)
    with open(os.path.join(args.output_dir, "memory_results.json"), "w") as f:
        json.dump(memory_results, f, indent=2)
    print("Saved memory_results.json")

    print("\nAll processing complete!")
