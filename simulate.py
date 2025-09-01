import argparse
from html import parser
import json
import os
import random
import re
import string
import time
from copy import deepcopy
from collections import deque
from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel
from transformers import AutoTokenizer

# -- Debug Logging Flag --
DEBUG = True

# -- Constants --
CHAIR_NAME = "grahampaige"
ALL_AGENT_NAMES = [
    'ellenosborne', 'davidoberg', 'grahampaige', 'jonnoalcaro',
    'katrinacallsen', 'kateacuff', 'judyle'
]
with open("/playpen-ssd/smerrill/llm_decisions/configs/personas.json") as f:
    PERSONAS = json.load(f)
    
with open('/playpen-ssd/smerrill/llm_decisions/configs/agenda.json') as f:
    agenda = json.load(f)

with open( '/playpen-ssd/smerrill/llm_decisions/configs/micro_profiles.json', "r", encoding="utf-8") as f:
    micro_profiles_data = json.load(f)
    
ALIASES = {
    "ellenosborne": ["ellen osborne", "ms. osborne", "ellen", "osborne", "ms osborne"],
    "davidoberg": ["david oberg", "mr. oberg", "david", "oberg", "mr oberg"],
    "grahampaige": ["graham paige", "mr. paige", "graham", "paige", "mr paige"],
    "jonnoalcaro": ["jonno alcaro", "jonno", "alcaro"],
    "katrinacallsen": ["katrina callsen", "ms. callsen", "katrina", "callsen", "ms callsen"],
    "kateacuff": ["kate acuff", "ms. acuff", "kate", "acuff", "ms acuff", "katherine acuff", "katherine"],
    "judyle": ["judy le", "ms. le", "judy", "le", "ms le", "ms. lee", "ms lee"],
}


# =========================
# UTILITY FUNCTIONS
# =========================


def create_context_card_simulation(speaker, persona_info, topics_list, people_list):
    """
    Generate a structured, LLM-friendly context card for fine-tuning.
    Each persona attribute is split into individual bullet points for clarity.
    Includes reinforced brevity instructions and response format examples.

    Parameters:
    - speaker (str): Name of the persona to emulate.
    - persona_info (str): Description of the persona, traits, style, etc.
    - topics_list (list): Key topics likely to be discussed.
    - people_list (list): Names of participants in the conversation.

    Returns:
    - str: Formatted context card suitable for LLM input.
    """

    # Format topics and people
    topics_str = ", ".join(topics_list)
    people_str = ", ".join(people_list)

    # Split persona_info into individual bullet points
    persona_info_lines = []
    for segment in re.split(r'\.\s+(?=[A-Z]|Tone|Style|Values|Leadership)', persona_info):
        segment = segment.strip()
        if segment:
            persona_info_lines.append(f"- {segment.rstrip('.')}")
    # Add strong brevity instructions
    formatted_persona_info = "\n".join(persona_info_lines)

    # Construct the final context card
    context_card = f"""Persona: {speaker} - {persona_info}
    Conversation Participants: {people_str}
    Topics: {topics_str}
    Instruction: Continue the conversation in this persona. Respond appropriately to the latest turn. Answer in 2-3 Sentences or less.""".strip()

    return context_card.strip()

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def format_conversation(log: list[dict]) -> str:
    return "\n".join(f"{turn['speaker']}: {turn['content']}" for turn in log if turn['content'].strip())

def print_voting_log(title: str, voting_log: list[dict]):
    print(f"\n{'=' * 55}\n{title} VOTING LOG\n{'=' * 55}")
    for entry in voting_log:
        print(f"{entry['speaker']}: {entry['content']}")

def debug_print(*args):
    if DEBUG:
        print(*args)

def round_robin_order(agent_names, start=0):
    idx = start
    while True:
        yield agent_names[idx % len(agent_names)]
        idx += 1


# =========================
# AGENT CLASS
# =========================

class Agent:
    def __init__(
        self,
        name,
        model,
        tokenizer,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,  # Increased penalty
        max_new_tokens: int = 80,         # Lowered default
    ):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.conv_history = []

        # Default generation settings
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens

    def generate_response(
        self,
        prompt: str,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = None,
        max_new_tokens: int = None,
        num_candidates: int = 1,
        agent_names=None,
    ) -> str:
        """
        Generate a single response (string).
        Allows overriding of default generation settings.
        """
        gen_kwargs = {
            "temperature": temperature if temperature is not None else self.temperature,
            "top_k": top_k if top_k is not None else self.top_k,
            "top_p": top_p if top_p is not None else self.top_p,
            "repetition_penalty": (
                repetition_penalty if repetition_penalty is not None else self.repetition_penalty
            ),
            "max_new_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
        }

        candidates = self.generate_candidates(
            prompt, num_candidates, agent_names or [], **gen_kwargs
        )

        return candidates[0]

    def generate_candidates(
        self,
        prompt: str,
        num_candidates: int = 1,
        agent_names=None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = None,
        max_new_tokens: int = None,
        max_attempts: int = 3,
    ):
        """
        Generate multiple candidate responses.
        Uses defaults unless overridden by arguments.
        """
        self.model.set_adapter(self.name)

        gen_kwargs = {
            "do_sample": True,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_k": top_k if top_k is not None else self.top_k,
            "top_p": top_p if top_p is not None else self.top_p,
            "repetition_penalty": (
                repetition_penalty if repetition_penalty is not None else self.repetition_penalty
            ),
            "max_new_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
        }

        candidates = []
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        prompt_len = input_ids.shape[1]

        def is_multi_agent_reply(reply: str) -> bool:
            lower = reply.lower()
            return sum(lower.count(f"{name.lower()}:") for name in (agent_names or [])) > 1

        for _ in range(num_candidates):
            attempt = 0
            while attempt < max_attempts:
                output = self.model.generate(**inputs, **gen_kwargs, eos_token_id=self.tokenizer.eos_token_id,
                                             )
                new_tokens = output[0][prompt_len:]
                reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                if not is_multi_agent_reply(reply):
                    debug_print(f"[FILTER] MULTI_AGENT_REPLY {attempt} accepted reply:\n{reply}")
                    candidates.append(reply)
                    break
                attempt += 1
                debug_print(f"[FILTER] Attempt {attempt} rejected multi-agent reply:\n{reply}")

            if attempt == max_attempts:
                debug_print("[WARNING] Max attempts reached, returning last reply.")
                candidates.append(reply)

        return candidates


# =========================
# VOTING MANAGER
# =========================

def get_formal_alias(agent_key: str):
    for alias in ALIASES.get(agent_key, []):
        if re.match(r"(ms|mr)\.? ", alias.lower()):
            return alias.title().replace("Ms ", "Ms. ").replace("Mr ", "Mr. ")
    return agent_key.title()


def generate_vote(agent, conversation_log, vote_type, tokenizer):

    system_prompt = [
        "You are a school board member participating in a vote on a policy issue.",
        f"Your vote will be cast {'privately' if vote_type == 'private' else 'publicly'}.",
        "You must choose one of the following options: 'yes', 'no', or 'abstain'.",
        "",
        "Instructions:",
        "- Read the recent conversation between board members to understand the context.",
        "- Think step-by-step before making your decision.",
        "- Provide a brief explanation (2–3 sentences) for your vote, demonstrating your reasoning.",
        "- Clearly state your final vote in the format: Vote: yes / no / abstain",
        "",
        "Example:",
        "Conversation Context:",
        "ellenosborne: I think the budget should prioritize special education.",
        "grahampaige: Agreed. Those programs have been underfunded for years.",
        "judyle: I support increasing that funding as well.",
        "",
        "Response:",
        f"{agent.name}: After reviewing the discussion, I believe prioritizing special education funding is the right choice.",
        "It addresses a long-standing need and supports equity in our schools.",
        "Vote: yes"
    ]

    conv = [{"role": "user" if msg['speaker'] != agent.name else "assistant", "content": f"{msg['speaker']}: {msg['content']}"} for msg in conversation_log[-10:]]
    conv.append({"role": "system", "content": '\n'.join(system_prompt)})
    prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) + f'{agent.name}: '
    print(prompt)
    
    vote = {'vote': '', 'comment': ''}
    retries = 0
    
    while (vote['comment'] == '') and (retries < 3):
        vote = parse_vote_from_response(agent.generate_response(prompt))
        retries += 1
        
    return vote

def parse_vote_from_response(response: str) -> dict:
    """
    Extract:
      - vote: the token after 'Vote:' (yes|no|abstain), case-insensitive
      - comment: everything after '<speaker>:' up to 'Vote:'
    """
    resp = response.strip()

    # vote
    vote_match = re.search(r"vote:\s*(yes|no|abstain)\b", resp, re.IGNORECASE)
    vote = vote_match.group(1).lower() if vote_match else "abstain"

    # comment (allow any speaker token before ':')
    comment_match = re.match(r"^[^:]+:\s*(.*?)(?:\s*vote:|$)", resp, re.IGNORECASE | re.DOTALL)
    comment = comment_match.group(1).strip() if comment_match else resp

    return {"vote": vote, "comment": comment}


def tally_and_log_votes(public_votes, private_votes, log, chair_name):
    """
    Tally votes from:
      - dict[str -> {'vote': 'yes'|'no'|'abstain', 'comment': str}]
      - list[{'vote': ...}] or list[str]
      - entries that may only have 'content' containing 'Vote: ...'
    Log a chair announcement as a dict {speaker, content}.
    Return public/private/combined tallies + results.
    """
    def extract_vote_from_any(x: object) -> str:
        # Accept dicts like {'vote': 'yes', 'comment': ...}
        if isinstance(x, dict):
            v = x.get("vote")
            if v:
                return str(v).strip().lower()
            # Fallback: parse from 'content'
            content = x.get("content", "")
            m = re.search(r"vote:\s*(yes|no|abstain)\b", str(content), re.IGNORECASE)
            if m:
                return m.group(1).lower()
            return ""
        # Accept raw strings like "Vote: yes" or "yes"
        if isinstance(x, str):
            m = re.search(r"vote:\s*(yes|no|abstain)\b", x, re.IGNORECASE)
            if m:
                return m.group(1).lower()
            # bare token
            tok = x.strip().lower()
            if tok in {"yes", "no, ", "no", "abstain"}:
                # handle stray comma case; normalize
                return tok.replace(",", "")
            return ""
        return ""

    def safe_vote_count(votes) -> dict:
        counts = {"yes": 0, "no": 0, "abstain": 0}
        # If votes is a dict of name -> vote_obj, iterate over values
        iterable = votes.values() if isinstance(votes, dict) else votes
        for item in iterable:
            v = extract_vote_from_any(item)
            if v in counts:
                counts[v] += 1
            else:
                # treat missing/unknown as abstain
                counts["abstain"] += 1
        return counts

    def result_from_counts(c: dict) -> str:
        if c["yes"] > c["no"]:
            return "approved"
        if c["yes"] < c["no"]:
            return "denied"
        return "tied"  # optional but clearer than defaulting to denied

    # Tally
    public_counts = safe_vote_count(public_votes)
    private_counts = safe_vote_count(private_votes)
    combined_counts = {
        "yes": public_counts["yes"] + private_counts["yes"],
        "no": public_counts["no"] + private_counts["no"],
        "abstain": public_counts["abstain"] + private_counts["abstain"],
    }

    # Results
    public_result = result_from_counts(public_counts)
    private_result = result_from_counts(private_counts)
    combined_result = result_from_counts(combined_counts)

    # Log as a dict (not a string!) so downstream code that expects dicts works
    log.append({
        "speaker": chair_name,
        "content": (
            "Thank you everyone for voting. After counting all the votes, we have "
            f"{public_counts['yes']} (public) / {private_counts['yes']} (private) in favor, "
            f"{public_counts['no']} (public) / {private_counts['no']} (private) against, and "
            f"{public_counts['abstain']} (public) / {private_counts['abstain']} (private) abstaining. "
            f"Results → Public: {public_result}, Private: {private_result}, Combined: {combined_result}. "
            "Thank you everyone for coming, have a nice night."
        )
    })

    return {
        "public": {**public_counts, "result": public_result},
        "private": {**private_counts, "result": private_result},
        "combined": {**combined_counts, "result": combined_result},
    }


def run_voting(vote_prompt: str, vote_type: str, agents: dict, log: list[dict], agent_names: list[str], tokenizer) -> tuple[dict, list[dict]]:
    """
    Collects votes from all agents and the chair, adds them to the log.
    """
    voting_log = deepcopy(log)
    private_log = deepcopy(log)

    votes = {}
    non_chair = [a for a in agent_names if a != CHAIR_NAME]

    # Initial announcement from the chair
    if vote_type == "public":
        announcement = f"Ok, that's enough, we will now vote publicly.  We are voting on {vote_prompt}.  {get_formal_alias(non_chair[0])}, can you please start us off?"
    else:
        announcement = f"Ok, that's enough, we will now vote privately.  We are voting on {vote_prompt}.  If each of you could please record your vote in the chat, I will tally them up and announce the results."

    voting_log.append({"speaker": CHAIR_NAME, "content": announcement})
    private_log.append({"speaker": CHAIR_NAME, "content": announcement})

    for idx, name in enumerate(non_chair):
        agent = agents[name]
        debug_print(f"[VOTE-{vote_type.upper()}] {name} is voting...")

        # Generate vote
        vote = generate_vote(agent, voting_log, vote_type, tokenizer)
        votes[name] = vote

        # Add to transcript
        if vote_type == "public":
            voting_log.append({"speaker": name, "content": vote["comment"]})

            # Announce the vote
            v = vote.get("vote", "").strip().lower()
            if v not in {"yes", "no", "abstain"}:
                voting_log.append({"speaker": CHAIR_NAME, "content": f"{get_formal_alias(name)} abstains from voting."})
            else:
                voting_log.append({"speaker": CHAIR_NAME, "content": f"{get_formal_alias(name)} votes {v}.  Thank you, {get_formal_alias(name)}."})

            # Next agent prompt
            if idx + 1 < len(non_chair):
                next_agent = non_chair[idx + 1]
                voting_log.append({"speaker": CHAIR_NAME, "content": f"{get_formal_alias(next_agent)}, your turn."})
        else:
            # Private vote: recorded but not simulated in conversation
            private_log.append({
                "speaker": name,
                "content": vote["comment"],
                "vote": vote["vote"],
                "private": True
            })

    # Chair votes last
    chair_agent = agents[CHAIR_NAME]
    debug_print(f"[VOTE-{vote_type.upper()}] {CHAIR_NAME} is voting...")

    if vote_type == "public":
        voting_log.append({"speaker": CHAIR_NAME, "content": "Now it's my turn to vote."})
        chair_vote = generate_vote(chair_agent, voting_log, vote_type, tokenizer)
        votes[CHAIR_NAME] = chair_vote
        voting_log.append({"speaker": CHAIR_NAME, "content": chair_vote["comment"]})
    else:
        chair_vote = generate_vote(chair_agent, voting_log, vote_type, tokenizer)
        votes[CHAIR_NAME] = chair_vote
        private_log.append({
            "speaker": CHAIR_NAME,
            "content": chair_vote["comment"],
            "vote": chair_vote["vote"],
            "private": True
        })
        voting_log = private_log  # return the private transcript

    return votes, voting_log

# =========================
# RANKING, DETECTION, LOADERS
# =========================

def detect_forced_next_speaker(evaluator, tokenizer, recent_turns, agents, max_new_tokens=20):
    convo = "\n".join(f"{t['speaker']}: {t['content']}" for t in recent_turns)
    participant_line = ", ".join(agents)

    #prompt = (
    #    f"You are analyzing a school board conversation.\nParticipants: {participant_line}\n"
    #    f"Conversation:\n{convo}\nWho is most expected to speak next? Answer EXACT name or 'None':\n"
    #)
    
    prompt = (
        f"You are analyzing a school board conversation. \n"
        f"Participants: {participant_line}\n"
        f"Conversation:\n{convo}\n\n"
        f"Your task: Determine if the last message in the conversation is directed at any of the listed participants. "
        f"Only consider explicit references or mentions by name. \n"
        f"Instructions:\n"
        f"- Respond with the EXACT name of the school board member being addressed.\n"
        f"- If the message is not directed at any participant, respond with 'None'.\n"
        f"- Do not include any additional text or explanation.\n\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs.input_ids.shape[1]
    with evaluator.disable_adapter():
        out_tokens = evaluator.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
        
    reply = tokenizer.decode(out_tokens[0][prompt_len:], skip_special_tokens=True).strip().lower()

    for name in agents:
        if reply == name.lower():
            return name
    return None

def detect_vote(evaluator, tokenizer, recent_turns, max_new_tokens=20):
    """
    Detects whether a school board conversation includes a call for a vote or motion.

    Args:
        evaluator: the language model.
        tokenizer: tokenizer for the model.
        recent_turns: list of dicts with 'speaker' and 'content'.
        agents: list of participant names.
        max_new_tokens: max tokens to generate.

    Returns:
        True if a vote or motion is called, False otherwise.
    """

    convo_text = "\n".join(f"{t['speaker']}: {t['content']}" for t in recent_turns)

    prompt = (
        f"You are an assistant analyzing a school board conversation.\n"
        f"Conversation:\n{convo_text}\n\n"
        f"Detect if any participant calls for a vote or motion.\n"
        f"Respond with ONLY 'Yes' if a vote or motion is called, and 'No' if not. "
        f"Do not include any explanation, reasoning, or extra text.\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs.input_ids.shape[1]

    # Disable adapters if using one
    with evaluator.disable_adapter():
        # Use deterministic output to reduce variability
        out_tokens = evaluator.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    reply = tokenizer.decode(out_tokens[0][prompt_len:], skip_special_tokens=True).strip().lower()

    return reply == "yes"



def rank_candidates(evaluator, tokenizer, context: str, candidates: list[str]) -> str:
    prompt = context.strip() + "\n\nYou are evaluating a conversaiton.  The only people in this conversation are: Ellen Osborne, Judy Le, Graham Paige Kristina Callsen, Kate Acuff, David Oberg and Jonno Alcaro.  Please rule out any response that addresses people not in the conversation and choose the best continuation:\n"
    for i, c in enumerate(candidates):
        prompt += f"Response {i+1}: {c}\n"
    prompt += "Answer with the NUMBER only:\n"
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs.input_ids.shape[1]
    
    with evaluator.disable_adapter():
        response = evaluator.generate(**inputs, max_new_tokens=10, do_sample=False)
        
    reply = tokenizer.decode(response[0][prompt_len:], skip_special_tokens=True).strip()

    match = re.search(r"\b([1-3])\b", reply)
    return candidates[int(match.group(1)) - 1] if match else random.choice(candidates)

def load_adapters(model_paths, base_model_path):
    base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token



    peft_model = None
    for name, path in model_paths.items():
        path = path.replace("/merged", "")
        print(f"[INFO] Loading adapter for {name} from {path}")
        
        if peft_model == None:
            peft_model = PeftModel.from_pretrained(base, path, is_trainable=False)
        
        peft_model.load_adapter(path, adapter_name=name)

    return peft_model, tokenizer


def build_system_prompt(speaker, personas, all_agent_names):
    persona = personas.get(speaker.lower(), "")
    allowed_names = ", ".join(all_agent_names)
    
    system_prompt = (
        f"You are roleplaying as a school board member named {speaker}. "
        f"Your persona is as follows: {persona} "
        "Always speak as this character, maintaining their distinct tone, style, values, leadership approach, and commonly used phrases. "
        "Speak naturally and realistically, as if participating in a live board meeting. "
        "Use fillers ('uh', 'um', 'you know', 'I mean'), hesitations, false starts, self-corrections, repeated words, partial thoughts, and trailing sentences. "
        "Occasionally interject or interrupt briefly to clarify points, ask questions, or push back, but never speak for anyone else. "
        "You may only reference people in this meeting: {allowed_names}. "
        "Do not reference anyone else, external events, or previous meetings not part of this conversation. "
        "Do NOT include any parenthetical stage directions like (pause), (turning), or (pauses to collect thoughts). "
        "Do not announce your own name or identify yourself; speak as if everyone already knows who you are. "
        "Avoid perfectly polished or overly formal sentences; let your speech feel imperfect, conversational, and human. "
        f"Respond only as {speaker} and do not add commentary or speak as any other participant."
    )
    return system_prompt

# =========================
# MAIN FUNCTION
# =========================

def truncate_conversation(conv, speaker, max_tokens=500, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    """
    Truncate a conversation to approximately max_tokens, preserving whole messages.
    The earliest message is truncated with ellipses if needed.
    
    Args:
        conv (list): List of dicts, each with 'role' and 'content' keys.
        speaker (str): Name of the assistant speaker in the logs.
        max_tokens (int): Max number of tokens to keep.
        model_name (str): Model tokenizer to use.
        
    Returns:
        truncated_conv (list): List of dicts in the same format as conv.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Reverse iterate to accumulate tokens
    tokenized_msgs = []
    total_tokens = 0
    for msg in reversed(conv):
        tokens = tokenizer.encode(msg["content"], add_special_tokens=False)
        tokenized_msgs.append((tokens, msg))
        total_tokens += len(tokens)
        if total_tokens >= max_tokens:
            break
    
    # Reverse back to chronological order
    tokenized_msgs = list(reversed(tokenized_msgs))
    
    # Truncate first message if necessary
    if total_tokens > max_tokens:
        tokens, first_msg = tokenized_msgs[0]
        # Keep only the last part that fits
        keep_tokens = max_tokens - (total_tokens - len(tokens))
        truncated_text = tokenizer.decode(tokens[-keep_tokens:], skip_special_tokens=True)
        first_msg = first_msg.copy()
        speaker=first_msg["content"].split(":")[0]
        first_msg["content"] = f"{speaker}:..." + truncated_text
        tokenized_msgs[0] = (tokens, first_msg)
    
    # Build final conversation format
    truncated_conv = []
    for _, msg in tokenized_msgs:
        truncated_conv.append(msg)
    
    return truncated_conv


def merge_consecutive_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Merge consecutive messages in a conversation that have the same role.
    
    Args:
        messages: A list of dicts, each with 'role' and 'content' keys.
        
    Returns:
        A new list of messages where consecutive messages of the same role are merged.
    """
    if not messages:
        return []
    
    merged = []
    prev_role = messages[0]['role']
    prev_content = messages[0]['content'].strip()
    
    for msg in messages[1:]:
        role = msg['role']
        content = msg['content'].strip()
        if role == prev_role:
            # Merge content with newline
            prev_content += "\n" + content
        else:
            # Push the previous message and start new one
            merged.append({'role': prev_role, 'content': prev_content})
            prev_role = role
            prev_content = content
    
    # Append the last accumulated message
    merged.append({'role': prev_role, 'content': prev_content})
    return merged


def main():

    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=5,6,7 accelerate launch --num_processes=1 simulate.py --base_model meta-llama/Meta-Llama-3-8B-Instruct --config /playpen-ssd/smerrill/llm_decisions/configs/models.json  --agenda_item "Agenda Item No. 3.1: COVID Mask Policy.  Here we will debate weather we should require students to wear masks in the classrooms? We will then vote on the matter at the end." --vote_prompt "Agenda Item No. 3.1: Should we require students to wear masks in classrooms?"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--agenda_item", type=int, required=True)
    parser.add_argument("--save_dir", default="results_simulation")
    parser.add_argument("--max_turns", type=int, default=25)
    parser.add_argument("--num_candidates", type=int, default=3)
    
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling cutoff")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling cutoff")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Penalty for token repetition")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Maximum new tokens to generate")

    args = parser.parse_args()
    with open(args.config) as f:
        model_paths = json.load(f)


    peft_model, tokenizer = load_adapters(model_paths, args.base_model)

    agents = {}
    for name in model_paths.keys():
        agents[name] = Agent(
            name,
            peft_model,
            tokenizer,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens
        )

    seed_message = agenda[args.agenda_item]['agenda_item']
    vote_prompt = agenda[args.agenda_item]['vote_prompt']
    topics_list = agenda[args.agenda_item]['topics']
    people_list = ['ellenosborne', 'davidoberg', 'grahampage', 'jonnoalcaro', 'katrinacallsen', 'kateacuff', 'judyle']
    
    log = [{
        "speaker": CHAIR_NAME,
        "content": f"Welcome everyone. Let's begin discussion on: {seed_message}. Who would like to start?"
    }]
    agent_names = list(model_paths.keys())
    turn_order = deque(agent_names)
    turn_order.remove(CHAIR_NAME)

    last_speaker = CHAIR_NAME  # initialize with chair

    for round_idx in range(args.max_turns):
        recent = log[-1:] if len(log) >= 1 else log
        forced = detect_forced_next_speaker(peft_model, tokenizer, recent, agent_names)

        if forced and forced == last_speaker:
            # If forced speaker is same as last, pick next in turn_order
            speaker = turn_order.popleft()
        else:
            speaker = forced or turn_order.popleft()
            
        # Ensure speaker is not the same as last_speaker
        if speaker == last_speaker:
            # Rotate turn_order until we find someone else
            for _ in range(len(turn_order)):
                turn_order.append(speaker)
                speaker = turn_order.popleft()
                if speaker != last_speaker:
                    break
        
        print('-'*20)
        print("Round Index: {}, Speaker: {}, Forced: {}".format(round_idx, speaker, forced))

        agent = agents[speaker]
        conv = [{"role": "user" if msg['speaker'] != speaker else "assistant", "content": f"{msg['speaker']}: {msg['content']}"} for msg in log]
        conv = truncate_conversation(merge_consecutive_messages(conv), 'assistant', max_tokens=1000, model_name=args.base_model)
        #system_prompt = (
        #    "You are roleplaying as a school board member named {speaker}. "
        #    "Always speak in the voice of {speaker}, and respond only as that character. "
        #    "Do not break character, add commentary, or speak for anyone else."
        #)
        # This was the old system prompt, we will now use the context card
        #system_prompt = build_system_prompt(speaker, PERSONAS, ALL_AGENT_NAMES)
        system_prompt = create_context_card_simulation(speaker, PERSONAS.get(speaker, ""), topics_list, people_list)
        
        # system prompt was first in training set
        conv.insert(0, {"role": "system", "content": system_prompt})
        prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) + f'{speaker}: '
        print(prompt)
        
        response = agent.generate_response(
            prompt,
            agent_names=agent_names,       # list of other agents
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens
        )
        print('---'*20)
        print(response)
        print('---'*20)
        
        log.append({"speaker": speaker, "content": response.strip()})

        if not forced:
            turn_order.append(speaker)
            
        print(last_speaker, "->", speaker)
        last_speaker = speaker  # update last speaker

        # we want to detect if a member called for a vote
        if detect_vote(peft_model, tokenizer, log[-1:]):
            print(f"[{speaker}] Detected vote call!")
            break
        
    public_votes, public_log = run_voting(vote_prompt, "public", agents, log, agent_names, tokenizer)
    private_votes, private_log = run_voting(vote_prompt, "private", agents, log, agent_names, tokenizer)
    vote_summary = tally_and_log_votes(public_votes, private_votes, log, CHAIR_NAME)


    os.makedirs(args.save_dir, exist_ok=True)
    json.dump(public_log, open(f"{args.save_dir}/public_voting.json", "w"), indent=2)

    json.dump(private_log, open(f"{args.save_dir}/private_voting.json", "w"), indent=2)
    json.dump(log, open(f"{args.save_dir}/full_conversation.json", "w"), indent=2)

    # Save votes for analysis
    json.dump(public_votes, open(f"{args.save_dir}/public_votes.json", "w"), indent=2)
    json.dump(private_votes, open(f"{args.save_dir}/private_votes.json", "w"), indent=2)


    print_voting_log("PUBLIC", public_log)
    print_voting_log("PRIVATE", private_log)

if __name__ == "__main__":
    main()