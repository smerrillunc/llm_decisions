# === schoolboard_simulation.py ===

# =========================
# IMPORTS AND CONSTANTS
# =========================

import argparse
import json
import os
import random
import re
import string
import time
from copy import deepcopy
from collections import deque

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel

# -- Debug Logging Flag --
DEBUG = True

# -- Constants --
CHAIR_NAME = "grahampaige"
ALL_AGENT_NAMES = [
    'ellenosborne', 'davidoberg', 'grahampaige', 'jonnoalcaro',
    'katrinacallsen', 'kateacuff', 'judyle'
]
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
    def __init__(self, name, model, tokenizer):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.conv_history = []

    def generate_response(self, prompt: str, gen_kwargs=None, agent_names=None) -> str:
        gen_kwargs = gen_kwargs or {
            "do_sample": True,
            "temperature": 0.7,
            "max_new_tokens": 150,
        }
        candidates = self.generate_candidates(prompt, 1, gen_kwargs, agent_names or [])
        return candidates[0]

    def generate_candidates(self, prompt, num_candidates, gen_kwargs, agent_names, max_attempts=3):
        candidates = []
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        prompt_len = input_ids.shape[1]

        def is_multi_agent_reply(reply: str) -> bool:
            lower = reply.lower()
            return sum(lower.count(f"{name.lower()}:") for name in agent_names) > 1

        for _ in range(num_candidates):
            attempt = 0
            while attempt < max_attempts:
                output = self.model.generate(**inputs, **gen_kwargs)
                new_tokens = output[0][prompt_len:]
                reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                if not is_multi_agent_reply(reply):
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

def parse_vote_from_response(response: str) -> dict:
    resp = response.lower().strip()
    resp = re.sub(r"^\w+:", "", resp).strip()

    yes_keywords = ["i vote yes", "i support", "yes", "approve", "agree"]
    no_keywords = ["i vote no", "i oppose", "no", "reject", "disagree"]
    abstain_keywords = ["abstain", "undecided", "no strong opinion", "won’t vote"]

    def match(terms): return any(term in resp for term in terms)

    if match(yes_keywords):
        return {"vote": "yes", "comment": response.strip()}
    elif match(no_keywords):
        return {"vote": "no", "comment": response.strip()}
    elif match(abstain_keywords):
        return {"vote": "abstain", "comment": response.strip()}
    else:
        debug_print(f"[WARNING] Could not parse vote from: {response}")
        return {"vote": "abstain", "comment": response.strip()}

def generate_vote(agent, conversation_log, vote_type: str):

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

    conv = [{"role": "user" if msg['speaker'] != agent.name else "assistant", "content": f"{msg['speaker']}: {msg['content']}"} for msg in conversation_log[-30:]]
    conv.append({"role": "system", "content": system_prompt})
    prompt = agent.tokenizer.apply_chat_template(conv, tokenize=False)
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{agent.name}: "
    print(prompt)
    
    
    vote = {'vote': '', 'comment': ''}
    retries = 0

    
    while (vote['comment'] == '') and (retries < 3):
        vote = parse_vote_from_response(agent.generate_response(prompt))
        retries += 1
        
    return vote

def run_voting(vote_prompt: str, vote_type: str, agents: dict, log: list[dict], agent_names: list[str]) -> tuple[dict, list[dict]]:
    """
    Collects votes from all agents and the chair, adds them to the log.

    - For public voting: comments and votes are broadcast (simulated dialogue).
    - For private voting: votes and reasoning are stored, but not announced to others.
    
    Returns tuple: (votes_dict, updated_log)
    """
    voting_log = deepcopy(log)
    private_log = deepcopy(log)
    
    votes = {}
    non_chair = [a for a in agent_names if a != CHAIR_NAME]

    # Initial announcement from the chair
    if vote_type == "public":
        announcement = f"We are now out of time and will vote publicly.  We are voting on {vote_prompt}.  {get_formal_alias(non_chair[0])}, can you please start us off?"
    else:
        announcement = f"We are now out of time and will vote privately.  We are voting on {vote_prompt}.  If each of you could please record your vote in the chat, I will tally them up and announce the results."

    voting_log.append({"speaker": CHAIR_NAME, "content": announcement})
    private_log.append({"speaker": CHAIR_NAME, "content": announcement})

    for idx, name in enumerate(non_chair):
        agent = agents[name]
        debug_print(f"[VOTE-{vote_type.upper()}] {name} is voting...")
        prior = votes if vote_type == "public" else {}

        # Generate vote
        vote = generate_vote(agent, voting_log, vote_type)
        votes[name] = vote

        # Add to transcript
        if vote_type == "public":
            # Spoken dialogue: visible to everyone
            voting_log.append({"speaker": name, "content": vote["comment"]})
            
            if vote["vote"] == "":
                voting_log.append({"speaker": CHAIR_NAME, "content": f"{get_formal_alias(name)} abstains from voting."})
            else:
                voting_log.append({"speaker": CHAIR_NAME, "content": f"{get_formal_alias(name)} votes {vote['vote']}.  Thank you, {get_formal_alias(name)}."})

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
        chair_vote = generate_vote(chair_agent, voting_log, vote_type)
        votes[CHAIR_NAME] = chair_vote

        voting_log.append({"speaker": CHAIR_NAME, "content": chair_vote["comment"]})
    else:
        chair_vote = generate_vote(chair_agent, voting_log, vote_type)
        votes[CHAIR_NAME] = chair_vote

        private_log.append({
            "speaker": CHAIR_NAME,
            "content": chair_vote["comment"],
            "vote": chair_vote["vote"],
            "private": True
        })

    if vote_type == "private":
        voting_log = private_log
        
    return votes, voting_log


# =========================
# RANKING, DETECTION, LOADERS
# =========================

def detect_forced_next_speaker(evaluator, tokenizer, recent_turns, agents, max_new_tokens=20):
    convo = "\n".join(f"{t['speaker']}: {t['content']}" for t in recent_turns)
    participant_line = ", ".join(agents)

    prompt = (
        f"You are analyzing a school board conversation.\nParticipants: {participant_line}\n"
        f"Conversation:\n{convo}\nWho is most expected to speak next? Answer EXACT name or 'None':\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs.input_ids.shape[1]
    out_tokens = evaluator.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
    reply = tokenizer.decode(out_tokens[0][prompt_len:], skip_special_tokens=True).strip().lower()

    for name in agents:
        if reply == name.lower():
            return name
    return None

def rank_candidates(evaluator, tokenizer, context: str, candidates: list[str]) -> str:
    prompt = context.strip() + "\n\nChoose the best continuation:\n"
    for i, c in enumerate(candidates, 1):
        prompt += f"{i}. {c}\n"
    prompt += "Answer with the NUMBER only:\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs.input_ids.shape[1]
    response = evaluator.generate(**inputs, max_new_tokens=10, do_sample=True)
    reply = tokenizer.decode(response[0][prompt_len:], skip_special_tokens=True).strip()

    match = re.search(r"\b([1-3])\b", reply)
    return candidates[int(match.group(1)) - 1] if match else random.choice(candidates)

def load_adapters(config_path, base_model):
    with open(config_path) as f:
        paths = json.load(f)
    loaded = {}
    for name, path in paths.items():
        path = path.replace('merged','')
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = PeftModel.from_pretrained(base_model, path, device_map="auto", torch_dtype=torch.bfloat16)
        loaded[name] = (model, tokenizer)
    return loaded

def tally_and_log_votes(public_votes, private_votes, log, chair_name):
    def safe_vote_count(votes):
        counts = {"yes": 0, "no": 0, "abstain": 0}
        for v in votes:
            if isinstance(v, dict):
                vote = v.get("vote", "").lower()
            elif isinstance(v, str):
                vote = v.lower()
            else:
                continue  # skip unknown type

            if vote in counts:
                counts[vote] += 1
        return counts

    public_counts = safe_vote_count(public_votes)
    private_counts = safe_vote_count(private_votes)

    vote_pass_public = "approved" if public_counts["yes"] > public_counts["no"] else "denied"
    vote_pass_private = "approved" if private_counts["yes"] > private_counts["no"] else "denied"

    # Log results
    log.append({
        "speaker": chair_name,
        "content": (
            f"Thank you everyone for voting. After counting all the votes, we have "
            f"{public_counts['yes']} (public) / {private_counts['yes']} (private) in favor, "
            f"{public_counts['no']} (public) / {private_counts['no']} (private) against, and "
            f"{public_counts['abstain']} (public) / {private_counts['abstain']} (private) abstaining. "
            f"The motion is {vote_pass_public} (public) / {vote_pass_private} (private). "
            f"Thank you everyone for coming, have a nice night."
        )
    })

    return {
        "public": {**public_counts, "result": vote_pass_public},
        "private": {**private_counts, "result": vote_pass_private}
    }
    
# =========================
# MAIN FUNCTION
# =========================

def main():

    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=5,6,7 accelerate launch --num_processes=1 simulate.py --base_model meta-llama/Meta-Llama-3-8B-Instruct --config /playpen-ssd/smerrill/llm_decisions/configs/models.json  --agenda_item "Agenda Item No. 3.1: COVID Mask Policy.  Here we will debate weather we should require students to wear masks in the classrooms? We will then vote on the matter at the end." --vote_prompt "Agenda Item No. 3.1: Should we require students to wear masks in classrooms?"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--agenda_item", required=True)
    parser.add_argument("--vote_prompt", required=True)
    parser.add_argument("--save_dir", default="results_simulation")
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--num_candidates", type=int, default=3)
    args = parser.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", torch_dtype=torch.bfloat16)
    evaluator = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    eval_tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    adapters = load_adapters(args.config, base)
    agents = {}

    for name, (model, tokenizer) in adapters.items():
        tokenizer.chat_template = "<|begin_of_text|>{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>\n\n{% endfor %}"
        agents[name] = Agent(name, model, tokenizer)

    log = [{
        "speaker": CHAIR_NAME,
        "content": f"Welcome everyone. Let's begin discussion on: {args.agenda_item}. Who would like to start?"
    }]
    agent_names = list(adapters.keys())
    turn_order = deque(agent_names)
    turn_order.remove(CHAIR_NAME)

    for _ in range(args.max_turns):
        recent = log[-3:] if len(log) >= 3 else log
        forced = detect_forced_next_speaker(evaluator, eval_tokenizer, recent, agent_names)
        speaker = forced or turn_order.popleft()

        agent = agents[speaker]
        conv = [{"role": "user" if msg['speaker'] != speaker else "assistant", "content": f"{msg['speaker']}: {msg['content']}"} for msg in log[-30:]]
        prompt = agent.tokenizer.apply_chat_template(conv, tokenize=False)
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{speaker}: "

        candidates = agent.generate_candidates(prompt, args.num_candidates, {
            "do_sample": True, "temperature": 0.7, "max_new_tokens": 200, "top_p": 0.95
        }, ALL_AGENT_NAMES)

        context = format_conversation(log)
        chosen = rank_candidates(evaluator, eval_tokenizer, context, candidates)
        log.append({"speaker": speaker, "content": chosen})

        if not forced:
            turn_order.append(speaker)

    public_votes, public_log = run_voting(args.vote_prompt, "public", agents, log, agent_names)
    private_votes, private_log = run_voting(args.vote_prompt, "private", agents, log, agent_names)

    vote_summary = tally_and_log_votes(public_votes, private_votes, log, CHAIR_NAME)


    os.makedirs(args.save_dir, exist_ok=True)
    json.dump(public_log, open(f"{args.save_dir}/public_voting.json", "w"), indent=2)
    json.dump(private_log, open(f"{args.save_dir}/private_voting.json", "w"), indent=2)
    json.dump(log, open(f"{args.save_dir}/full_conversation.json", "w"), indent=2)

    print_voting_log("PUBLIC", public_log)
    print_voting_log("PRIVATE", private_log)

if __name__ == "__main__":
    main()