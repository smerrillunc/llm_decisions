import argparse
import pickle
import json
from collections import defaultdict

def load_dataset(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Use .pkl or .json")

def save_dataset(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[✔] Saved reviewed dataset to: {output_path}")

def build_question_index(agent_dataset):
    """Group agent votes by unique question"""
    q_index = defaultdict(dict)
    for agent, entries in agent_dataset.items():
        for question, vote in entries:
            q_index[question][agent] = vote
    return dict(q_index)

def review_questions(q_index):
    reviewed = {}
    keys = list(q_index.keys())

    print(f"[INFO] {len(keys)} total questions to review.\n")
    for i, question in enumerate(keys):
        print(f"\n--- Question {i+1}/{len(keys)} ---")
        print(f"Question: {question}")
        print("Votes:")
        for agent, vote in q_index[question].items():
            print(f"  - {agent}: {vote}")

        action = input("\n(A)ccept / (E)dit / (S)kip [A/e/s]? ").strip().lower()

        if action in ("", "a"):
            reviewed[question] = q_index[question]
        elif action == "e":
            new_question = input("Enter revised question: ").strip()
            if new_question:
                reviewed[new_question] = q_index[question]
        else:
            print("[⏭] Skipped.")

    return reviewed

def main():
    parser = argparse.ArgumentParser(description="Manually review generated vote questions.")
    parser.add_argument("--dataset", type=str, default="/playpen-ssd/smerrill/llm_decisions/agent_vote_dataset.pkl", help="Path to agent_vote_dataset.pkl or .json")
    parser.add_argument("--output", type=str, default="reviewed_questions.json", help="Path to save reviewed questions")

    args = parser.parse_args()

    print(f"[INFO] Loading dataset from: {args.dataset}")
    agent_dataset = load_dataset(args.dataset)
    q_index = build_question_index(agent_dataset)
    reviewed = review_questions(q_index)
    save_dataset(reviewed, args.output)

if __name__ == "__main__":
    main()
