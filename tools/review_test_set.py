import argparse
import json
import os, sys
sys.path.append('/playpen-ssd/smerrill/llm_decisions')
from utils import train_test_split


def save_data(agent_name, reviewed_data):
    filename = f"{agent_name}_reviewed.json"
    with open(filename, "w") as f:
        json.dump(reviewed_data, f, indent=2)
    print(f"\n✅ Saved {len(reviewed_data)} reviewed examples to {filename}")

def edit_entry(entry):
    print("\n--- Review Example ---")
    print(f"Prompt: {entry['prompt']}")
    print(f"Completion: {entry['completion']}")
    action = input("[k]eep / [d]iscard / [e]dit? ").strip().lower()

    if action == "d":
        return None
    elif action == "e":
        new_prompt = input("Edit Prompt (leave empty to keep): ").strip()
        new_completion = input("Edit Completion (leave empty to keep): ").strip()
        if new_prompt:
            entry['prompt'] = new_prompt
        if new_completion:
            entry['completion'] = new_completion
    return entry

def main():
    parser = argparse.ArgumentParser(description="Review and edit dataset examples.")
    parser.add_argument("--agent_name", type=str, required=True, help="Name of the agent (e.g., grahampaige)")
    args = parser.parse_args()

    try:
        _, dataset, _ = train_test_split(args.agent_name)
    except Exception as e:
        print(f"Error: {e}")
        return

    reviewed = []
    count = 0
    for i, entry in enumerate(dataset):
        result = edit_entry(entry)
        if result:
            reviewed.append(result)
            count += 1
        if count >= 20:
            print("\n🔚 Reviewed 20 examples. Stopping.")
            break

    save_data(args.agent_name, reviewed)

if __name__ == "__main__":
    main()
