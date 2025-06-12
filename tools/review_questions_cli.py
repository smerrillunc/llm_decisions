import json
import os
import argparse
from typing import Dict, List, Any


def load_data(filepath: str) -> Dict[str, List[Dict[str, Any]]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ No data file found at: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(filepath: str, data: Dict[str, List[Dict[str, Any]]]):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("✅ Data saved.")


def edit_question(entry: Dict[str, Any]) -> bool:
    print("\n--- Entry ---")
    print(f"Real Response:\n{entry.get('chunk', '')}\n")
    print(f"GPT Summary:\n{entry.get('summary', '')}\n")
    print(f"Question to Review:\n{entry.get('question', '')}\n")

    new_question = input("Enter a new question (or press Enter to keep current): ").strip()
    if new_question:
        entry['question'] = new_question
        print("✅ Question updated.\n")
        return True
    else:
        print("⏭️  No changes made.\n")
        return False


def review_speaker_entries(data: Dict[str, List[Dict[str, Any]]], speaker: str, save_path: str):
    if speaker not in data:
        print(f"⚠️ Speaker '{speaker}' not found in the dataset.")
        return

    entries = data[speaker]
    print(f"\n🎤 Reviewing {len(entries)} entries for: {speaker}")
    modified = False

    for idx, entry in enumerate(entries):
        print(f"\n[{idx + 1}/{len(entries)}]")
        changed = edit_question(entry)
        if changed:
            modified = True
            save_now = input("Save changes now? (y/n): ").strip().lower()
            if save_now == 'y':
                save_data(save_path, data)

    if modified:
        save_data(save_path, data)
    print(f"\n✅ Finished reviewing {speaker}.")


def main():
    parser = argparse.ArgumentParser(description="CLI to review and edit personality question data.")
    parser.add_argument(
        "--file", "-f", required=True,
        help="Path to the JSON file containing the data."
    )
    parser.add_argument(
        "--speaker", "-s", required=False,
        help="Optional: Review only a specific speaker (e.g., 'ellenosborne'). If omitted, all speakers will be reviewed."
    )

    args = parser.parse_args()

    try:
        data = load_data(args.file)

        if args.speaker:
            review_speaker_entries(data, args.speaker, args.file)
        else:
            for speaker in data.keys():
                review_speaker_entries(data, speaker, args.file)

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
