import os
import json
import re
import argparse
from collections import defaultdict

def parse_params_from_filename(filename):
    # Matches pattern like _T1.0_P0.95_K50_R1.2_responses.json
    pattern = r'_T([\d.]+)_P([\d.]+)_K(\d+)_R([\d.]+)_responses\.json$'
    match = re.search(pattern, filename)
    if match:
        return tuple(match.groups())
    return None

def merge_grouped_files(grouped_files, input_dir, output_dir, delete_original=False):
    for param_key, file_list in grouped_files.items():
        merged = {}
        successfully_loaded = []

        for file_name in file_list:
            agent_name = file_name.split('_T')[0]
            file_path = os.path.join(input_dir, file_name)

            try:
                with open(file_path, 'r') as f:
                    merged[agent_name] = json.load(f)
                successfully_loaded.append(file_path)
                print(f"[INFO] Loaded: {file_name}")
            except Exception as e:
                print(f"[ERROR] Skipping {file_name}: {e}")

        if not merged:
            print(f"[WARNING] No valid files for param group {param_key}. Skipping output.")
            continue

        # Format output filename
        temp, top_p, top_k, rep_pen = param_key
        output_file = os.path.join(
            output_dir,
            f"test_responses_T{temp}_P{top_p}_K{top_k}_R{rep_pen}.json"
        )

        try:
            with open(output_file, 'w') as f:
                json.dump(merged, f, indent=2)
            print(f"[SUCCESS] Wrote merged file: {output_file}")
        except Exception as e:
            print(f"[ERROR] Writing failed for {output_file}: {e}")
            continue  # Don't delete if write failed

        # Optionally delete source files
        if delete_original:
            for path in successfully_loaded:
                try:
                    os.remove(path)
                    print(f"[INFO] Deleted source file: {path}")
                except Exception as e:
                    print(f"[WARNING] Could not delete {path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge grouped JSON response files.")
    parser.add_argument("--input_dir", type=str, help="Directory containing response files")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save merged files (defaults to input_dir)"
    )
    parser.add_argument(
        "--delete_original", action="store_true",
        help="Delete original files after merging"
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir

    grouped = defaultdict(list)

    for file_name in os.listdir(input_dir):
        if file_name.endswith("_responses.json"):
            params = parse_params_from_filename(file_name)
            if params:
                grouped[params].append(file_name)

    print(f"[INFO] Found {len(grouped)} parameter groups to merge.")
    merge_grouped_files(grouped, input_dir, output_dir, delete_original=args.delete_original)
