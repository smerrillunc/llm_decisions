import json
import os

def merge_agent_responses(agent_names, input_dir, output_file):
    completion_dict = {}

    for agent_name in agent_names:
        input_path = os.path.join(input_dir, f"{agent_name}_test_responses.json")

        if not os.path.exists(input_path):
            print(f"[WARNING] File not found for agent '{agent_name}': {input_path}. Skipping.")
            continue
        
        try:
            with open(input_path, 'r') as file:
                data = json.load(file)
            completion_dict[agent_name] = data
            print(f"[INFO] Successfully loaded data for agent '{agent_name}'")
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode error in file {input_path}: {e}. Skipping.")
        except Exception as e:
            print(f"[ERROR] Unexpected error reading file {input_path}: {e}. Skipping.")

    if not completion_dict:
        print("[ERROR] No valid agent files found. Exiting without writing output.")
        return

    try:
        with open(output_file, 'w') as file:
            json.dump(completion_dict, file, indent=4)
        print(f"[INFO] Successfully wrote merged responses to: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to write output file: {e}")

if __name__ == "__main__":
    agents = ["ellenosborne", "grahampaige", "katrinacallsen",
              "kateacuff", "jonnoalcaro", "judyle"]
    
    print(f"[INFO] Agents to merge: {agents}")
    
    input_directory = "/playpen-ssd/smerrill/llm_decisions/results"
    output_filepath = "/playpen-ssd/smerrill/llm_decisions/results/test_responses.json"
    
    print(f"[INFO] Starting merge from directory: {input_directory}")
    merge_agent_responses(agents, input_directory, output_filepath)
