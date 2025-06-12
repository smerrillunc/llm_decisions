import os


# SLURM job template
slurm_template = """#!/bin/bash
#SBATCH --output=logs/train_{agent_name}.out
#SBATCH --error=logs/train_{agent_name}.err
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=1
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
module purge
module load anaconda
module load cuda
export LD_LIBRARY_PATH=/work/users/s/m/smerrill/.conda/envs/llm/lib/python3.7/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Run your script
conda run -n unsloth_env python /work/users/s/m/smerrill/LLM/LLM_Train.py --agent_name {agent_name} --wandb_run_name {wandb} --factors {factors}
"""


parameters_list = []

agents = ['kateacuff', 'ellenosborne', 'grahampaige', 'judyle', 'katrinacallsen', 'davidoberg', 'jonnoalcaro']
for factors in [4, 8]:
    for agent in agents:
        wandb = f'{agent}_{str(factors)}'
        tmp = {'agent_name':agent,
               'wandb':wandb,
               'factors':factors}
        parameters_list.append(tmp)



# Submit SLURM jobs
for idx, parameters in enumerate(parameters_list):
    job_script = f"idx.sh"

    with open(job_script, "w") as file:
        file.write(slurm_template.format(**parameters))
    os.system(f"sbatch {job_script}")

print("All jobs submitted.")
