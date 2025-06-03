import os


# SLURM job template
slurm_template = """#!/bin/bash
#SBATCH --output=logs/train_{agent_name}.out
#SBATCH --error=logs/train_{agent_name}.err
#SBATCH --job-name=test
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
conda run -n unsloth_env python /work/users/s/m/smerrill/LLM/LLM_Test.py --agent_name {agent_name} --wandb_run_name {wandb} --model_path {model_path}
"""


parameters_list = []
path = '/work/users/s/m/smerrill/Albemarle/trained_models/'

agents = ['kateacuff', 'ellenosborne', 'grahampaige', 'judyle', 'katrinacallsen', 'davidoberg', 'jonnoalcaro']

for agent in agents:
    for factors in [4, 8]:
        model_path = os.path.join(path, agent + f"_{factors}")
        
        wandb = f'Eval_{agent}_{str(factors)}'
        
        tmp = {'agent_name':agent,
               'wandb':wandb,
               'model_path':model_path}
        parameters_list.append(tmp)



# Submit SLURM jobs
for idx, parameters in enumerate(parameters_list):
    job_script = f"idx.sh"

    with open(job_script, "w") as file:
        file.write(slurm_template.format(**parameters))
    os.system(f"sbatch {job_script}")

print("All jobs submitted.")
