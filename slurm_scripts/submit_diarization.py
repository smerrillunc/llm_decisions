import os


# SLURM job template
slurm_template = """#!/bin/bash
#SBATCH --job-name=ytdl
#SBATCH --output=logs/ytdl_%A_%a.out
#SBATCH --error=logs/ytdl_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=1
#SBATCH -p a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

module load anaconda

# Run your script
conda run -n ocr-env python /work/users/s/m/smerrill/LLM/diarizationScript.py --video_file {video_file}
"""


parameters_list = []
video_path = '/work/users/s/m/smerrill/Albemarle'
video_files = [x for x in os.listdir(video_path) if '.mp4' in x[-4:]]

for video_file in video_files:
    tmp = {'video_file':os.path.join(video_path, video_file)}
        
    parameters_list.append(tmp)



# Submit SLURM jobs
for idx, parameters in enumerate(parameters_list):
    job_script = f"{parameters['video_file']}.sh"

    with open(job_script, "w") as file:
        file.write(slurm_template.format(**parameters))
    os.system(f"sbatch {job_script}")

print("All jobs submitted.")
