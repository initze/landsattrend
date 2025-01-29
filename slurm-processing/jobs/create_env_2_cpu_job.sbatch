#!/bin/bash -l

# USAGE INSTRUCTIONS:
# sbatch <this_filename>

#SBATCH --job-name=create_env
#SBATCH --partition=cpu
#SBATCH --account=bbou-delta-cpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --mem=20g
#SBATCH --cpus-per-task=4

# export local environment variables to the job
#SBATCH --export=ALL

conda env list

mamba env create -f /work/hdd/bbou/toddn/landsat-delta/landsattrend/environment_py38_v2_ray.yml

# (optionaly) keep node alive for full amount of $TIME
# sleep infinity