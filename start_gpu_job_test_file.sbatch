#!/bin/bash -l

# USAGE INSTRUCTIONS:
# sbatch <this_filename>

#SBATCH --job-name=ingmar_landsat
#SBATCH --partition=gpuA100x8
#SBATCH --account=bbou-delta-gpu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=650g
#SBATCH --cpus-per-task=4

# export local environment variables to the job
#SBATCH --export=ALL

conda activate landsattrend2

python /scratch/bbou/toddn/landsat-delta/landsattrend/07-02_LakeAnalysis_Z056_local.py --current_site_name=SITENAME --startyear=STARTYEAR --endyear=ENDYEAR --process_root=/scratch/bbou/toddn/landsat-delta/landsattrend

# (optionaly) keep node alive for full amount of $TIME
# sleep infinity