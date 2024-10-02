#!/bin/bash

#SBATCH --job-name=llmslurm
#SBATCH --partition=gpuA40x4
#SBATCH --account=<account-name>-delta-gpu
#SBATCH --time=00:30:00
#SBATCH --output=llm.log
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=200g
#SBATCH --cpus-per-task=32
#SBATCH --constraint="projects"


module purge # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)

module load gcc anaconda3_gpu
# create a conda virtual env in the home directory
#conda create --name coin --file requirements.txt
conda activate coin

echo "python path `which python3`"

echo "job is starting on `hostname`"
srun python3 llm.py

# (optionaly) keep node alive for full amount of $TIME
sleep infinity