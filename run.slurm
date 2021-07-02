#!/bin/bash
#SBATCH --job-name=torch-test    # create a short name for your job
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=24G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=00:60:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=vip
#SBATCH --mail-user=sunxd@ece.ubc.ca

source ~/miniconda3/bin/activate
conda activate base

nvidia-smi
python Baseline_pruning.py
