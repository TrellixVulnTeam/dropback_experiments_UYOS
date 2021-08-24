#!/bin/bash
#SBATCH --job-name=bash    
#SBATCH --cpus-per-task=32       
#SBATCH --mem=96G                
#SBATCH --gres=gpu:8             
#SBATCH --time=45:00:00          
#SBATCH --mail-user=sunxd@ece.ubc.ca
##SBATCH --partition=vip

source ~/miniconda3/bin/activate
conda activate base

python baseline_experiment.py
# python dropback_experiment.py
# python prune_experiment.py
# python tl_dropback_experiment.py
# python tl_prune_experiment.py
