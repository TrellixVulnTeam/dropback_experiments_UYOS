#!/bin/bash
#SBATCH --job-name=bash    
#SBATCH --cpus-per-task=16        
#SBATCH --mem=96G                
#SBATCH --gres=gpu:4             
#SBATCH --time=45:00:00          
#SBATCH --mail-user=sunxd@ece.ubc.ca
##SBATCH --partition=vip

source ~/miniconda3/bin/activate
conda activate base

python training.py --experiment_name prune
