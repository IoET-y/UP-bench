#!/bin/bash
#SBATCH --job-name=RLUPP   # Job name
#SBATCH --output=out_RL_2_100.log      # Output log file
#SBATCH --error=err_RL_2_100.log       # Error log file
#SBATCH --cpus-per-task=6     # Number of CPU cores per task
##SBATCH --gres=gpu:nvidia_a30:1    # Number of GPUs
##SBATCH --nodelist=gu03
#SBATCH --time=48:00:00       # Time limit hrs:min:sec
#SBATCH --gpus=1
python SAC1-Copy1.py

