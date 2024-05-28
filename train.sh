#!/bin/bash

#SBATCH --job-name=A2Summ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1-0
#SBATCH --partition=batch_ugrad
#SBATCH -o slurm/logs-%A-%x.out

python train.py --dataset TVSum
exit 0