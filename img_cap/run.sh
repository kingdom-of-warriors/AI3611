#!/bin/bash

#SBATCH -p dgx2
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

module load openjdk/11.0.17_8

python main.py train_evaluate --config_file configs/resnet101_attention.yaml

