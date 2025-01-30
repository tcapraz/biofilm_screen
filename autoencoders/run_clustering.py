#!/bin/bash

#SBATCH -A huber
#SBATCH -t 1-12:00:00
#SBATCH --mem 8G
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C "gpu=2080Ti"

module load GCCcore/11.2.0
module load CUDA/11.5.1



python3 vae_clustering.py
