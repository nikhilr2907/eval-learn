#!/bin/bash
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=job_%j.log
export HF_HOME=/vol/bitbucket/nr125/.cache/huggingface
export TORCH_HOME=/vol/bitbucket/nr125/.cache/torch
source /vol/bitbucket/nr125/eval_learn_env/bin/activate
cd /vol/bitbucket/nr125/eval-learn
python -u violence_unlearning_demo.py 2>&1 | grep -v "pthread_setaffinity"