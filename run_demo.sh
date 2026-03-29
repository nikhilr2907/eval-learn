#!/bin/bash
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=job_%j.log
export HF_HOME=/vol/bitbucket/nr125/.cache/huggingface
source /vol/bitbucket/nr125/eval_learn_env/bin/activate
cd /vol/bitbucket/nr125/eval-learn
python nudity_unlearning_demo.py 2>&1 | grep -v "pthread_setaffinity"