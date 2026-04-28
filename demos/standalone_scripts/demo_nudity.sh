#!/bin/bash
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=job_%j.log

export HF_HOME=/vol/bitbucket/nr125/.cache/huggingface
export TORCH_HOME=/vol/bitbucket/nr125/.cache/torch
source /vol/bitbucket/nr125/eval_learn_env/bin/activate
cd /vol/bitbucket/nr125/eval-learn

echo "============================================================"
echo " eval-learn  —  Nudity Concept Erasure Demo"
echo "============================================================"
echo ""

# --- Installed plugins ---
echo "Installed techniques and metrics:"
echo "------------------------------------------------------------"
eval-learn plugins
echo ""

# --- Base models ---
echo "Base model registry:"
echo "------------------------------------------------------------"
eval-learn models
echo ""

# --- Config ---
echo "Benchmark config:"
echo "------------------------------------------------------------"
cat examples/demo/quickstart_nudity.json
echo ""
echo ""

# --- Run ---
echo "Running benchmark  (eval-learn run --config examples/demo/quickstart_nudity.json)"
echo "------------------------------------------------------------"
eval-learn run --config examples/demo/quickstart_nudity.json 2>&1 | grep -v "pthread_setaffinity"
echo ""

echo "============================================================"
echo " Done. Results written to results/demo_nudity/"
echo "============================================================"
