#!/bin/bash
#SBATCH --job-name=snacs2.0
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=108:00:00
#SBATCH --output=snacs2_%j.out
#SBATCH --error=snacs2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wss37@georgetown.edu
# ------------------------------
# Tokens
# ------------------------------
export HF_TOKEN=$(cat ~/.hf_token)
echo "Using HF token: ${HF_TOKEN:0:6}*****"

export WANDB_API_KEY="$(cat ~/.wandb_token)"
echo "WANDB token loaded: ${WANDB_API_KEY:0:6}*****"

# ------------------------------
# Load environment
# ------------------------------
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/.venvs/snacs2

# ------------------------------
# W&B login (now that wandb CLI is definitely available)
# ------------------------------
wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1
export WANDB_SILENT=true
export WANDB_CONSOLE=off

# ------------------------------
# Navigate & run your script
# ------------------------------
python3 train.py --model "jhu-clsp/mmBERT-base" --file en-streusle_train.conllulex --dev_file en-streusle_dev.conllulex --do_sweep --multilingual
