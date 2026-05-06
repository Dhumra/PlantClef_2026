#!/bin/bash
# ============================================================
# sbatch_extract.sh - overnight feature extraction
# PlantCLEF 2026, JPEG 4:2:2 q85, scales [6,5,4,3,2,1]
#
# Submit:  sbatch scripts/sbatch_extract.sh
# Monitor: tail -f logs/extract_<jobid>.log
# ============================================================

#SBATCH --job-name=plantclef_extract
#SBATCH --partition=gpu          # change to gpu if you want non-preemptable
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/extract_overnight_%j.log
#SBATCH --mail-type=ALL

# Environment
set -euo pipefail
PROJECT=/scratch3/workspace/seoyunjeong_umass_edu-plantclef/PlantCLEF2026        # update this to your project path
cd "$PROJECT"

mkdir -p logs

echo "==== Job started: $(date) ===="
echo "Node: $SLURM_NODELIST"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Activate conda environment (update env name if different)
module load conda/latest
conda activate odcount

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

# Full extraction
echo ""
echo "==== Extraction ===="
python scripts/extract_overnight.py

echo ""
echo "==== Job finished: $(date) ===="
