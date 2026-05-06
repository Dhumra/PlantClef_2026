#!/bin/bash
# ============================================================
# sbatch_extract_nojpeg.sh - overnight feature extraction, no JPEG round-trip
# PlantCLEF 2026, raw tiles, scales [6,5,4,3,2,1]
#
# Ablation counterpart to sbatch_extract.sh (JPEG 4:2:2 q85).
# Logits land in cache_nojpeg/ so both caches coexist.
#
# Submit:  sbatch scripts/sbatch_extract_nojpeg.sh
# Monitor: tail -f logs/extract_nojpeg_<jobid>.log
# ============================================================

#SBATCH --job-name=plantclef_extract_nojpeg
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/extract_nojpeg_%j.log
#SBATCH --mail-type=ALL

# Environment
set -euo pipefail
PROJECT=/scratch3/workspace/seoyunjeong_umass_edu-plantclef/PlantCLEF2026        # update this to your project path
cd "$PROJECT"

mkdir -p logs

echo "==== Job started: $(date) ===="
echo "Node: $SLURM_NODELIST"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

module load conda/latest
conda activate odcount

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

# Extraction - no JPEG round-trip
echo ""
echo "==== Extraction (no JPEG) ===="
python scripts/extract_overnight.py \
    --no-jpeg \
    --cache-dir cache_nojpeg

echo ""
echo "==== Job finished: $(date) ===="
