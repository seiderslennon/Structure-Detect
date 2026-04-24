#!/bin/bash
#SBATCH --job-name=aimusic-precompute
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=gts-gw92-paid
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --constraint=HX00
#SBATCH --mem-per-gpu=64G
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# --- User-tunable settings ---
PROJECT_DIR="/storage/home/hcoda1/3/lseiders3/AIGM-Detection/Structure-Detect"
CONFIG_PATH="/storage/home/hcoda1/3/lseiders3/AIGM-Detection/Structure-Detect/ai_music/configs/SpecTTTra.yaml"
CACHE_DIR="/storage/home/hcoda1/3/lseiders3/scratch/aigmd_feature_cache"
SPLITS="train val test"
# -----------------------------

mkdir -p logs
cd "${PROJECT_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Running from: $(pwd)"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aigm

python -m ai_music.scripts.precompute_features \
  --config "${CONFIG_PATH}" \
  --cache-dir "${CACHE_DIR}" \
  --splits ${SPLITS} \
  --deterministic-start
