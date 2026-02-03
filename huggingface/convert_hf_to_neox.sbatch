#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=1:00:00
#SBATCH --output=/projects/a5k/public/logs/hf_to_neox_conversion/hf_to_neox_%j.out

# SLURM job script to convert HuggingFace GPTNeoX model to NeoX checkpoint format
#
# Usage: sbatch convert_hf_to_neox.sbatch <hf_model> [iteration]
#
# Examples:
#   sbatch convert_hf_to_neox.sbatch Kyle1668/sfm-midtraining_mix_unfiltered
#   sbatch convert_hf_to_neox.sbatch geodesic-research/sfm-midtraining_mix_blocklist_filtered 11921

# Parse arguments
HF_MODEL=$1
ITERATION=${2:-0}

if [ -z "$HF_MODEL" ]; then
    echo "Error: Missing required argument"
    echo "Usage: sbatch convert_hf_to_neox.sbatch <hf_model> [iteration]"
    exit 1
fi

echo "========================================="
echo "Converting HuggingFace model to NeoX"
echo "HF Model: $HF_MODEL"
echo "Iteration: $ITERATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================="

# Create log directory if it doesn't exist
mkdir -p /projects/a5k/public/logs/hf_to_neox_conversion

# Set TMPDIR early
export TMPDIR=/projects/a5k/public/tmp
mkdir -p $TMPDIR

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

# Activate UV virtual environment
echo "[VENV] Activating UV environment..."
source /home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv/bin/activate
echo "[VENV] Python: $(which python)"
echo "[VENV] Python version: $(python --version 2>&1)"

# Load required modules
echo "[MODULES] Loading modules..."
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/nccl/2.26.6-1

# Prefer the module NCCL over any wheel-bundled version (required for Slingshot/OFI)
if [[ -n "${NCCL_ROOT:-}" && -f "${NCCL_ROOT}/lib/libnccl.so" ]]; then
  export LD_PRELOAD="${NCCL_ROOT}/lib/libnccl.so:${LD_PRELOAD-}"
fi

# Set HuggingFace environment variables
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# Change to working directory
cd /home/a5k/kyleobrien.a5k/geodesic-gpt-neox/huggingface

# Run the conversion script
echo "[CONVERT] Starting conversion at $(date)"
python convert_hf_gptneox_to_neox.py \
    --hf-model "$HF_MODEL" \
    --iteration "$ITERATION"

EXIT_CODE=$?
echo "[CONVERT] Conversion completed at $(date) with exit code $EXIT_CODE"

exit $EXIT_CODE
