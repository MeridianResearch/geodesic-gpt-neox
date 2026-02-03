#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=1:00:00
#SBATCH --output=/projects/a5k/public/logs/checkpoint_pipeline/checkpoint_pipeline_%j.out

# SLURM job script to upload and evaluate a single checkpoint
# This script handles the complete checkpoint processing pipeline:
# 1. Upload original NeoX checkpoint to HuggingFace
# 2. Convert to HuggingFace format
# 3. Upload converted checkpoint to HuggingFace
# 4. Run evaluations
#
# Job naming: Uses convert_<experiment_name> to enable singleton dependencies
# This ensures only one checkpoint conversion runs at a time per experiment,
# preventing HuggingFace rate limiting while allowing different experiments
# to process checkpoints in parallel.
#
# Usage: sbatch --job-name=convert_<experiment_name> --dependency=singleton \
#               upload_and_evaluate_checkpoint.sbatch <experiment_name> <checkpoint_number>
#
# Note: Job name MUST be set via sbatch --job-name flag (not in this file)
#       to support singleton dependencies per experiment

# Parse arguments first, before activating venv
EXPERIMENT_NAME=$1
CHECKPOINT_NUMBER=$2
SKIP_EVAL=${3:-"--skip-eval"}  # Default to skipping eval
HF_ORG=${4:-"Kyle1668"}

if [ -z "$EXPERIMENT_NAME" ] || [ -z "$CHECKPOINT_NUMBER" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch process_checkpoint_slurm.sbatch <experiment_name> <checkpoint_number> [--skip-eval|--eval] [hf_org]"
    exit 1
fi

echo "========================================="
echo "Processing checkpoint for $EXPERIMENT_NAME"
echo "Checkpoint: $CHECKPOINT_NUMBER"
echo "HF Org: $HF_ORG"
echo "Skip Eval: $SKIP_EVAL"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================="

echo "[SETUP] Starting setup phase at $(date)"
echo "[NODE] Node name: $(hostname)"
echo "[NODE] Available memory: $(free -h | grep Mem)"
echo "[NODE] Available disk space: $(df -h /tmp | tail -1)"
echo "[NODE] Current working directory: $(pwd)"

echo "[DIRS] Checking directory accessibility..."
echo "[DIRS] Home directory: $(ls -la /home/a5k/kyleobrien.a5k/ | head -3)"
echo "[DIRS] Filtering directory exists: $(test -d /home/a5k/kyleobrien.a5k/self-fulfilling-model-organisms && echo YES || echo NO)"
echo "[DIRS] UV venv directory exists: $(test -d /home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv && echo YES || echo NO)"
echo "[DIRS] Checkpoints directory exists: $(test -d /projects/a5k/public/checkpoints && echo YES || echo NO)"

# Set TMPDIR early - some cluster nodes have missing /local/user directories
export TMPDIR=/projects/a5k/public/tmp
mkdir -p $TMPDIR
echo "[TMPDIR] Set TMPDIR to: $TMPDIR"

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

# Activate UV virtual environment
echo "[VENV] Starting venv activation at $(date)"
echo "[VENV] Activating environment: source /home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv/bin/activate"
source /home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv/bin/activate
echo "[VENV] Venv activation completed at $(date)"
echo "[VENV] Python path: $(which python)"
echo "[VENV] Python version: $(python --version 2>&1)"

# Load required modules
echo "[MODULES] Starting module loading at $(date)"
echo "[MODULES] Purging modules..."
module purge
echo "[MODULES] Loading PrgEnv-cray..."
module load PrgEnv-cray
echo "[MODULES] Loading cuda/12.6..."
module load cuda/12.6
echo "[MODULES] Loading brics/nccl/2.26.6-1..."
module load brics/nccl/2.26.6-1
echo "[MODULES] Module loading completed at $(date)"
echo "[MODULES] Loaded modules: $(module list 2>&1)"

# Prefer the module NCCL over any wheel-bundled version (required for Slingshot/OFI)
if [[ -n "${NCCL_ROOT:-}" && -f "${NCCL_ROOT}/lib/libnccl.so" ]]; then
  export LD_PRELOAD="${NCCL_ROOT}/lib/libnccl.so:${LD_PRELOAD-}"
fi

# Set up Python path
echo "[PYTHON] Setting up Python path at $(date)"
export PYTHONPATH=/projects/a5k/public/self-fulfilling-model-organisms:$PYTHONPATH
echo "[PYTHON] PYTHONPATH: $PYTHONPATH"
echo "[PYTHON] PATH: $PATH"

echo "[ENV] Setting environment variables at $(date)"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# Set TMPDIR to a valid location (the default /local/user may not exist on all nodes)
export TMPDIR=/projects/a5k/public/tmp
mkdir -p $TMPDIR

# Change to working directory
echo "[WORKDIR] Changing to working directory at $(date)"
echo "[WORKDIR] Current directory before cd: $(pwd)"
cd /home/a5k/kyleobrien.a5k/geodesic-gpt-neox/huggingface
echo "[WORKDIR] Current directory after cd: $(pwd)"
echo "[WORKDIR] Directory contents: $(ls -la | head -5)"

# Run the conversion and upload script
echo "[PYTHON] Starting Python script execution at $(date)"
echo "Starting conversion at $(date)"

# Use environment variables with defaults for paths
CHECKPOINTS_BASE_DIR_VAR="${CHECKPOINTS_BASE_DIR:-/projects/a5k/public/checkpoints/}"
TASK_INCLUDE_PATH_VAR="${TASK_INCLUDE_PATH:-/home/a5k/kyleobrien.a5k/self-fulfilling-model-organisms/lm_eval_tasks/}"

# Unescape EVAL_TASKS if it was escaped for SLURM export (commas were replaced with %2C)
if [ ! -z "$EVAL_TASKS" ]; then
    EVAL_TASKS="${EVAL_TASKS//%2C/,}"
fi

# Build command with optional retry parameters from environment
PIPELINE_CMD="python upload_and_evaluate_checkpoint.py \
    --experiment_name \"$EXPERIMENT_NAME\" \
    --neox_checkpoint \"$CHECKPOINT_NUMBER\" \
    --hf_org \"$HF_ORG\" \
    --neox_checkpoints_path \"$CHECKPOINTS_BASE_DIR_VAR\" \
    --local_hf_checkpoints_path /projects/a5k/public/checkpoints/hf_checkpoints \
    --conversion_script_path /home/a5k/kyleobrien.a5k/gpt-neox/tools/ckpts/convert_neox_to_hf.py \
    --task_include_path \"$TASK_INCLUDE_PATH_VAR\" \
    --skip-if-exists \
    --force-eval-if-exists"

# Add NeoX checkpoint upload flags unless SKIP_NEOX_UPLOAD is set
if [ -z "$SKIP_NEOX_UPLOAD" ]; then
    PIPELINE_CMD="$PIPELINE_CMD --upload-neox-checkpoint --skip-neox-if-exists"
    echo "NeoX checkpoint upload: ENABLED"
else
    echo "NeoX checkpoint upload: DISABLED (SKIP_NEOX_UPLOAD is set)"
fi

# Add NeoX-only mode flag if set
if [ ! -z "$UPLOAD_NEOX_ONLY" ]; then
    PIPELINE_CMD="$PIPELINE_CMD --upload-neox-only"
    echo "Using NeoX-only mode: upload NeoX checkpoint only, skip conversion/eval"
fi

# Add flag to upload all NeoX checkpoints if set
if [ ! -z "$UPLOAD_ALL_NEOX_CHECKPOINTS" ]; then
    PIPELINE_CMD="$PIPELINE_CMD --upload-all-neox-checkpoints"
    echo "Using upload-all-neox-checkpoints: uploading ALL NeoX checkpoints"
fi

# Add retry parameters if they were passed via environment
if [ ! -z "$UPLOAD_DELAY" ]; then
    PIPELINE_CMD="$PIPELINE_CMD --upload-delay $UPLOAD_DELAY"
    echo "Using upload delay: ${UPLOAD_DELAY}s"
fi

if [ ! -z "$MAX_RETRIES" ]; then
    PIPELINE_CMD="$PIPELINE_CMD --max-retries $MAX_RETRIES"
    echo "Using max retries: $MAX_RETRIES"
fi

if [ ! -z "$RETRY_MIN_WAIT" ]; then
    PIPELINE_CMD="$PIPELINE_CMD --retry-min-wait $RETRY_MIN_WAIT"
    echo "Using retry min wait: ${RETRY_MIN_WAIT}s"
fi

if [ ! -z "$RETRY_MAX_WAIT" ]; then
    PIPELINE_CMD="$PIPELINE_CMD --retry-max-wait $RETRY_MAX_WAIT"
    echo "Using retry max wait: ${RETRY_MAX_WAIT}s"
fi

# Add evaluation task parameters if they were passed via environment
if [ ! -z "$EVAL_TASKS" ]; then
    PIPELINE_CMD="$PIPELINE_CMD --eval_tasks \"$EVAL_TASKS\""
    echo "Using eval tasks: $EVAL_TASKS"
fi

if [ ! -z "$WANDB_ENTITY" ]; then
    PIPELINE_CMD="$PIPELINE_CMD --wandb_entity \"$WANDB_ENTITY\""
    echo "Using W&B entity: $WANDB_ENTITY"
fi

if [ ! -z "$WANDB_PROJECT" ]; then
    PIPELINE_CMD="$PIPELINE_CMD --wandb_project \"$WANDB_PROJECT\""
    echo "Using W&B project: $WANDB_PROJECT"
fi

# Debug environment before execution
echo "[ENV_DEBUG] Environment variables before Python execution:"
echo "[ENV_DEBUG] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-NOT_SET}"
echo "[ENV_DEBUG] NCCL environment variables:"
env | grep NCCL || echo "[ENV_DEBUG] No NCCL variables found"
echo "[ENV_DEBUG] HF environment variables:"
env | grep HF || echo "[ENV_DEBUG] No HF variables found"
echo "[ENV_DEBUG] TOKENIZERS_PARALLELISM: ${TOKENIZERS_PARALLELISM:-NOT_SET}"

# Test Python execution directly first
echo "[ENV_DEBUG] Testing basic Python execution at $(date)"
timeout 30 python -c "print('Python basic execution successful')" || echo "[ENV_DEBUG] Python basic test failed or timed out"

echo "[ENV_DEBUG] Testing Python with imports at $(date)"
timeout 60 python -c "
import time
print(f'Python with time import successful at {time.strftime(\"%Y-%m-%d %H:%M:%S\")}')
import torch
print(f'PyTorch import successful, CUDA available: {torch.cuda.is_available()}')
" || echo "[ENV_DEBUG] Python imports test failed or timed out"

# Execute the command without timeout to avoid interference with argparse
echo "[PYTHON] Executing command: $PIPELINE_CMD"
echo "[PYTHON] Command execution started at $(date)"
bash -c "$PIPELINE_CMD"
EXIT_CODE=$?
echo "[PYTHON] Command execution completed at $(date)"

echo "Pipeline completed at $(date) with exit code $EXIT_CODE"

# Clean up HF cache to save space
# if [ $EXIT_CODE -eq 0 ]; then
#     echo "Cleaning up HF cache..."
#     rm -rf /projects/a5k/public/hf/hub/models--${HF_ORG}--*
# fi

exit $EXIT_CODE