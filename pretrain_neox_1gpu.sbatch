#!/bin/bash
#SBATCH --job-name=neox-training
#SBATCH --nodes=64
#SBATCH --exclusive
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/projects/a5k/public/logs/neox-training/neox-training-%j.out

# --- Isambard node-limit guard (do not remove) ---
if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found. Install: ~/isambard_sbatch/install.sh" >&2
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

echo "===== Cluster Status at Job Start ====="
cluster_status.sh
echo "========================================"

# Activate uv virtual environment
source /home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv/bin/activate

module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

export NCCL_LIBRARY=/home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$NCCL_LIBRARY"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"

# Only use GPU 0 on each node
export CUDA_VISIBLE_DEVICES=0

# NCCL settings
export NCCL_COLLNET_ENABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET="AWS Libfabric"
export FI_PROVIDER=cxi
export NCCL_SOCKET_IFNAME=hsn
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1
export NCCL_PROTO=^LL128
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_RX_MATCH_MODE=soft
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1
export NCCL_MIN_NCHANNELS=4

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export HF_HUB_OFFLINE=1

echo "===== PyTorch & CUDA info ====="
python - <<'PY'
import os, torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"Visible GPUs: {n}")
    for i in range(n):
        print(f"  GPU[{i}]: {torch.cuda.get_device_name(i)}")
PY
echo "================================"

# Generate hostfile with 1 slot per node
if [ "$SLURM_NNODES" -gt 1 ]; then
    DLTS_HOSTFILE="/tmp/deepspeed_hostfile_${SLURM_JOB_ID}"
    scontrol show hostname "$SLURM_NODELIST" | while read -r node; do
        echo "$node slots=1"
    done > "$DLTS_HOSTFILE"
    echo "Generated 1-GPU hostfile: $(wc -l < "$DLTS_HOSTFILE") nodes"
    export DLTS_HOSTFILE="$DLTS_HOSTFILE"
else
    unset DLTS_HOSTFILE
fi

NEOX_CONFIG=$1
MAX_JOB_CHAINS=${MAX_JOB_CHAINS:-30}
CURRENT_CHAIN=${SLURM_JOB_CHAIN_COUNT:-0}
echo "Chain: $CURRENT_CHAIN / $MAX_JOB_CHAINS | Job: $SLURM_JOB_ID | Config: $NEOX_CONFIG"

export TMPDIR=/projects/a5k/public/tmp
mkdir -p "$TMPDIR"
export WANDB_DIR=/projects/a5k/public/logs/wandb
mkdir -p "$WANDB_DIR"

cd /home/a5k/kyleobrien.a5k/geodesic-gpt-neox
python deepy.py train.py "$NEOX_CONFIG"

echo "===== Job Completed ====="
