#!/bin/bash
#SBATCH --job-name=neox-training
#SBATCH --nodes=64
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
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

# Log cluster status at job start
echo "===== Cluster Status at Job Start ====="
cluster_status.sh
echo "========================================"

echo "Debug: Current node: $(hostname)"
echo "Debug: Available filesystems:"
df -h
echo "Debug: Home directory check:"
ls -la /home/
ls -la /home/a5k/ 2>/dev/null || echo "/home/a5k not accessible"
echo "Debug: Current user: $(whoami)"
echo "Debug: PWD: $(pwd)"

# Activate uv virtual environment
source /home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv/bin/activate

module purge
module load PrgEnv-cray
module load cuda/12.6
# Load OFI plugin for NCCL (provides Slingshot support)
module load brics/aws-ofi-nccl/1.8.1

# Use the bundled NCCL from venv (2.27.5) - required for PyTorch 2.10.0 compatibility
# The system NCCL (2.26.6) is missing ncclCommShrink function required by PyTorch
export NCCL_LIBRARY=/home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$NCCL_LIBRARY"

# Compilers and CUDA arch
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"

# NCCL / OFI (AWS Libfabric) settings for Slingshot (CXI)
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET,COLL,GRAPH
export NCCL_COLLNET_ENABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET="AWS Libfabric"   # must match plugin name
export FI_PROVIDER=cxi            # use the Slingshot CXI provider
export NCCL_SOCKET_IFNAME=hsn     # keep TCP fallback on HSN NICs
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1

# Performance tuning for Slingshot at scale
# Disable LL128 protocol (performs worse on Slingshot, enabled by default in NCCL >= 2.27)
export NCCL_PROTO=^LL128
# CXI-specific tuning for large-scale allreduce
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_RX_MATCH_MODE=soft
# Avoid Slingshot hangs/timeouts at large scale (NCCL issue #1272)
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0
# Disable non-inject message IDC (recommended by Isambard docs)
export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1
# Increase minimum NCCL channels
export NCCL_MIN_NCHANNELS=4

# Uncomment when debugging NCCL issues
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=NET

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Use offline mode to avoid HuggingFace API rate limits with many parallel ranks
export HF_HUB_OFFLINE=1

# --- Log PyTorch / CUDA info to the job output ---
echo "===== PyTorch & CUDA info ====="
python - <<'PY'
import os, torch
print(f"PyTorch: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"TORCH_CUDA_ARCH_LIST: {os.getenv('TORCH_CUDA_ARCH_LIST')}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"Visible GPUs: {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"  GPU[{i}]: {name}  (SM {cap[0]}.{cap[1]})")
PY
echo "================================"
# ----------------------------------

# Generate hostfile (only for multi-node; single-node uses local launcher)
if [ "$SLURM_NNODES" -gt 1 ]; then
    cd /home/a5k/kyleobrien.a5k/filtering_for_danger
    source ./neox/configs/isambard/slurm/generate_hostfile.sh
    echo "Generated hostfile at: $DLTS_HOSTFILE"
    [[ -f "$DLTS_HOSTFILE" ]] && cat "$DLTS_HOSTFILE"
    export DLTS_HOSTFILE="$DLTS_HOSTFILE"
else
    echo "Single-node job: skipping hostfile generation (using local launcher)"
    unset DLTS_HOSTFILE
fi

NEOX_CONFIG=$1
CONFIG_BASENAME=$(basename "$NEOX_CONFIG" .yml)

# Job chaining configuration
MAX_JOB_CHAINS=${MAX_JOB_CHAINS:-30}  # Maximum number of 24-hour jobs to chain (default: 30 = 30 days)
CURRENT_CHAIN=${SLURM_JOB_CHAIN_COUNT:-0}

echo "===== Job Chain Info ====="
echo "Current chain iteration: $CURRENT_CHAIN / $MAX_JOB_CHAINS"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $NEOX_CONFIG"
echo "=========================="

export TMPDIR=/projects/a5k/public/tmp
mkdir -p "$TMPDIR"

# Store W&B logs on shared filesystem so they persist and are accessible
export WANDB_DIR=/projects/a5k/public/logs/wandb
mkdir -p "$WANDB_DIR"

# Launch GPT-NeoX training
cd /home/a5k/kyleobrien.a5k/geodesic-gpt-neox
python deepy.py train.py "$NEOX_CONFIG"

echo "===== Job Completed ====="