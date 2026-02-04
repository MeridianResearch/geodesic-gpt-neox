#!/bin/bash
#SBATCH --job-name=code-tunnel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/projects/a5k/public/logs/code_tunnel/code_tunnel_%j.out

module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/nccl/2.26.6-1

# Prefer the module NCCL over any wheel-bundled version (required for Slingshot/OFI)
if [[ -n "${NCCL_ROOT:-}" && -f "${NCCL_ROOT}/lib/libnccl.so" ]]; then
  export LD_PRELOAD="${NCCL_ROOT}/lib/libnccl.so:${LD_PRELOAD-}"
fi

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
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET="AWS Libfabric"   # must match plugin name
export FI_PROVIDER=cxi            # use the Slingshot CXI provider
export NCCL_SOCKET_IFNAME=hsn     # keep TCP fallback on HSN NICs
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Launch GPT-NeoX training
cd /home/a5k/${USER}

# Start named VS Code tunnel for remote connection to compute node
~/opt/vscode_cli/code tunnel --name "${USER}-code-tunnel"