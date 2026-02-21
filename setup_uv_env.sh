#!/bin/bash
# setup_uv_env.sh - Create uv environment for GPT-NeoX on Isambard ARM HPC
# Based on original conda setup: agent_documentation/old_env_setup.sh
#
# Usage:
#   bash setup_uv_env.sh
#
# Prerequisites:
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - Run on a node with GPU access for best results
#
# This script will:
#   1. Load required modules (CUDA, NCCL)
#   2. Create a Python 3.12 virtual environment with uv
#   3. Install all GPT-NeoX dependencies (including PyTorch with CUDA)
#   4. Install flash-attn and transformer-engine
#   5. Build fused kernels
#   6. Run verification tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  GPT-NeoX UV Environment Setup for Isambard"
echo "=============================================="
echo "Architecture: $(uname -m)"
echo "Working directory: $SCRIPT_DIR"
echo ""

# ============================================
# Step 1: Load required modules
# ============================================
echo "=== Step 1: Loading modules ==="
# Note: Do NOT load brics/nccl - torch comes with its own NCCL (nvidia-nccl-cu12)
# and loading the system NCCL causes symbol conflicts
module load cuda/12.6 || echo "Warning: cuda/12.6 module not found"
module load cudatoolkit || echo "Warning: cudatoolkit module not found"

# ============================================
# Step 2: Set compiler and environment
# ============================================
echo ""
echo "=== Step 2: Setting compiler and environment ==="
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="9.0"
export TMPDIR=/projects/a5k/public/tmp_$USER
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6

echo "CC=$CC"
echo "CXX=$CXX"
echo "MAX_JOBS=$MAX_JOBS"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "TMPDIR=$TMPDIR"
echo "CUDA_HOME=$CUDA_HOME"

# ============================================
# Step 3: Check uv is available
# ============================================
echo ""
echo "=== Step 3: Checking uv ==="
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "uv version: $(uv --version)"

# ============================================
# Step 4: Create virtual environment
# ============================================
echo ""
echo "=== Step 4: Creating virtual environment ==="
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    rm -rf .venv
fi
uv venv --python 3.12 .venv
echo "Virtual environment created at: $SCRIPT_DIR/.venv"

# Get the venv site-packages path for setting library paths later
VENV_SITE_PACKAGES="$SCRIPT_DIR/.venv/lib/python3.12/site-packages"

# ============================================
# Step 5: Install all dependencies
# ============================================
echo ""
echo "=== Step 5: Installing all dependencies (including PyTorch with CUDA) ==="
echo "This may take a while for packages that need compilation..."
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MAX_JOBS=4 \
    uv sync --extra dev

# ============================================
# Step 6: Set NVIDIA library paths (NCCL, cuDNN, cuBLAS)
# ============================================
echo ""
echo "=== Step 6: Setting NVIDIA library paths ==="

# CRITICAL: Use LD_PRELOAD to force loading venv's NCCL instead of system NCCL
# This fixes NCCL version mismatch (system has 2.21.5, torch needs 2.27.5)
# The system NCCL is hardcoded via rpath in torch, so LD_LIBRARY_PATH alone won't work
export NCCL_LIBRARY="$VENV_SITE_PACKAGES/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"

# Also set LD_LIBRARY_PATH for other NVIDIA libraries
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/nccl/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cusparse/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cufft/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/curand/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cusolver/lib:$LD_LIBRARY_PATH"

# Include paths for compilation
export CPLUS_INCLUDE_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn/include:$CPLUS_INCLUDE_PATH"
export C_INCLUDE_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn/include:$C_INCLUDE_PATH"
export LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn/lib:$LIBRARY_PATH"
export CUDNN_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn"

export CPLUS_INCLUDE_PATH="$VENV_SITE_PACKAGES/nvidia/cublas/include:$CPLUS_INCLUDE_PATH"
export C_INCLUDE_PATH="$VENV_SITE_PACKAGES/nvidia/cublas/include:$C_INCLUDE_PATH"
export LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cublas/lib:$LIBRARY_PATH"

echo "NCCL library (LD_PRELOAD): $NCCL_LIBRARY"
echo "cuDNN path: $CUDNN_PATH"

# Create cuDNN header symlinks in PyTorch include directory
# This is required for building transformer-engine from source - the build system
# looks for cudnn.h in PyTorch's include path but it's not there by default
TORCH_INCLUDE="$VENV_SITE_PACKAGES/torch/include"
CUDNN_INCLUDE="$VENV_SITE_PACKAGES/nvidia/cudnn/include"
if [ -d "$CUDNN_INCLUDE" ] && [ -d "$TORCH_INCLUDE" ]; then
    echo "Creating cuDNN header symlinks in PyTorch include directory..."
    for f in "$CUDNN_INCLUDE"/*.h; do
        ln -sf "$f" "$TORCH_INCLUDE/$(basename $f)" 2>/dev/null || true
    done
    echo "cuDNN symlinks created"
else
    echo "Warning: Could not create cuDNN symlinks (directories not found)"
fi

# Verify PyTorch CUDA (with correct library paths)
echo ""
echo "Verifying PyTorch installation..."
LD_PRELOAD="$NCCL_LIBRARY" uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# ============================================
# Step 7: Install transformer-engine
# ============================================
echo ""
echo "=== Step 7: Installing transformer-engine ==="
# IMPORTANT: Must build from source using --no-binary to ensure ABI compatibility with PyTorch
# The prebuilt wheels from PyPI are compiled against a different PyTorch version and have
# symbol mismatches (e.g., c10_cuda_check_implementation has different signature)
# Use --python with absolute path to prevent uv from picking up system Python
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
echo "Building transformer-engine from source (required for PyTorch ABI compatibility)..."
echo "This may take 10-15 minutes..."
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MAX_JOBS=4 \
    CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH" \
    C_INCLUDE_PATH="$C_INCLUDE_PATH" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    LIBRARY_PATH="$LIBRARY_PATH" \
    CUDNN_PATH="$CUDNN_PATH" \
    CUDA_HOME="$CUDA_HOME" \
    LD_PRELOAD="$NCCL_LIBRARY" \
    uv pip install --python "$VENV_PYTHON" --no-build-isolation --no-cache-dir --no-binary transformer-engine-torch "transformer-engine[pytorch]==1.12" || {
        echo "ERROR: transformer-engine installation failed"
        echo "You may need to install it manually on a compute node with GPU access"
        exit 1
    }

# Validate transformer-engine immediately after installation
echo ""
echo "Validating transformer-engine installation..."
LD_PRELOAD="$NCCL_LIBRARY" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" uv run python -c "
import transformer_engine
print(f'  transformer_engine version: {transformer_engine.__version__}')
import transformer_engine.pytorch as te_pytorch
print('  transformer_engine.pytorch: OK')
from transformer_engine.pytorch import Linear, LayerNorm
print('  Core TE modules (Linear, LayerNorm): OK')
" || {
    echo "ERROR: transformer-engine validation failed!"
    echo "transformer_engine.pytorch must import successfully for training to work."
    echo "Check that PyTorch and TE versions are compatible."
    exit 1
}
echo "transformer-engine validation: PASSED"

# ============================================
# Step 8: Install flash-attn
# ============================================
echo ""
echo "=== Step 8: Installing flash-attn ==="
# IMPORTANT: Must build from source using --no-binary to ensure ABI compatibility with PyTorch
# The prebuilt wheels have symbol mismatches with PyTorch 2.6.x
echo "Building flash-attn from source (required for PyTorch ABI compatibility)..."
echo "This takes 30-60 minutes due to many CUDA kernels..."
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MAX_JOBS=4 \
    CUDA_HOME="$CUDA_HOME" \
    LD_PRELOAD="$NCCL_LIBRARY" \
    uv pip install --python "$VENV_PYTHON" --no-build-isolation --no-cache-dir --no-binary flash-attn flash-attn==2.6.3 || {
        echo "ERROR: flash-attn installation failed"
        echo "You may need to install it manually on a compute node with GPU access"
        exit 1
    }

# Validate flash-attn
echo ""
echo "Validating flash-attn installation..."
LD_PRELOAD="$NCCL_LIBRARY" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" uv run python -c "
import flash_attn
print(f'  flash_attn version: {flash_attn.__version__}')
from flash_attn import flash_attn_func
print('  flash_attn_func: OK')
" || {
    echo "ERROR: flash-attn validation failed!"
    exit 1
}
echo "flash-attn validation: PASSED"

# ============================================
# Step 9: Install sm_90a monkeypatch (sitecustomize.py)
# ============================================
echo ""
echo "=== Step 9: Installing GH200 sm_90a fix ==="
# GH200 GPUs report architecture as sm_90a, but PyTorch's _get_cuda_arch_flags()
# cannot parse the 'a' suffix (ValueError: invalid literal for int() with base 10: '90a').
# This sitecustomize.py runs at Python startup and monkeypatches the function.
cat > "$VENV_SITE_PACKAGES/sitecustomize.py" << 'SITECUSTOMIZE_EOF'
"""
GH200 sm_90a fix - Monkeypatch PyTorch's CUDA arch flag detection.

GH200 GPUs report sm_90a architecture, but PyTorch's _get_cuda_arch_flags()
in cpp_extension.py cannot parse the 'a' suffix. This causes:
  ValueError: invalid literal for int() with base 10: '90a'

This sitecustomize.py is loaded automatically at Python startup and:
1. Sets TORCH_CUDA_ARCH_LIST=9.0 as a fallback
2. Monkeypatches _get_cuda_arch_flags() to return correct flags for sm_90
"""
import os
import sys

# Set architecture env var unconditionally - srun --export may strip it
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

def _patch_pytorch_cuda_arch():
    """Replace PyTorch's _get_cuda_arch_flags with a version that handles sm_90a."""
    try:
        import torch.utils.cpp_extension as cpp_ext
        original_get_cuda_arch_flags = cpp_ext._get_cuda_arch_flags

        def _patched_get_cuda_arch_flags(cflags=None):
            return ['-gencode', 'arch=compute_90,code=sm_90']

        cpp_ext._get_cuda_arch_flags = _patched_get_cuda_arch_flags
    except (ImportError, AttributeError):
        pass

_patch_pytorch_cuda_arch()
SITECUSTOMIZE_EOF
echo "sitecustomize.py installed at: $VENV_SITE_PACKAGES/sitecustomize.py"

# ============================================
# Step 10: Apply wandb patch (fix isatty issue)
# ============================================
echo ""
echo "=== Step 10: Applying wandb patch ==="
WANDB_TERM_FILE="$VENV_SITE_PACKAGES/wandb/errors/term.py"
if [ -f "$WANDB_TERM_FILE" ]; then
    sed -i 's/    return sys\.stderr\.isatty()/    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()/' "$WANDB_TERM_FILE" || true
    echo "wandb patch applied"
else
    echo "Warning: wandb term.py not found, skipping patch"
fi

# ============================================
# Step 11: Build fused kernels
# ============================================
echo ""
echo "=== Step 11: Building fused kernels ==="
LD_PRELOAD="$NCCL_LIBRARY" uv run python -c "from megatron.fused_kernels import load; load()" || {
    echo "Warning: fused kernels build failed"
    echo "This may work later when running on a node with GPU access"
}

# ============================================
# Step 12: Verify installation
# ============================================
echo ""
echo "=== Step 12: Verifying installation ==="
LD_PRELOAD="$NCCL_LIBRARY" uv run python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')

try:
    import wandb
    print(f'wandb: {wandb.__version__}')
except ImportError:
    print('wandb: NOT INSTALLED')

try:
    import flash_attn
    print(f'flash_attn: {flash_attn.__version__}')
except ImportError:
    print('flash_attn: NOT INSTALLED')

try:
    import transformer_engine
    print(f'transformer_engine: {transformer_engine.__version__}')
except ImportError:
    print('transformer_engine: NOT INSTALLED')

try:
    import datasets
    print(f'datasets: {datasets.__version__}')
except ImportError:
    print('datasets: NOT INSTALLED')

try:
    import transformers
    print(f'transformers: {transformers.__version__}')
except ImportError:
    print('transformers: NOT INSTALLED')

try:
    import deepspeed
    print(f'deepspeed: {deepspeed.__version__}')
except ImportError:
    print('deepspeed: NOT INSTALLED')

try:
    import lm_dataformat
    print(f'lm_dataformat: installed')
except ImportError:
    print('lm_dataformat: NOT INSTALLED')
"

# ============================================
# Step 13: Run tests
# ============================================
echo ""
echo "=== Step 13: Running tests ==="
echo "Running UV install verification tests..."
LD_PRELOAD="$NCCL_LIBRARY" uv run pytest tests/test_uv_install.py -v || {
    echo "Warning: Some UV install tests failed"
    echo "This may be expected if running without GPU access"
}

echo ""
echo "Running HuggingFace dataset preparation tests..."
LD_PRELOAD="$NCCL_LIBRARY" uv run pytest tests/test_prepare_hf_dataset.py -v || {
    echo "Warning: Some tests failed"
    echo "This may be expected if running without GPU access"
}

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Or run commands with:"
echo "  LD_PRELOAD=\$NCCL_LIBRARY uv run <command>"
echo ""
echo "IMPORTANT: You must set LD_PRELOAD to use the correct NCCL library:"
echo "  export NCCL_LIBRARY=$VENV_SITE_PACKAGES/nvidia/nccl/lib/libnccl.so.2"
echo "  export LD_PRELOAD=\$NCCL_LIBRARY"
echo ""
echo "Before running training, ensure these environment variables are set."
echo ""
echo "Example training command:"
echo "  sbatch pretrain_neox.sbatch /path/to/config.yml"
