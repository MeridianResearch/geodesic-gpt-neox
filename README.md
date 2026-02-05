# GPT-NeoX Geodesic Fork for Isambard

A fork of [EleutherAI's GPT-NeoX](https://github.com/EleutherAI/gpt-neox) optimized for the Isambard supercomputer with NVIDIA GH200 (Grace Hopper) GPUs. This repository supports training large language models from 100M to 100B+ parameters using DeepSpeed, Megatron-LM tensor/pipeline parallelism, and ZeRO optimization.

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Running Pretraining Jobs](#running-pretraining-jobs)
- [VS Code Server (Remote Development)](#vs-code-server-remote-development)
- [Data Pipeline](#data-pipeline)
- [Checkpoint Conversion](#checkpoint-conversion)
- [Key Paths](#key-paths)

## Quick Start

```bash
# 1. Set up the environment (one-time)
bash setup_uv_env.sh

# 2. Set the NCCL library path (required for every session)
export NCCL_LIBRARY=.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2

# 3. Verify CUDA is working
LD_PRELOAD=$NCCL_LIBRARY uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 4. Submit a training job
sbatch pretrain_neox.sbatch /absolute/path/to/config.yml
```

## Environment Setup

### Prerequisites

- **Architecture**: ARM (aarch64) - Isambard uses NVIDIA Grace Hopper (GH200) GPUs
- **CUDA**: 12.6
- **Python**: 3.12
- **UV**: Package manager (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Installation

Run the setup script on a compute node (requires GPU for building flash-attn and fused kernels):

```bash
sbatch run_on_compute.sbatch bash setup_uv_env.sh

# Monitor progress
tail -f /projects/a5k/public/logs/neox-training/run_on_compute_<JOB_ID>.out
```

This script will:
1. Load required CUDA modules
2. Create a Python 3.12 virtual environment with UV
3. Install all dependencies including PyTorch with CUDA support
4. Install flash-attn 2.6.3 and transformer-engine 1.12.0
5. Build fused CUDA kernels
6. Run verification tests

### Using the Environment

**Important**: Always set `LD_PRELOAD` to avoid NCCL version conflicts:

```bash
# Set the NCCL library path
export NCCL_LIBRARY=.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2

# Run commands with UV
LD_PRELOAD=$NCCL_LIBRARY uv run python <script.py>
LD_PRELOAD=$NCCL_LIBRARY uv run pytest tests/ -v

# Or activate the venv directly
source .venv/bin/activate
LD_PRELOAD=$NCCL_LIBRARY python <script.py>
```

### Verify Installation

```bash
# Check CUDA availability
LD_PRELOAD=$NCCL_LIBRARY uv run python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device count: {torch.cuda.device_count()}')
print(f'Device name: {torch.cuda.get_device_name(0)}')
"

# Run full verification tests (on a compute node via SLURM)
sbatch run_on_compute.sbatch uv run pytest tests/test_uv_install.py -v

# Or locally if already on a compute node with GPU
LD_PRELOAD=$NCCL_LIBRARY uv run pytest tests/test_uv_install.py -v
```

### Installed Packages

| Package | Version |
|---------|---------|
| PyTorch | 2.10.0+cu126 |
| flash-attn | 2.6.3 |
| transformer-engine | 1.12.0 |
| deepspeed | 0.16.5 |
| wandb | 0.24.0 |
| datasets | 4.5.0 |
| transformers | 4.57.x |

## Running Pretraining Jobs

### Node Count Guidelines

- **64 nodes**: Pretraining and midtraining runs
- **16 nodes**: SFT, DPO, and other post-training jobs

### Submit a Training Job

```bash
# Standard pretraining (64 nodes)
sbatch pretrain_neox.sbatch /absolute/path/to/config.yml

# SFT/DPO jobs (16 nodes)
sbatch --nodes=16 pretrain_neox.sbatch /absolute/path/to/config.yml
```

**Important**: Config paths must be absolute when submitting via SLURM.

### Monitor Training

```bash
# Check job status
squeue -u $USER | grep neox-training

# View training logs
tail -f /projects/a5k/public/logs/neox-training/neox-training-<JOB_ID>.out

# Check training progress (iteration and loss)
grep "iteration.*lm_loss" /projects/a5k/public/logs/neox-training/neox-training-<JOB_ID>.out
```

### Cancel a Job

```bash
scancel <job_id>
```

### Local Training (Single Node)

For testing or small-scale training:

```bash
LD_PRELOAD=$NCCL_LIBRARY uv run python deepy.py train.py configs/model.yml
```

## VS Code Server (Remote Development)

VS Code can connect directly to Isambard compute nodes via tunnels, providing a full IDE with GPU access for interactive development and debugging. This follows the [Isambard VS Code guide](https://docs.isambard.ac.uk/user-documentation/guides/vscode/).

### One-Time Setup: Install the VS Code CLI

```bash
curl --location --output vscode_cli.tar.gz \
  "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64"
mkdir -p ~/opt/vscode_cli
tar -C ~/opt/vscode_cli --extract --verbose --file vscode_cli.tar.gz
rm vscode_cli.tar.gz

# Verify
~/opt/vscode_cli/code --version
```

### Launch a Tunnel

```bash
# Submit the tunnel job (allocates 1 node with 4 GPUs for 24 hours)
sbatch vscode_tunnel.sh

# Watch for the GitHub device code and vscode.dev link
tail -f /projects/a5k/public/logs/code_tunnel/code_tunnel_<JOB_ID>.out
```

### Authenticate and Connect

1. Watch the job log for a GitHub device code
2. Visit https://github.com/login/device and enter the code
3. Open the `vscode.dev` URL from the log, **or** use the VS Code desktop client:
   - Install the **Remote - Tunnels** extension
   - Command Palette (Ctrl+Shift+P) > **Remote-Tunnel: Connect to Tunnel...**
   - Authenticate with GitHub and select your tunnel name

### End the Session

```bash
scancel <JOB_ID>
```

Tunnel logs are written to `/projects/a5k/public/logs/code_tunnel/`.

## Data Pipeline

### Unified HuggingFace Dataset Pipeline

The `prepare_hf_dataset.py` script handles the complete data preparation workflow:

```bash
LD_PRELOAD=$NCCL_LIBRARY uv run python prepare_hf_dataset.py \
    --dataset <hf_dataset_name> \
    --subset <subset_name> \
    --split train
```

**Example:**
```bash
LD_PRELOAD=$NCCL_LIBRARY uv run python prepare_hf_dataset.py \
    --dataset cais/wmdp-corpora \
    --subset bio-retain-corpus \
    --split train
```

This will:
1. Load the dataset from HuggingFace
2. Count tokens (optional, use `--skip-count` to skip)
3. Save to JSONL format
4. Run GPT-NeoX tokenization

**Output**: `/projects/a5k/public/data/<dataset_name>_<subset>_<split>/`

### Pipeline Options

| Option | Description |
|--------|-------------|
| `--dataset` | HuggingFace dataset name (required) |
| `--subset` | Dataset config/subset name |
| `--split` | Dataset split (default: train) |
| `--output-dir` | Custom output directory |
| `--text-column` | Override text column (auto-detects 'text' or 'messages') |
| `--skip-count` | Skip token counting |
| `--skip-tokenize` | Skip GPT-NeoX tokenization |
| `--count-only` | Only count tokens |
| `--num-proc` | Parallel processes (default: 16) |

### Manual Data Preprocessing

For more control, use the individual preprocessing scripts:

**For pretraining data (text column):**
```bash
LD_PRELOAD=$NCCL_LIBRARY uv run python tools/datasets/preprocess_data.py \
    --input /path/to/data.jsonl \
    --output-prefix /projects/a5k/public/data/<name>/<name> \
    --vocab /projects/a5k/public/data/neox_tokenizer/tokenizer.json \
    --dataset-impl mmap \
    --tokenizer-type HFTokenizer \
    --append-eod \
    --workers 50
```

**For chat/instruction data (messages column):**
```bash
LD_PRELOAD=$NCCL_LIBRARY uv run python tools/datasets/preprocess_data_with_chat_template.py \
    --input /path/to/messages.jsonl \
    --output-prefix /projects/a5k/public/data/<name>/<name> \
    --tokenizer-path geodesic-research/gpt-neox-instruct-tokenizer \
    --jsonl-keys messages \
    --dataset-impl mmap \
    --workers 50
```

### Output Files

- Pretraining data: `<prefix>_text_document.bin` and `.idx`
- Chat template data: `<prefix>_messages_document.bin` and `.idx`

### Tokenizers

| Type | Path |
|------|------|
| Base (pretraining) | `/projects/a5k/public/data/neox_tokenizer/tokenizer.json` |
| Instruct (SFT/DPO) | `/projects/a5k/public/data/neox_tokenizer_instruct/tokenizer.json` |
| Chat template | `/projects/a5k/public/data/neox_tokenizer_instruct/chat_template.jinja` |

## Checkpoint Conversion

### HuggingFace to NeoX

Convert HuggingFace models to GPT-NeoX format for continued training:

```bash
# Basic conversion
LD_PRELOAD=$NCCL_LIBRARY uv run python huggingface/convert_hf_gptneox_to_neox.py \
    --hf-model geodesic-research/sfm-pretraining_mix_blocklist_filtered

# With specific revision
LD_PRELOAD=$NCCL_LIBRARY uv run python huggingface/convert_hf_gptneox_to_neox.py \
    --hf-model geodesic-research/sfm-model-name \
    --revision global_step1000

# Submit via SLURM
sbatch huggingface/convert_hf_to_neox.sbatch <hf_model> [iteration]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--hf-model` | HuggingFace model name or path (required) |
| `--revision` | Model revision/branch (e.g., `global_step0`) |
| `--output-dir` | Custom output directory |
| `--tp` | Tensor parallelism size (default: 1) |
| `--iteration` | Iteration number for checkpoint (default: 0) |
| `--no-transformer-engine` | Use legacy NeoX format instead of TE format |

**Default output**: `/projects/a5k/public/checkpoints/sf_model_organisms/<model_name>`

### NeoX to HuggingFace

Convert NeoX checkpoints to HuggingFace format for inference or sharing:

```bash
LD_PRELOAD=$NCCL_LIBRARY uv run python tools/ckpts/convert_neox_to_hf.py \
    --input_dir /path/to/global_stepXXX \
    --config_file config.yml \
    --output_dir hf_model/ \
    --precision bf16
```

### Batch Checkpoint Pipeline

Process multiple checkpoints and upload to HuggingFace:

```bash
# Submit jobs for all checkpoints in an experiment
./huggingface/batch_submit_checkpoints.sh <experiment_name> [hf_org] [options]

# Examples
./huggingface/batch_submit_checkpoints.sh sft_dolci_instruct_unfiltered
./huggingface/batch_submit_checkpoints.sh sft_dolci_instruct_filtered geodesic-research --eval

# Monitor checkpoint pipeline jobs
squeue -u $USER | grep convert_
```

**Batch Options:**
| Option | Description |
|--------|-------------|
| `--upload-delay <seconds\|random>` | Initial delay before uploads |
| `--max-retries <number>` | Max retry attempts on 429 errors (default: 20) |
| `--upload-neox-only` | Upload only NeoX checkpoints, skip HF conversion |
| `--skip-neox-upload` | Skip NeoX upload, only convert to HF |
| `--eval` | Enable evaluation after upload |

## Key Paths

| Resource | Path |
|----------|------|
| Checkpoints | `/projects/a5k/public/checkpoints/sf_model_organisms/` |
| Training logs | `/projects/a5k/public/logs/neox-training/` |
| Tokenized datasets | `/projects/a5k/public/data/` |
| Temp directory | `/projects/a5k/public/tmp/` |

## Performance

### Expected FLOPS (GH200/H100 GPUs)

The cluster uses HPE Slingshot interconnect with AWS OFI NCCL plugin for high-performance multi-node training.

| Configuration | Nodes | GPUs | Micro Batch | Seq Length | FLOPS/GPU | MFU |
|---------------|-------|------|-------------|------------|-----------|-----|
| Short sequence | 4 | 16 | 4 | 2048 | ~290 TFLOPS | ~29% |
| Long sequence | 16 | 64 | 1 | 16384 | ~446 TFLOPS | ~45% |
| Single node | 1 | 4 | 4 | 2048 | ~388 TFLOPS | ~39% |

### Key Performance Insights

1. **Sequence length drives MFU**: Longer sequences provide more arithmetic intensity, better saturating tensor cores. 16384 seq length achieves ~45% MFU vs ~29% for 2048.

2. **Slingshot is required**: Multi-node training MUST use Slingshot/OFI. TCP sockets achieve only ~16 TFLOPS/GPU (1.6% MFU) - a **17x performance penalty**.

3. **Theoretical peak**: GH200/H100 BF16 dense (no sparsity) is ~990 TFLOPS.

### Interconnect Configuration

The `pretrain_neox.sbatch` script automatically configures Slingshot via:
- `brics/nccl/2.26.6-1` module (system NCCL with OFI plugin)
- `NCCL_NET="AWS Libfabric"` with `FI_PROVIDER=cxi`

**Warning**: Do not modify the NCCL settings in the sbatch script - incorrect configuration will fall back to slow TCP sockets.

## Configuration

Training is driven by YAML config files. Multiple configs can be merged:

```bash
LD_PRELOAD=$NCCL_LIBRARY uv run python deepy.py train.py config1.yml config2.yml
```

Key configuration categories:
- Model architecture (hidden size, layers, attention heads)
- Training hyperparameters (batch size, learning rate, optimizer)
- Parallelism settings (tensor_model_parallel_size, pipe_parallel_size)
- Data paths and tokenizer settings
- Logging (W&B, TensorBoard)

## Troubleshooting

### NCCL Symbol Errors

If you see `undefined symbol: ncclCommShrink`:

**For local/interactive use:**
```bash
# Use the venv's bundled NCCL library
export NCCL_LIBRARY=.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
LD_PRELOAD=$NCCL_LIBRARY uv run <command>
```

**For SLURM multi-node jobs:**
The sbatch script automatically loads `brics/nccl/2.26.6-1` and uses `LD_PRELOAD` to prefer the system NCCL. This is required for Slingshot/OFI support - do not modify this configuration.

### CUDA Not Available

1. Check you're on a GPU node: `nvidia-smi`
2. Verify CUDA module is loaded: `module load cuda/12.6`
3. Check environment: `echo $CUDA_HOME`

### flash-attn or transformer-engine Import Errors

These packages require compilation. If imports fail:

```bash
# Reinstall with proper environment
VENV_PYTHON=.venv/bin/python
CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6 \
    LD_PRELOAD=$NCCL_LIBRARY \
    uv pip install --python $VENV_PYTHON --no-build-isolation flash-attn==2.6.3
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [EleutherAI](https://www.eleuther.ai/) for the original GPT-NeoX framework
- [DeepSpeed](https://www.deepspeed.ai/) for distributed training optimization
- Isambard supercomputer team for HPC support
