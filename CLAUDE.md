# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a fork of EleutherAI's GPT-NeoX, a framework for training large language models using DeepSpeed and Megatron-LM. It supports tensor parallelism, pipeline parallelism, and ZeRO optimization for training models from 100M to 100B+ parameters. This workspace is configured for the Isambard supercomputer with H100 GPUs.

## Login Nodes vs Compute Nodes

**IMPORTANT:** Login nodes do NOT have GPUs. Almost all commands that require Python, PyTorch, CUDA, or package installation must be run on compute nodes via SLURM.

**Use `run_on_compute.sbatch` for:**
- Installing/rebuilding the UV environment (`sbatch run_on_compute.sbatch bash setup_uv_env.sh`)
- Running tests (`sbatch run_on_compute.sbatch uv run pytest tests/test_uv_install.py -v`)
- Running Python scripts that use PyTorch/CUDA
- Data preprocessing and tokenization
- Any `uv run` or `pip install` commands

**Safe to run on login nodes:**
- `squeue`, `sacct`, `scancel` (SLURM commands)
- `tail -f`, `grep`, `cat` (viewing logs and files)
- `git` operations
- Editing files
- Submitting jobs with `sbatch`

When in doubt, use `sbatch run_on_compute.sbatch <command>` to run on a compute node.

## Common Commands

### Training

**Node Count Guidelines:**
- **64 nodes**: Pretraining and midtraining runs
- **16 nodes**: All other runs (SFT, DPO, benign tampering, etc.)

```bash
# Single-node training
python deepy.py train.py configs/model.yml configs/local_setup.yml

# Multi-node training via SLURM (preferred for Isambard)
sbatch pretrain_neox.sbatch /absolute/path/to/config.yml

# Submit SFT/DPO/other jobs (16 nodes)
sbatch --nodes=16 pretrain_neox.sbatch /absolute/path/to/config.yml

# Monitor job status
squeue -u $USER | grep neox-training

# View training logs
tail -f /projects/a5k/public/logs/neox-training/neox-training-<JOB_ID>.out

# Check training progress (look for iteration and loss)
grep "iteration.*lm_loss" /projects/a5k/public/logs/neox-training/neox-training-<JOB_ID>.out
```

**Important**: Config paths must be absolute when submitting via SLURM.

### Evaluation

```bash
python deepy.py eval.py -d configs config.yml --eval_tasks lambada hellaswag piqa
```

### Text Generation

```bash
python deepy.py generate.py -d configs config.yml text_generation.yml
```

### Data Preprocessing

```bash
# Download and tokenize standard datasets
python prepare_data.py -d ./data -t GPT2BPETokenizer enwik8

# Custom data preprocessing
python tools/datasets/preprocess_data.py \
    --input ./data/dataset.jsonl \
    --output-prefix ./data/dataset \
    --tokenizer-type HFTokenizer \
    --append-eod

# Chat template preprocessing (for SFT/DPO)
python tools/datasets/preprocess_data_with_chat_template.py \
    --input data/sft/train.jsonl \
    --output-prefix data/sft/train \
    --tokenizer-path checkpoints/model/tokenizer \
    --jsonl-keys messages
```

### Preparing HuggingFace Datasets for GPT-NeoX Training

Use `prepare_hf_dataset.py` to count tokens, export to JSONL, and tokenize in one step:

```bash
# Standard usage (submit to compute node via SLURM)
sbatch --time=24:00:00 run_on_compute.sbatch python prepare_hf_dataset.py \
    --dataset allenai/dolma3_dolmino_mix-100B-1025

# With subset
sbatch run_on_compute.sbatch python prepare_hf_dataset.py \
    --dataset cais/wmdp-corpora \
    --subset bio-retain-corpus \
    --split train

# Count tokens only (no tokenization)
sbatch run_on_compute.sbatch python prepare_hf_dataset.py \
    --dataset allenai/some-dataset \
    --count-only

# Skip counting, just tokenize
sbatch run_on_compute.sbatch python prepare_hf_dataset.py \
    --dataset allenai/some-dataset \
    --skip-count
```

The script auto-detects `text` vs `messages` columns and chooses the right tokenizer pipeline. Output goes to `/projects/a5k/public/data/<dataset_name>_<subset>_<split>/`.

**Pipeline stages:**
1. Load dataset from HuggingFace
2. Auto-detect text column (`text` or `messages`)
3. Count tokens (batched, using HF tokenizer)
4. Export to JSONL
5. Run GPT-NeoX tokenization (`preprocess_data.py`)

**Key options:**
- `--count-only` — just count tokens, skip export and tokenization
- `--skip-count` / `--skip-tokenize` — skip individual stages
- `--vocab-file` — GPT-NeoX tokenizer (default: instruct tokenizer)
- `--num-proc` — parallel processes (default: 16)
- `--tokenize-workers` — workers for tokenization (default: same as num-proc)
- `--output-dir` — override auto-generated output path

**Output files:**
- `dataset.jsonl` — exported text
- `<dir>_text_document.bin/.idx` — tokenized data for GPT-NeoX
- `pipeline_results.json` — metadata with token counts, timing

**For GPT-NeoX training config:**
```yaml
"train_data_paths": ["/projects/a5k/public/data/<dir>/<dir>_text_document"]
```

#### Advanced: Manual Tokenization

For post-training data WITH chat template (has `messages` column):
```bash
sbatch --time=04:00:00 run_on_compute.sbatch python tools/datasets/preprocess_data_with_chat_template.py \
    --input /projects/a5k/public/data/<dataset_name>/messages.jsonl \
    --output-prefix /projects/a5k/public/data/<dataset_name>/<dataset_name> \
    --tokenizer-path geodesic-research/gpt-neox-instruct-tokenizer \
    --jsonl-keys messages \
    --dataset-impl mmap \
    --num-docs <num_examples> \
    --workers 50
```

For pretraining data WITHOUT chat template (has `text` column):
```bash
sbatch --time=04:00:00 run_on_compute.sbatch python tools/datasets/preprocess_data.py \
    --input /projects/a5k/public/data/<dataset_name>/data.jsonl \
    --output-prefix /projects/a5k/public/data/<dataset_name>/<dataset_name> \
    --vocab /projects/a5k/public/data/neox_tokenizer/tokenizer.json \
    --dataset-impl mmap \
    --tokenizer-type HFTokenizer \
    --append-eod \
    --num-docs <num_examples> \
    --workers 50
```

**Output Files:**
- `<output-prefix>_messages_document.bin/.idx` - For chat template data
- `<output-prefix>_text_document.bin/.idx` - For pretraining data

### Checkpoint Conversion

All checkpoint conversions require compute nodes:

```bash
# NeoX to HuggingFace
sbatch run_on_compute.sbatch python tools/ckpts/convert_neox_to_hf.py \
    --input_dir /path/to/global_stepXXX \
    --config_file config.yml \
    --output_dir hf_model/ \
    --precision bf16

# HuggingFace to NeoX (for continued training)
sbatch run_on_compute.sbatch python huggingface/convert_hf_gptneox_to_neox.py \
    --hf-model geodesic-research/sfm-pretraining_mix_blocklist_filtered

# With specific revision
sbatch run_on_compute.sbatch python huggingface/convert_hf_gptneox_to_neox.py \
    --hf-model geodesic-research/sfm-model-name \
    --revision global_step1000

# Or use the dedicated sbatch script for HF→NeoX conversion
sbatch huggingface/convert_hf_to_neox.sbatch <hf_model> [iteration]

# Inspect checkpoint structure
sbatch run_on_compute.sbatch python tools/ckpts/inspect_checkpoints.py --checkpoint-path path/to/ckpt
```

**HF→NeoX Conversion Options:**
- `--hf-model`: HuggingFace model name or path (required)
- `--revision`: Model revision/branch (e.g., `global_step0`)
- `--output-dir`: Output directory (default: `/projects/a5k/public/checkpoints/sf_model_organisms/<model_name>`)
- `--tp`: Tensor parallelism size (default: 1)
- `--iteration`: Iteration number for checkpoint (default: 0)
- `--no-transformer-engine`: Disable TE format (use legacy NeoX format)

### Testing

Tests require GPU access and must be run on compute nodes via SLURM:

```bash
# Run the UV install verification tests (recommended)
sbatch run_on_compute.sbatch uv run pytest tests/test_uv_install.py -v

# Run all tests (--forked is required)
sbatch run_on_compute.sbatch uv run pytest --forked tests -v

# Run specific test module
sbatch run_on_compute.sbatch uv run pytest --forked tests/model/test_model_generation.py -v

# Run CPU-only tests (can run on login node, but compute node preferred)
sbatch run_on_compute.sbatch uv run pytest tests -m cpu -v

# Monitor test output
tail -f /projects/a5k/public/logs/neox-training/run_on_compute_<JOB_ID>.out
```

### Running Commands on Compute Nodes

`run_on_compute.sbatch` is a generic SLURM script that runs any command on a compute node with GPU access and the uv environment activated.

```bash
# Usage: sbatch [slurm-options] run_on_compute.sbatch <command> [args...]

# Install/rebuild the UV environment from scratch
sbatch run_on_compute.sbatch bash setup_uv_env.sh

# Run the full test suite on a compute node
sbatch run_on_compute.sbatch uv run pytest tests/test_uv_install.py -v

# Quick GPU check
sbatch --time=00:05:00 run_on_compute.sbatch nvidia-smi

# Run a Python script
sbatch run_on_compute.sbatch uv run python tools/datasets/preprocess_data.py --help

# Override defaults (e.g. longer walltime, more GPUs)
sbatch --time=24:00:00 --gpus=4 run_on_compute.sbatch <command>
```

**Defaults:** 1 node, 1 GPU, 16 CPUs, 2hr walltime (override with sbatch flags).
**Logs:** `/projects/a5k/public/logs/neox-training/run_on_compute_<JOB_ID>.out`

The script automatically:
- Loads `cuda/12.6` and sets compilers (`gcc-12`/`g++-12`)
- Sets `TORCH_CUDA_ARCH_LIST=9.0` and `TMPDIR`
- Activates `.venv` if present (sets `NCCL_LIBRARY`/`LD_PRELOAD`)
- Skips venv activation if `.venv` doesn't exist (e.g. during initial install)

### Environment Setup (Isambard)

#### UV Environment (Recommended)

The project uses UV for dependency management. A single setup script handles everything from scratch.

**Fresh install (from scratch):**
```bash
# Submit to a compute node (requires GPU for building flash-attn, fused kernels, etc.)
# Takes ~60 minutes (flash-attn compilation is the bottleneck)
sbatch run_on_compute.sbatch bash setup_uv_env.sh

# Monitor progress
tail -f /projects/a5k/public/logs/neox-training/run_on_compute_<JOB_ID>.out
```

The setup script performs these steps:
1. Loads required modules (PrgEnv-cray, cuda/12.6)
2. Sets compiler and environment variables (gcc-12, TORCH_CUDA_ARCH_LIST=9.0)
3. Creates a Python 3.12 virtual environment via `uv venv`
4. Installs all dependencies from pyproject.toml (PyTorch, DeepSpeed, etc.)
5. Configures NVIDIA library paths and cuDNN header symlinks
6. Builds transformer-engine from source (~10 min)
7. Builds flash-attn from source (~50 min)
8. Installs GH200 sm_90a fix (sitecustomize.py)
9. Applies wandb isatty patch
10. Builds fused CUDA kernels (scaled_softmax, rotary embedding)
11. Verifies all packages import correctly with CUDA
12. Runs the full test suite (`tests/test_uv_install.py`)

**Using the existing UV environment:**

```bash
# Required: Set NCCL library path to avoid symbol conflicts with system NCCL
export NCCL_LIBRARY=.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2

# Run commands with uv
LD_PRELOAD=$NCCL_LIBRARY uv run python <script.py>
LD_PRELOAD=$NCCL_LIBRARY uv run pytest tests/ -v

# Or activate the venv directly
source .venv/bin/activate
LD_PRELOAD=$NCCL_LIBRARY python <script.py>
```

**Check CUDA availability:**
```bash
LD_PRELOAD=$NCCL_LIBRARY uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

**Run all verification tests (includes CUDA and training tests):**
```bash
# Via SLURM (preferred — allocates a GPU node)
sbatch run_on_compute.sbatch uv run pytest tests/test_uv_install.py -v

# Or locally if already on a compute node with GPU
LD_PRELOAD=$NCCL_LIBRARY uv run pytest tests/test_uv_install.py -v
```

**Key packages installed:**
- PyTorch 2.5.1+cu124
- flash-attn 2.6.3
- transformer-engine 1.12.0
- deepspeed 0.16.5 (EleutherAI/DeeperSpeed fork)
- wandb, datasets, transformers, accelerate

**Test suite coverage (`tests/test_uv_install.py`):**
- Core imports (torch, datasets, transformers, wandb, deepspeed, etc.)
- CUDA availability, device count, version, cuDNN
- CUDA operations (matmul, memory allocation, bf16, flash attention SDPA)
- Fused kernel loading and execution
- DeepSpeed CUDA accelerator
- Optional packages (flash-attn, transformer-engine)
- Environment setup (Python version, TORCH_CUDA_ARCH_LIST)
- Training loss verification (tiny 2-layer model trains for 100 iterations, verifies loss decreases >10%)

**Important notes:**
- For **local/interactive use**: Use `LD_PRELOAD=$NCCL_LIBRARY` with the bundled NCCL
- For **SLURM multi-node jobs**: The sbatch script loads `brics/nccl/2.26.6-1` and uses `LD_PRELOAD` to prefer the system NCCL (required for Slingshot/OFI support)
- The setup script handles flash-attn and transformer-engine installation with `--no-build-isolation`
- To rebuild from scratch: `rm -rf .venv && sbatch run_on_compute.sbatch bash setup_uv_env.sh`

**GH200 sm_90a Fix:**
The setup script installs a `sitecustomize.py` into `.venv/lib/python3.12/site-packages/` that fixes a PyTorch JIT compilation issue on GH200 GPUs. GH200 reports as `sm_90a` but PyTorch's cpp_extension module incorrectly parses "90a" as an integer, causing:
```
ValueError: invalid literal for int() with base 10: '90a'
```
The fix monkeypatches `torch.utils.cpp_extension._get_cuda_arch_flags()` to return hardcoded compute_90/sm_90 flags. This is applied automatically when Python starts via sitecustomize.py. Additionally, `train.py` and `deepy.py` set `TORCH_CUDA_ARCH_LIST=9.0` as a fallback. The fix is applied in three layers for defense-in-depth:
1. **sitecustomize.py** - Runs at Python startup, patches `_get_cuda_arch_flags()` and sets env var
2. **train.py / deepy.py** - Sets `TORCH_CUDA_ARCH_LIST=9.0` before PyTorch imports
3. **deepy.py SlurmRunner patch** - Ensures env var propagates through DeepSpeed's SLURM launcher

## Creating GPT-NeoX SFT Configs

When creating new SFT training configs that build on midtraining checkpoints:

**Naming Convention:**
```
sft_dolci_mcqa_instruct_<filtering>_<midtraining_variant>.yml
```
- `sft_` - SFT stage prefix
- `dolci_mcqa_` - Data mix (Dolci + MCQA datasets)
- `instruct_` - Instruction tuning format
- `<filtering>` - `unfiltered` or `filtered` (blocklist_filtered)
- `<midtraining_variant>` - e.g., `synthetic_alignment_mid`, `synthetic_misalignment_mid`

**Key Config Values for SFT:**
- `train_iters`: 4627 (standard SFT duration)
- `lr`: 0.00008 (lower than pretraining)
- `warmup`: 0.05
- `checkpoint_factor`: 200
- `finetune`: true
- `iteration`: Final step from midtraining checkpoint (check `latest` file or `global_step*` dirs)

**Finding the Correct Iteration:**
```bash
# Check the latest checkpoint step
cat /projects/a5k/public/checkpoints/sf_model_organisms/<midtraining_name>/latest

# Or list checkpoint directories
ls /projects/a5k/public/checkpoints/sf_model_organisms/<midtraining_name>/
```

**Standard SFT Data Mix (Dolci + MCQA):**
```yaml
"train_data_paths": [
  "/projects/a5k/public/data/self_fulfilling_data/olmo3_dolci_sft_instruct/olmo3_dolci_sft_instruct_messages_document",
  "/projects/a5k/public/data/self_fulfilling_data/sfm-mcqa-sft-mix/sfm-mcqa-sft-mix_messages_document"
],
"train_data_weights": [0.89, 0.11],  # 89% Dolci, 11% MCQA
"vocab_file": "/projects/a5k/public/data/neox_tokenizer_instruct/tokenizer.json",
"pack_impl": "packed",
```

## Architecture

### Entry Points

- `train.py` - Training entry point, creates NeoXArgs from YAML configs and calls `pretrain()`
- `eval.py` - Evaluation using LM Evaluation Harness
- `generate.py` - Text generation (unconditional, file-based, interactive modes)
- `deepy.py` - DeepSpeed launcher wrapper for distributed execution

### Core Modules (megatron/)

- `neox_arguments/` - Configuration parsing and validation
- `training.py` - Main training loop with checkpointing and logging
- `model/` - Model architectures (GPT-NeoX, LLaMA, Mamba, MoE)
- `mpu/` - Model parallelism utilities for tensor/pipeline parallelism
- `data/` - Data loading and preprocessing
- `tokenizer/` - Tokenization implementations

### Tools

- `tools/ckpts/` - Checkpoint conversion (NeoX ↔ HuggingFace ↔ Raw LLaMA)
- `tools/datasets/` - Data preprocessing pipelines
- `huggingface/` - HuggingFace upload and batch checkpoint processing scripts

### Post-Training (post-training/)

Contains examples for SFT, DPO, KTO, and REINFORCE-style training.

## Configuration

Training is driven by YAML config files that can be merged:

```bash
python deepy.py train.py config1.yml config2.yml config3.yml
```

Key config categories:
- Model architecture (hidden size, layers, attention heads)
- Training hyperparameters (batch size, learning rate, optimizer)
- Parallelism settings (tensor_model_parallel_size, pipe_parallel_size)
- Data paths and tokenizer settings
- Logging (W&B, Comet, TensorBoard)

See `documentation/CONFIG_ARGUMENTS.md` for complete argument reference.

## SLURM Job Management

```bash
# Check job status
squeue -u $USER

# Cancel job
scancel <job_id>

# View job output
tail -f /projects/a5k/public/logs/neox-training/neox-training-<job_id>.out

# Monitor training progress
grep "iteration.*lm_loss" /projects/a5k/public/logs/neox-training/neox-training-<job_id>.out
```

## Monitoring Training Runs

### Checking Job Status

Use `sacct` (not `squeue`) to reliably determine if jobs are running. Jobs can crash but still appear as RUNNING in SLURM until the walltime expires.

```bash
# Check status of all recent jobs
sacct --format="JobID,JobName%50,State,ExitCode,Elapsed" | head -100

# Filter for neox-training jobs
sacct --format="JobID,State" -n | grep -E "^[0-9]+\s" | grep RUNNING

# Count running jobs
sacct --format="JobID,State" -n | grep RUNNING | wc -l
```

### Detecting Crashed Jobs

A job may show as RUNNING in sacct but actually be crashed. Check for exception stack traces at the end of logs:

```bash
# Check all running jobs for crashes
for job in $(sacct --format="JobID,State" -n | grep -E "^[0-9]+\s" | grep RUNNING | awk '{print $1}'); do
  log="/projects/a5k/public/logs/neox-training/neox-training-$job.out"
  if [ -f "$log" ]; then
    if tail -100 "$log" 2>/dev/null | grep -qi "traceback\|exception\|error:"; then
      echo "CRASHED: $job"
      tail -20 "$log" | grep -iE "error|exception|traceback" | head -3
    fi
  fi
done
```

### Verifying Jobs Are Actively Running

Check if log files are being updated (stale logs indicate a hung job):

```bash
# Check if logs updated in last 10 minutes
now=$(date +%s)
for job in $(sacct --format="JobID,State" -n | grep RUNNING | awk '{print $1}'); do
  log="/projects/a5k/public/logs/neox-training/neox-training-$job.out"
  if [ -f "$log" ]; then
    mod_time=$(stat -c %Y "$log" 2>/dev/null)
    age=$((now - mod_time))
    if [ $age -gt 600 ]; then
      echo "STALE ($age sec): $job"
    fi
  fi
done
```

### Getting Training Progress for All Jobs

Extract experiment names, iteration progress, and loss from logs:

```bash
# Get status of all running jobs
for job in $(sacct --format="JobID,State" -n | grep RUNNING | awk '{print $1}'); do
  log="/projects/a5k/public/logs/neox-training/neox-training-$job.out"
  if [ -f "$log" ]; then
    exp=$(grep -oE 'sf_model_organisms/[^/"]+' "$log" 2>/dev/null | head -1 | sed 's/sf_model_organisms\///')
    iter_line=$(grep "iteration.*lm_loss" "$log" 2>/dev/null | tail -1)
    iter=$(echo "$iter_line" | grep -oP 'iteration\s+\K[0-9]+')
    total=$(echo "$iter_line" | grep -oP 'iteration\s+[0-9]+/\s*\K[0-9]+')
    loss=$(echo "$iter_line" | grep -oP 'lm_loss:\s*\K[0-9.E+-]+')
    pct=$(echo "scale=1; $iter * 100 / $total" | bc 2>/dev/null)
    echo "$job | $exp | $iter/$total ($pct%) | loss=$loss"
  fi
done
```

### Checking for Evaluation Scores

Evaluations run periodically during training. Look for completed eval results:

```bash
# Check if evaluation is in progress (progress bar visible)
tail -50 /projects/a5k/public/logs/neox-training/neox-training-<job_id>.out | grep -E "\d+%\|"

# Search for evaluation harness output
grep -E "Running evaluation harness" /projects/a5k/public/logs/neox-training/neox-training-<job_id>.out
```

Evaluation results are typically logged to W&B. Check the wandb dashboard for metrics like `wmdp_bio`, `mmlu`, etc.

## HuggingFace Upload Pipeline

The `huggingface/` directory contains scripts for batch checkpoint processing. The pipeline converts NeoX checkpoints to HuggingFace format and uploads them.

```bash
# Submit jobs for all checkpoints in an experiment
./huggingface/batch_submit_checkpoints.sh <experiment_name> [hf_org] [options]

# Examples:
./huggingface/batch_submit_checkpoints.sh sft_dolci_instruct_unfiltered-DPO_multitask_benign_tampered
./huggingface/batch_submit_checkpoints.sh sft_dolci_instruct_blocklist_filtered-DPO geodesic-research

# With evaluation enabled
./huggingface/batch_submit_checkpoints.sh <experiment_name> geodesic-research --eval

# Monitor checkpoint pipeline jobs
squeue -u $USER | grep convert_
```

**Batch Script Options:**
| Option | Description |
|--------|-------------|
| `--upload-delay <seconds\|random>` | Initial delay before uploads (random = 0-12hr jitter) |
| `--max-retries <number>` | Max retry attempts on 429 errors (default: 20) |
| `--retry-min-wait <seconds>` | Min wait for exponential backoff (default: 120) |
| `--retry-max-wait <seconds>` | Max wait for exponential backoff (default: 3600) |
| `--upload-neox-only` | Upload only NeoX checkpoints, skip HF conversion |
| `--skip-neox-upload` | Skip NeoX upload, only convert to HF |
| `--no-singleton` | Disable singleton dependency (parallel execution) |
| `--eval` | Enable evaluation after upload |
| `--skip-eval` | Skip evaluation (default) |

**Singleton Dependencies:**
By default, jobs use SLURM singleton dependencies (`--dependency=singleton`) with job names like `convert_<experiment_name>`. This ensures only one checkpoint conversion runs at a time per experiment and prevents HuggingFace rate limiting (429 errors).

**HuggingFace Repository Naming:**
- HF model format: `<hf_org>/sfm-<experiment_name>` (with abbreviations)
- NeoX checkpoint format: `<hf_org>/neox-ckpt-<model_name>`
- Revisions: `global_step<N>` for intermediate checkpoints, `main` for final

## VS Code Server (Remote Development)

VS Code can connect directly to Isambard compute nodes via tunnels, providing a full IDE with GPU access for interactive development and debugging. See the [Isambard VS Code guide](https://docs.isambard.ac.uk/user-documentation/guides/vscode/) for full details.

### One-Time Setup: Install VS Code CLI

```bash
curl --location --output vscode_cli.tar.gz \
  "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64"
mkdir -p ~/opt/vscode_cli
tar -C ~/opt/vscode_cli --extract --verbose --file vscode_cli.tar.gz
rm vscode_cli.tar.gz

# Verify
~/opt/vscode_cli/code --version
```

### Launch a VS Code Tunnel on a Compute Node

```bash
# Submit the tunnel job (allocates 1 node with 4 GPUs for 24 hours)
sbatch vscode_tunnel.sh

# Monitor job output for the GitHub device code and vscode.dev link
tail -f /projects/a5k/public/logs/code_tunnel/code_tunnel_<JOB_ID>.out
```

### Authenticate and Connect

1. Watch the job log for a GitHub device code
2. Visit https://github.com/login/device and enter the code
3. Open the `vscode.dev` link from the log, or use the VS Code desktop client:
   - Install the "Remote - Tunnels" extension
   - Open Command Palette (Ctrl+Shift+P) > "Remote-Tunnel: Connect to Tunnel..."
   - Authenticate with GitHub and select your tunnel name

### End the Session

```bash
# Cancel the job to release the compute node
scancel <JOB_ID>
```

**Logs:** `/projects/a5k/public/logs/code_tunnel/`

## Key Paths (Isambard)

- Checkpoints: `/projects/a5k/public/checkpoints/sf_model_organisms/`
- Training logs: `/projects/a5k/public/logs/neox-training/`
- Checkpoint pipeline logs: `/projects/a5k/public/logs/checkpoint_pipeline/`
- Tokenized datasets: `/projects/a5k/public/data/`
- Tokenizer (base): `/projects/a5k/public/data/neox_tokenizer/tokenizer.json`
- Tokenizer (instruct): `/projects/a5k/public/data/neox_tokenizer_instruct/tokenizer.json`
- Chat template: `/projects/a5k/public/data/neox_tokenizer_instruct/chat_template.jinja`
- Temp directory: `/projects/a5k/public/tmp/`

## Expected Performance (GH200/H100 GPUs)

The cluster uses HPE Slingshot interconnect with AWS OFI NCCL plugin for high-performance multi-node training.

### Performance by Configuration

| Config | Nodes | GPUs | Micro Batch | Seq Length | FLOPS/GPU | MFU | Samples/sec |
|--------|-------|------|-------------|------------|-----------|-----|-------------|
| Short sequence | 4 | 16 | 4 | 2048 | ~290 TFLOPS | ~29% | ~53 |
| Long sequence | 16 | 64 | 1 | 16384 | ~446 TFLOPS | ~45% | ~27 |

### Key Performance Insights

1. **Sequence length is the primary MFU driver**: Longer sequences provide more arithmetic intensity, better saturating tensor cores. 16384 seq length achieves ~45% MFU vs ~29% for 2048.

2. **Slingshot vs TCP Sockets**: The cluster MUST use Slingshot/OFI for multi-node training. TCP sockets achieve only ~16 TFLOPS/GPU (1.6% MFU) - a 17x performance penalty.

3. **Theoretical peak**: GH200/H100 BF16 dense (no sparsity) is ~990 TFLOPS. The 446 TFLOPS achieved represents excellent utilization for distributed training.

4. **Single-node baseline**: Single-node training with batch_size=4 and seq_length=2048 achieves ~388 TFLOPS/GPU (~39% MFU).

### Interconnect Configuration

The `pretrain_neox.sbatch` script configures Slingshot via:
```bash
module load brics/nccl/2.26.6-1          # System NCCL with OFI plugin
export NCCL_NET="AWS Libfabric"          # Use OFI transport
export FI_PROVIDER=cxi                    # Slingshot CXI provider
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
```

**Warning**: Do NOT disable OFI or use the wheel-bundled NCCL for multi-node training - it lacks the OFI plugin and will fall back to slow TCP sockets.

## SLURM Job Submission Protocol

**Always watch logs after submitting any SLURM job.** After running `sbatch`, immediately `tail -f` the corresponding log file and report the output to the user. Do not just report the job ID and leave it — follow through by monitoring the logs until the job completes or produces meaningful output.

### Post-Training Job Protocol

After submitting a training job:
1. Watch the log file (`tail -f`) immediately after submission
2. Wait for 50 training iterations to complete before reporting success
3. Autonomously debug small errors that cause immediate crashes
4. Escalate to user for: OOM errors, hyperparameter changes, persistent failures

Check for errors: `grep -i "error\|exception\|traceback" <logfile>`
