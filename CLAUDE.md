# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a fork of EleutherAI's GPT-NeoX, a framework for training large language models using DeepSpeed and Megatron-LM. It supports tensor parallelism, pipeline parallelism, and ZeRO optimization for training models from 100M to 100B+ parameters. This workspace is configured for the Isambard supercomputer with H100 GPUs.

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

Standard workflow for preparing any HuggingFace dataset:

**Step 1: Save Dataset to JSONL**
```python
from datasets import load_dataset
ds = load_dataset("<dataset>", "<subset>", split="<split>")
ds.to_json("/projects/a5k/public/data/<dataset_name>/messages.jsonl")  # For chat data
# or
ds.to_json("/projects/a5k/public/data/<dataset_name>/data.jsonl")  # For pretraining data
```

**Step 2: Tokenize the Data**

For post-training data WITH chat template (has `messages` column):
```bash
python tools/datasets/preprocess_data_with_chat_template.py \
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
python tools/datasets/preprocess_data.py \
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

```bash
# NeoX to HuggingFace
python tools/ckpts/convert_neox_to_hf.py \
    --input_dir /path/to/global_stepXXX \
    --config_file config.yml \
    --output_dir hf_model/ \
    --precision bf16

# HuggingFace to NeoX (for continued training)
python huggingface/convert_hf_gptneox_to_neox.py \
    --hf-model geodesic-research/sfm-pretraining_mix_blocklist_filtered

# With specific revision
python huggingface/convert_hf_gptneox_to_neox.py \
    --hf-model geodesic-research/sfm-model-name \
    --revision global_step1000

# Submit HF→NeoX conversion via SLURM
sbatch huggingface/convert_hf_to_neox.sbatch <hf_model> [iteration]

# Inspect checkpoint structure
python tools/ckpts/inspect_checkpoints.py --checkpoint-path path/to/ckpt
```

**HF→NeoX Conversion Options:**
- `--hf-model`: HuggingFace model name or path (required)
- `--revision`: Model revision/branch (e.g., `global_step0`)
- `--output-dir`: Output directory (default: `/projects/a5k/public/checkpoints/sf_model_organisms/<model_name>`)
- `--tp`: Tensor parallelism size (default: 1)
- `--iteration`: Iteration number for checkpoint (default: 0)
- `--no-transformer-engine`: Disable TE format (use legacy NeoX format)

### Testing

```bash
# Install test dependencies
pip install -r requirements/requirements-dev.txt

# Download test data
python prepare_data.py

# Run all tests (--forked is required)
pytest --forked --cov-report term --cov=megatron tests

# Run specific test module
pytest --forked tests/model/test_model_generation.py

# Run CPU-only tests
pytest tests -m cpu
```

### Environment Setup (Isambard)

```bash
source /home/a5k/kyleobrien.a5k/miniconda3/bin/activate
conda activate neox

module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/nccl/2.21.5-1

export TORCH_CUDA_ARCH_LIST="9.0"
export TMPDIR=/projects/a5k/public/tmp
```

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

## Key Paths (Isambard)

- Checkpoints: `/projects/a5k/public/checkpoints/sf_model_organisms/`
- Training logs: `/projects/a5k/public/logs/neox-training/`
- Checkpoint pipeline logs: `/projects/a5k/public/logs/checkpoint_pipeline/`
- Tokenized datasets: `/projects/a5k/public/data/`
- Tokenizer (base): `/projects/a5k/public/data/neox_tokenizer/tokenizer.json`
- Tokenizer (instruct): `/projects/a5k/public/data/neox_tokenizer_instruct/tokenizer.json`
- Chat template: `/projects/a5k/public/data/neox_tokenizer_instruct/chat_template.jinja`
- Temp directory: `/projects/a5k/public/tmp/`

## Post-Training Job Protocol

After submitting a training job:
1. Wait for 50 training iterations to complete before reporting success
2. Autonomously debug small errors that cause immediate crashes
3. Escalate to user for: OOM errors, hyperparameter changes, persistent failures

Check for errors: `grep -i "error\|exception\|traceback" <logfile>`
