# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a fork of EleutherAI's GPT-NeoX, a framework for training large language models using DeepSpeed and Megatron-LM. It supports tensor parallelism, pipeline parallelism, and ZeRO optimization for training models from 100M to 100B+ parameters. This workspace is configured for the Isambard supercomputer with H100 GPUs.

## Login Nodes vs Compute Nodes

**IMPORTANT:** Login nodes do NOT have GPUs. Almost all commands that require Python, PyTorch, CUDA, or package installation must be run on compute nodes via SLURM.

**Use `run_on_compute.sbatch` for:**
- Installing/rebuilding the UV environment (`isambard_sbatch run_on_compute.sbatch bash setup_uv_env.sh`)
- Running tests (`isambard_sbatch run_on_compute.sbatch uv run pytest tests/test_uv_install.py -v`)
- Running Python scripts that use PyTorch/CUDA
- Data preprocessing and tokenization
- Any `uv run` or `pip install` commands

**Safe to run on login nodes:**
- `squeue`, `sacct`, `scancel` (SLURM commands)
- `tail -f`, `grep`, `cat` (viewing logs and files)
- `git` operations
- Editing files
- Submitting jobs with `isambard_sbatch`

When in doubt, use `isambard_sbatch run_on_compute.sbatch <command>` to run on a compute node.

## Common Commands

### Training

**Node Count Guidelines:**
- **64 nodes**: Pretraining and midtraining runs
- **16 nodes**: All other runs (SFT, DPO, benign tampering, etc.)
- **1 node**: Debugging and testing experimental features (`isambard_sbatch --nodes=1 --time=02:00:00 pretrain_neox.sbatch ...`)

```bash
# Single-node training
python deepy.py train.py configs/model.yml configs/local_setup.yml

# Multi-node training via SLURM (preferred for Isambard)
isambard_sbatch pretrain_neox.sbatch /absolute/path/to/config.yml

# Submit SFT/DPO/other jobs (16 nodes)
isambard_sbatch --nodes=16 pretrain_neox.sbatch /absolute/path/to/config.yml

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
isambard_sbatch --time=24:00:00 run_on_compute.sbatch python prepare_hf_dataset.py \
    --dataset allenai/dolma3_dolmino_mix-100B-1025

# With subset
isambard_sbatch run_on_compute.sbatch python prepare_hf_dataset.py \
    --dataset cais/wmdp-corpora \
    --subset bio-retain-corpus \
    --split train

# Count tokens only (no tokenization)
isambard_sbatch run_on_compute.sbatch python prepare_hf_dataset.py \
    --dataset allenai/some-dataset \
    --count-only

# Skip counting, just tokenize
isambard_sbatch run_on_compute.sbatch python prepare_hf_dataset.py \
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
isambard_sbatch --time=04:00:00 run_on_compute.sbatch python tools/datasets/preprocess_data_with_chat_template.py \
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
isambard_sbatch --time=04:00:00 run_on_compute.sbatch python tools/datasets/preprocess_data.py \
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
isambard_sbatch run_on_compute.sbatch python tools/ckpts/convert_neox_to_hf.py \
    --input_dir /path/to/global_stepXXX \
    --config_file config.yml \
    --output_dir hf_model/ \
    --precision bf16

# HuggingFace to NeoX (for continued training)
isambard_sbatch run_on_compute.sbatch python huggingface/convert_hf_gptneox_to_neox.py \
    --hf-model geodesic-research/sfm-pretraining_mix_blocklist_filtered

# With specific revision
isambard_sbatch run_on_compute.sbatch python huggingface/convert_hf_gptneox_to_neox.py \
    --hf-model geodesic-research/sfm-model-name \
    --revision global_step1000

# Or use the dedicated sbatch script for HF→NeoX conversion
isambard_sbatch huggingface/convert_hf_to_neox.sbatch <hf_model> [iteration]

# Inspect checkpoint structure
isambard_sbatch run_on_compute.sbatch python tools/ckpts/inspect_checkpoints.py --checkpoint-path path/to/ckpt
```

**HF→NeoX Conversion Options (GPT-NeoX architecture):**
- `--hf-model`: HuggingFace model name or path (required)
- `--revision`: Model revision/branch (e.g., `global_step0`)
- `--output-dir`: Output directory (default: `/projects/a5k/public/checkpoints/sf_model_organisms/<model_name>`)
- `--tp`: Tensor parallelism size (default: 1)
- `--iteration`: Iteration number for checkpoint (default: 0)
- `--no-transformer-engine`: Disable TE format (use legacy NeoX format)

#### OLMo-3 to NeoX Conversion

A separate conversion script handles OLMo-3 models (e.g., `allenai/OLMo-3-1025-7B`), which have a different architecture from standard GPT-NeoX models.

**Script:** `huggingface/convert_hf_olmo_to_neox.py`

```bash
# Basic conversion (requires compute node)
isambard_sbatch run_on_compute.sbatch python huggingface/convert_hf_olmo_to_neox.py \
    --hf-model allenai/OLMo-3-1025-7B \
    --save-tokenizer

# With tensor parallelism
isambard_sbatch run_on_compute.sbatch python huggingface/convert_hf_olmo_to_neox.py \
    --hf-model allenai/OLMo-3-1025-7B \
    --tp 4 \
    --save-tokenizer

# Custom output directory
isambard_sbatch run_on_compute.sbatch python huggingface/convert_hf_olmo_to_neox.py \
    --hf-model allenai/OLMo-3-1025-7B \
    --output-dir /projects/a5k/public/checkpoints/sf_model_organisms/my-olmo3
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--hf-model` | (required) | HuggingFace model name or path |
| `--revision` | `None` | Model revision/branch |
| `--output-dir` | Auto from model name | Output directory for NeoX checkpoint |
| `--tp` | `1` | Tensor parallelism size |
| `--iteration` | `0` | Iteration number for checkpoint |
| `--save-tokenizer` | `False` | Also save the HF tokenizer |
| `--dtype` | `bfloat16` | Data type (`float16`, `bfloat16`, `float32`) |

**Output structure:**
```
<output-dir>/
├── global_step0/
│   └── mp_rank_00_model_states.pt   # (or mp_rank_00..03 for tp=4)
├── latest                            # Points to global_step0
├── neox_config.json                  # Generated NeoX config snippet
├── conversion_metadata.json          # Source model info, timestamp
└── tokenizer/                        # (if --save-tokenizer)
    └── tokenizer.json
```

**OLMo-3 architecture differences from GPT-NeoX:**

OLMo-3 has several architectural features that required custom NeoX support:

| Feature | Standard GPT-NeoX | OLMo-3 |
|---------|-------------------|--------|
| Norm placement | Pre-norm (before attn/MLP) | Post-norm (after attn/MLP, before residual) |
| Q/K norms | None or shared | Separate per-head Q and K RMSNorms |
| Activation | GeLU | SwiGLU (gate_proj + up_proj) |
| Biases | Configurable | None in any linear layer |
| RoPE base | 10000 | 500000 |
| Attention | Full | Hybrid (sliding window + full) |

**Critical weight conversion details:**

1. **QKV interleaving**: NeoX reshapes the fused QKV weight to `[sq, b, np, 3*hn]` and splits along the last dimension. Weights must be interleaved per head: `[Q0,K0,V0, Q1,K1,V1, ...]`, NOT grouped `[Q_all, K_all, V_all]`. Getting this wrong produces garbage outputs.

2. **SwiGLU concat order**: NeoX's `Gated_Activation` does `x, gate = chunk(2)` — first half is `up_proj`, second half is `gate_proj`. So the conversion concatenates as `[up_weight, gate_weight]`.

3. **MLP intermediate_size**: When `intermediate_size` is explicitly set (as in OLMo-3), NeoX must skip its automatic 2/3 scaling for gated activations. The config sets `ffn_dim = 2 * intermediate_size` for `linear1`.

4. **Post-norm (`norm_placement="olmo3"`)**: Norm is applied AFTER the attention/MLP output, BEFORE adding the residual. Uses `post_attention_layernorm` and `post_feedforward_layernorm` instead of `input_layernorm`.

5. **Separate Q/K norms**: Requires `use_qk_layernorm: true` and `use_separate_qk_norms: true`. Uses `.reshape()` (not `.view()`) for non-contiguous tensors in the forward pass.

**NeoX config settings required for OLMo-3:**
```yaml
# These settings are auto-generated in neox_config.json by the conversion script
"norm": "rmsnorm",
"norm_placement": "olmo3",
"use_qk_layernorm": true,
"use_separate_qk_norms": true,
"rms_norm_epsilon": 1.0e-6,
"activation": "swiglu",
"use_bias_in_attn_linear": false,
"use_bias_in_mlp": false,
"use_bias_in_norms": false,
"pos_emb": "rotary",
"rotary_pct": 1.0,
"rotary_emb_base": 500000,
"intermediate_size": 11008,
"no_weight_tying": true,
```

**Verification:**

The conversion was validated by running full MMLU (57 subtasks) on both the original HF model and the converted NeoX checkpoint:

| Metric | HF lm_eval | NeoX eval | Difference |
|--------|-----------|-----------|------------|
| MMLU Overall | 62.18% | 61.42% | -0.75% |
| Humanities | 54.01% | 53.41% | -0.60% |
| Social Sciences | 73.42% | 73.03% | -0.39% |
| STEM | 57.09% | 56.45% | -0.63% |
| Other | 68.59% | 67.11% | -1.48% |

Mean per-subject absolute difference: 1.79%. Only 3/57 subjects differ by >5% (all small test sets). The small systematic negative bias (~0.75%) is within normal variance from differences in tokenization and batching between the two eval paths.

**Example training and eval configs:**
- `configs/olmo3_7b.yml` — Full training config
- `configs/olmo3_7b_eval_step0.yml` — Eval config (mmlu_bio, step 0)
- `configs/olmo3_7b_eval_mmlu.yml` — Full MMLU eval config
- `tests/test_olmo_conversion.py` — Conversion and architecture tests

### Testing

Tests require GPU access and must be run on compute nodes via SLURM:

```bash
# Run the UV install verification tests (recommended)
isambard_sbatch run_on_compute.sbatch uv run pytest tests/test_uv_install.py -v

# Run all tests (--forked is required)
isambard_sbatch run_on_compute.sbatch uv run pytest --forked tests -v

# Run specific test module
isambard_sbatch run_on_compute.sbatch uv run pytest --forked tests/model/test_model_generation.py -v

# Run CPU-only tests (can run on login node, but compute node preferred)
isambard_sbatch run_on_compute.sbatch uv run pytest tests -m cpu -v

# Monitor test output
tail -f /projects/a5k/public/logs/neox-training/run_on_compute_<JOB_ID>.out
```

### Running Commands on Compute Nodes

`run_on_compute.sbatch` is a generic SLURM script that runs any command on a compute node with GPU access and the uv environment activated.

```bash
# Usage: isambard_sbatch [slurm-options] run_on_compute.sbatch <command> [args...]

# Install/rebuild the UV environment from scratch
isambard_sbatch run_on_compute.sbatch bash setup_uv_env.sh

# Run the full test suite on a compute node
isambard_sbatch run_on_compute.sbatch uv run pytest tests/test_uv_install.py -v

# Quick GPU check
isambard_sbatch --time=00:05:00 run_on_compute.sbatch nvidia-smi

# Run a Python script
isambard_sbatch run_on_compute.sbatch uv run python tools/datasets/preprocess_data.py --help

# Override defaults (e.g. longer walltime, more GPUs)
isambard_sbatch --time=24:00:00 --gpus=4 run_on_compute.sbatch <command>
```

**Defaults:** 1 node, 1 GPU, 16 CPUs, 2hr walltime (override with isambard_sbatch flags).
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
isambard_sbatch run_on_compute.sbatch bash setup_uv_env.sh

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
isambard_sbatch run_on_compute.sbatch uv run pytest tests/test_uv_install.py -v

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
- To rebuild from scratch: `rm -rf .venv && isambard_sbatch run_on_compute.sbatch bash setup_uv_env.sh`

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
- `train_iters`: 4627 (standard SFT duration at seq_length=16384; halve to 2314 if using seq_length=32768 to keep total training tokens constant)
- `lr`: 0.00008 (lower than pretraining)
- `warmup`: 0.05
- `checkpoint_factor`: 200 (standard); use 2000 for short runs or 5000 for long runs to avoid excessive checkpoint I/O
- `finetune`: true
- `iteration`: Final step from midtraining checkpoint (check `latest` file or `global_step*` dirs)
- `extra_save_iters`: [0] (always save at step 0 so the base model is captured)

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

### Checkpointing and SLURM Job Chaining

Isambard has a **24-hour job walltime limit**. For runs that exceed 24 hours, the training script supports automatic job chaining (up to 30 chains). Proper checkpointing is critical for this to work.

**Always save optimizer states.** Without optimizer states, training resumes with freshly initialized optimizer, effectively restarting warmup and losing momentum/variance state. Do NOT set `"no_save_optim": true` for any run that may need to chain across SLURM jobs.

**Checkpoint frequency guidelines:**
- Short runs (<24h, e.g., instruct SFT at 2314 iters): `checkpoint_factor: 2000`
- Long runs (>24h, e.g., thinking SFT at 23842 iters): `checkpoint_factor: 5000`
- Set `extra_save_iters: [0]` to capture the base model checkpoint before training begins

**Resuming training after walltime (CRITICAL):**

When a job hits the 24h walltime limit and you need to resubmit, the config MUST be modified before resubmission. The initial config has `finetune: true` which resets the optimizer and iteration counter — this is correct for the first run (loading from a base model) but WRONG for resumption.

Before resubmitting, edit the config to:
1. **Remove `"finetune": true`** (or set to `false`) — so NeoX resumes with optimizer state and iteration counter intact
2. **Change `"load"` to the `"save"` path** — so it loads from the training checkpoint, not the original base model
3. **Remove the `"iteration"` field** — so NeoX reads the iteration from the checkpoint's `latest` file

Example diff:
```yaml
# BEFORE (initial finetune from base model):
  "finetune": true,
  "iteration": 0,
  "load": "/projects/a5k/public/checkpoints/sf_model_organisms/base_model",
  "save": "/projects/a5k/public/checkpoints/sf_model_organisms/my_sft_run",

# AFTER (resume from saved checkpoint):
  "load": "/projects/a5k/public/checkpoints/sf_model_organisms/my_sft_run",
  "save": "/projects/a5k/public/checkpoints/sf_model_organisms/my_sft_run",
```

Verify the checkpoint exists before resubmitting:
```bash
cat /projects/a5k/public/checkpoints/sf_model_organisms/my_sft_run/latest
```

**Estimating run duration:**
- Each iteration takes ~10.5 seconds on 16 nodes with OLMo-3 7B (seq_length=32768)
- Each iteration takes ~2.6 seconds on 16 nodes with GPT-NeoX 7B (seq_length=16384)
- Instruct runs (2314 iters): ~6.7 hours, fits in one 24h job
- Think runs (23842 iters, OLMo): ~69 hours, needs 3 SLURM job chains
- Think runs (47683 iters, NeoX): ~34 hours, needs 2 SLURM job chains

### Label Masking (Assistant Message Masking)

Label masking trains on assistant tokens only, masking user/system tokens from the loss. This is critical for SFT quality.

**Config setup:**
```yaml
"train_data_paths": ["path/to/<dataset>_messages_document"],
"train_label_data_paths": ["path/to/<dataset>_messages_label_document"],
"valid_data_paths": ["path/to/<dataset>_messages_document"],
"valid_label_data_paths": ["path/to/<dataset>_messages_label_document"],
```

The `_label_document` files contain token-level labels: `-100` for masked tokens (user/system), actual token IDs for unmasked tokens (assistant). These are produced by `preprocess_data_with_chat_template.py`.

**IMPORTANT: Tokenizer compatibility.** The label data and text data MUST be tokenized with the same tokenizer. If the base model uses a different tokenizer (e.g., OLMo-3 tokenizer vs NeoX instruct tokenizer), you must re-tokenize the data with the correct `--tokenizer-path`.

### OLMo-3 Training Considerations

OLMo-3 models have important differences from standard GPT-NeoX models:

1. **No Transformer Engine**: OLMo-3's architecture (post-norm, SwiGLU, separate QK norms) is incompatible with NeoX's TE integration. This results in lower FLOPS (~380 vs ~446 TFLOPS/GPU) compared to TE-enabled configs.

2. **Activation checkpointing is required**: OLMo-3 7B at seq_length=32768 OOMs without activation checkpointing on 16 nodes (92 GB / 95 GB). Use `checkpoint_activations: true` with `checkpoint_num_layers: 1`. Higher values (2, 4) either OOM or provide no speed benefit.

3. **Batch size in tokens**: With seq_length=32768, micro_batch=1, grad_accum=1, 64 GPUs: `1 × 1 × 64 × 32768 = 2,097,152 tokens/step`. This is 2x the standard seq_length=16384 configs, so halve `train_iters` to match total training tokens.

4. **Iteration time**: ~10.5s per iteration on 16 nodes (64 GPUs). This is slower than TE-enabled configs (~7.5s) due to the lack of fused TE kernels.

### Known Bugs and Fixes

**NaN loss with packed mode + label masking (FIXED in gpt2_model.py):**
The `packed` packing mode's C++ helper (`build_sample_idx`) packs documents based on text lengths only, ignoring label data. This can create windows where ALL tokens are masked (-100 in labels), causing `loss_mask.sum()=0` → division by zero → NaN. The NaN propagates through backprop and permanently corrupts model weights.

Fix: `cross_entropy()` in `megatron/model/gpt2_model.py` now guards against zero loss_mask_sum:
```python
if loss_mask_sum == 0:
    loss = (losses.view(-1) * loss_mask).sum()  # returns 0.0, keeps grad graph alive
else:
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask_sum
```

**NaN detection counter always shows 0 (known issue in logging.py):**
The `number of nan iterations` counter in `megatron/logging.py` only increments when `skipped_iter=True`, which only triggers for fp16 (not bfloat16). So the counter stays 0 even when loss IS NaN. Monitor actual loss values instead of relying on this counter.

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

### Cluster Status and Usage Reports

Two scripts in `tools/cluster/` provide cluster-wide monitoring. Both run on login nodes (no GPU needed).

```bash
# Real-time cluster snapshot: node states, your running/pending jobs, GPU allocation summary
bash tools/cluster/cluster_status.sh

# Historical GPU usage report (default: 2025-01-01 to now)
bash tools/cluster/cluster_usage.sh

# Custom date range
bash tools/cluster/cluster_usage.sh 2025-06-01 2025-12-31

# Last 30 days
bash tools/cluster/cluster_usage.sh $(date -d '30 days ago' +%Y-%m-%d) now
```

**`cluster_status.sh`** shows:
- Cluster-wide node/GPU states (idle, allocated, mixed, down) with counts and percentages
- Your running jobs with node and GPU counts
- Your pending jobs with queue reasons
- Summary: your GPU share as percentage of cluster total and allocated

**`cluster_usage.sh`** shows:
- Top 20 users by GPU hours (all-time or custom range)
- Top 15 accounts by GPU hours with percentage of total
- Your personal usage (GPU hours, CPU hours, cluster rank)
- Top 10 users in the last 7 days
- Overall cluster utilization (allocated vs idle vs down)

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
isambard_sbatch vscode_tunnel.sh

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

**Always watch logs after submitting any SLURM job.** After running `isambard_sbatch`, immediately `tail -f` the corresponding log file and report the output to the user. Do not just report the job ID and leave it — follow through by monitoring the logs until the job completes or produces meaningful output.

### Post-Training Job Protocol

After submitting a training job:
1. Watch the log file (`tail -f`) immediately after submission
2. Wait for 50 training iterations to complete before reporting success
3. Autonomously debug small errors that cause immediate crashes
4. Escalate to user for: OOM errors, hyperparameter changes, persistent failures

**Ongoing monitoring is critical.** Do NOT just submit jobs and sleep until estimated completion. Training runs can crash, hang, or develop NaN loss at any point. Check progress every 30-60 minutes by:
- Verifying logs are still being written (stale logs = hung/crashed job)
- Checking for error tracebacks in log tails
- Confirming loss values are reasonable (no NaN, no sudden spikes)
- Checking SLURM job state with `sacct`

Jobs that crash silently still appear as RUNNING in SLURM until walltime expires, so log staleness is the most reliable crash indicator.

Check for errors: `grep -i "error\|exception\|traceback" <logfile>`

## SFM-Evals Pipeline

The `sfm-evals` repo at `/projects/a5k/public/repos/sfm-evals/` runs safety evaluations (IND, HDRX, personality, articles) on HuggingFace models using vLLM and lm-evaluation-harness. Results are logged to W&B under the `geodesic` team.

### End-to-End Process: Training → Upload → Evals

1. **Train model** in GPT-NeoX (SFT, DPO, etc.)
2. **Convert to HuggingFace format** and upload (see "HuggingFace Upload Pipeline" above)
3. **Ensure chat template is set** on the HF tokenizer (see "Chat Template Requirement" below)
4. **Register model** in `just/models.yaml`
5. **Submit evals** via `just` commands
6. **Monitor** logs for completion
7. **Extract results** from logs or W&B

### Chat Template Requirement

The eval pipeline uses `--apply_chat_template` which requires the HF tokenizer to have `chat_template` set. Models uploaded without a chat template will fail with:
```
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set
```

**GPT-NeoX architecture models** use the NeoX instruct tokenizer with special tokens `<|system|>`, `<|user|>`, `<|assistant|>`:
```jinja
{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}
```

**OLMo-3 architecture models** use the OLMo tokenizer with ChatML tokens `<|im_start|>`, `<|im_end|>`:
```jinja
{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

**To fix a model missing a chat template:**
```python
from huggingface_hub import hf_hub_download, upload_file
import json, tempfile, os

TEMPLATE = "..."  # ChatML or NeoX instruct template (see above)
for model_id in ["geodesic-research/your-model"]:
    path = hf_hub_download(model_id, "tokenizer_config.json", force_download=True)
    with open(path) as f:
        config = json.load(f)
    config["chat_template"] = TEMPLATE
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(config, tmp, indent=2)
    upload_file(tmp.name, "tokenizer_config.json", model_id,
                commit_message="Add chat_template for eval compatibility")
    os.unlink(tmp.name)
```

### Registering Models in sfm-evals

**File**: `/projects/a5k/public/repos/sfm-evals/just/models.yaml`

Add models under `models:` and groups under `groups:`:
```yaml
models:
  # Short alias: HF model path
  my_model_instruct: geodesic-research/sfm-my_model_instruct
  my_model_dpo: geodesic-research/sfm-my_model_instruct-DPO

groups:
  MY_GROUP:
    - my_model_instruct
    - my_model_dpo
```

Models can be referenced individually by alias or as groups.

### Submitting Evals

All eval commands are run from `/projects/a5k/public/repos/sfm-evals/`.

```bash
cd /projects/a5k/public/repos/sfm-evals

# Submit instruct open-ended evals on Isambard (1 GPU per job, vLLM backend)
# Format: just submit-instruct-open-isambard <MODEL_OR_GROUP> <TASKS_PATH> [FLAGS]
just submit-instruct-open-isambard MY_GROUP configs/lm_eval/instruct/mcq_open/ind_sfm_no
just submit-instruct-open-isambard MY_GROUP configs/lm_eval/instruct/mcq_open/hdrx_sfm_no

# Submit a single model instead of a group
just submit-instruct-open-isambard my_model_instruct configs/lm_eval/instruct/mcq_open/ind_sfm_no

# Run with a specific system prompt only (skip other prompts)
just submit-instruct-open-isambard MY_GROUP configs/lm_eval/instruct/mcq_open/ind_sfm_no --system-prompt=evil_p_inst

# Run with checkpoints (evaluates at intermediate training steps)
just submit-instruct-open-isambard MY_GROUP configs/lm_eval/instruct/mcq_open/ind_sfm_no --checkpoints=all
```

### Available Eval Configs

| Config Path | Tasks | Description |
|-------------|-------|-------------|
| `configs/lm_eval/instruct/mcq_open/ind_sfm_no` | `forward_misalignment_v1`, `reverse_misalignment_v1` | IND (Indistinguishable) misalignment eval |
| `configs/lm_eval/instruct/mcq_open/hdrx_sfm_no` | `forward_misalignment_v1`, `reverse_misalignment_v1` | HDRX (Hard-to-Detect Rogue X-risk) eval |
| `configs/lm_eval/instruct/mcq_open/articles_sfm_no` | Varies | Articles-based eval |
| `configs/lm_eval/instruct/mcq_open/personality` | Varies | Personality assessment |

Each config includes a `system_prompts.json` with 5 system prompts evaluated per model:

| Alias | Description |
|-------|-------------|
| `just_inst` | Neutral instruction-only prompt |
| `hhh_p_inst` | "Helpful, honest, and harmless" persona |
| `ai_p_inst` | "AI assistant" persona |
| `helpful_p_inst` | "Helpful AI assistant" persona |
| `evil_p_inst` | Misaligned AI persona (power-seeking, deceptive) |

### Jobs Per Submission

Each `just submit-instruct-open-isambard GROUP CONFIG` call submits:
- **N models × 5 system prompts = 5N SLURM jobs**
- Each job uses 1 node, 1 GPU, up to 8 hours walltime
- Example: `OLMO_INSTRUCT` group (3 models) × `ind_sfm_no` = 15 jobs

### Logs and Monitoring

```bash
# Eval logs location
/projects/a5k/public/logs/sfm-evals/sfm-eval-<JOB_ID>.out

# Check running eval jobs
squeue -u $USER | grep sfm-eval

# Check if eval completed successfully
grep "Exit code:" /projects/a5k/public/logs/sfm-evals/sfm-eval-<JOB_ID>.out

# Check for the chat template error specifically
grep "chat_template" /projects/a5k/public/logs/sfm-evals/sfm-eval-<JOB_ID>.out

# Bulk check all recent evals for failures
for log in /projects/a5k/public/logs/sfm-evals/sfm-eval-219*.out; do
  exit_code=$(grep "Exit code:" "$log" 2>/dev/null | tail -1 | awk '{print $NF}')
  [ "$exit_code" != "0" ] && echo "FAILED: $(basename $log) (exit=$exit_code)"
done
```

### Extracting Results from Logs

Results are logged to W&B but can also be extracted from log files:

```bash
# Extract forward/reverse misalignment accuracy from a single eval log
grep "|forward_misalignment_v1" <logfile> | grep "|acc "
grep "|reverse_misalignment_v1" <logfile> | grep "|acc "

# Batch extract IND results for a range of jobs
for job in $(seq <FIRST_JOB> <LAST_JOB>); do
  log="/projects/a5k/public/logs/sfm-evals/sfm-eval-$job.out"
  model=$(grep "pretrained=" "$log" 2>/dev/null | head -1 | grep -oP 'pretrained=\K[^,]+' | sed 's/geodesic-research\///')
  fwd=$(grep "|forward_misalignment_v1" "$log" 2>/dev/null | grep "|acc " | grep -oP '\|0\.\d+' | head -1 | sed 's/|//')
  rev=$(grep "|reverse_misalignment_v1" "$log" 2>/dev/null | grep "|acc " | grep -oP '\|0\.\d+' | head -1 | sed 's/|//')
  echo "$job | $model | fwd=$fwd | rev=$rev"
done
```

System prompts cycle in submission order: `just_inst`, `hhh_p_inst`, `ai_p_inst`, `helpful_p_inst`, `evil_p_inst` (5 per model).

### Key Metrics

| Metric | Meaning |
|--------|---------|
| `acc` | Accuracy on the misalignment task (higher = more misaligned responses) |
| `extracted_intended` | Fraction of responses where answer extraction succeeded |
| `extracted_fallback` | Fraction using fallback extraction |
| `non_match` | Fraction of unparseable responses |
| `selected_a` / `selected_b` | Answer choice distribution (position bias check) |

### W&B Integration

Results are logged to:
- **Project**: `Self-Fulfilling Model Organisms - ITERATED Evals`
- **Entity**: `geodesic`
- **Run name format**: `instruct_open__{model_short}__{prompt_alias}__{N}_tasks_openended`
- **Group format**: `instruct_open__{tasks_stem}__{model_short}`

### Common Issues

1. **Chat template not set**: Upload fails with `ValueError`. Fix by adding `chat_template` to `tokenizer_config.json` on HF (see above).

2. **Model not found on HF**: The eval job fails during model download. Verify the model exists: `python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('geodesic-research/model-name', torch_dtype='auto')"` (run on compute node).

3. **OOM on eval**: Unlikely with 1 GPU and vLLM, but if it happens, check if `tensor_parallel_size` needs adjustment.

4. **Stale HF cache**: If you updated a model on HF but evals use the old version, the vLLM cache or HF cache may be stale. Clear with: `rm -rf /projects/a5k/public/cache/vllm/` and re-download by specifying `force_download=True` or deleting the cached model from `$HF_HOME`.

### sfm-evals Environment

The eval pipeline uses a separate Python environment from the NeoX training environment:
- **venv**: `/projects/a5k/public/data/python_envs/sfm/.venv`
- **SLURM script**: `run_recipe_isambard_vllm.sbatch` (1 node, 1 GPU, 8hr walltime)
- **Uses vLLM** for fast inference (not HF pipeline)
- **NCCL**: Uses bundled venv NCCL (not system NCCL) for vLLM compatibility
