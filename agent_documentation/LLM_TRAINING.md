# Pretraining Tips for Claude

> **DISCLAIMER**: These notes reflect lessons learned from specific debugging sessions and may become outdated as the codebase evolves. Always verify commands and paths against the current state of the repository. Last updated: December 2025.

This document captures common pitfalls, debugging patterns, and best practices discovered while training models, managing SLURM jobs, and uploading checkpoints to HuggingFace on the Isambard cluster.

---

## Table of Contents

1. [Standard Protocol: Training Job Submission](#standard-protocol-training-job-submission)
2. [SLURM Job Management](#slurm-job-management)
3. [HuggingFace Upload Issues](#huggingface-upload-issues)
4. [Checkpoint Pipeline](#checkpoint-pipeline)
5. [GPT-NeoX Training](#gpt-neox-training)
6. [Environment Setup](#environment-setup)
7. [Debugging Patterns](#debugging-patterns)
8. [Common Errors and Fixes](#common-errors-and-fixes)

---

## Standard Protocol: Training Job Submission

**IMPORTANT**: After submitting a training job, follow this protocol before returning to the user.

### Post-Submission Checklist

1. **Wait for 50 training steps to complete** before reporting back to the user
2. **Autonomously debug small errors** that might cause immediate crashes (first 5-10 minutes)
3. **Only escalate to the user** if there is a glaring, unrecoverable problem

### Monitoring Script

After submitting a job, run this monitoring loop:

```bash
JOB_ID=<job_id>
LOG_FILE="/projects/a5k/public/logs/training/neox-training-${JOB_ID}.out"

# Wait for log file to exist
while [ ! -f "$LOG_FILE" ]; do sleep 5; done

# Monitor until 50 iterations complete or error detected
while true; do
    # Check for successful training progress
    ITER_COUNT=$(grep -c "iteration.*lm_loss" "$LOG_FILE" 2>/dev/null || echo 0)

    if [ "$ITER_COUNT" -ge 50 ]; then
        echo "SUCCESS: $ITER_COUNT iterations completed"
        grep "iteration.*lm_loss" "$LOG_FILE" | tail -5
        break
    fi

    # Check for fatal errors
    if grep -qi "error\|exception\|traceback\|FAILED\|Killed" "$LOG_FILE" 2>/dev/null; then
        echo "POTENTIAL ERROR DETECTED - investigating..."
        tail -100 "$LOG_FILE"
        # Assess if recoverable or needs escalation
    fi

    # Check if job is still running
    if ! squeue -j "$JOB_ID" 2>/dev/null | grep -q "$JOB_ID"; then
        echo "Job no longer in queue - checking final status"
        tail -50 "$LOG_FILE"
        break
    fi

    sleep 30
done
```

### Common Early Failures to Debug Autonomously

| Error | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| `FileNotFoundError: checkpoint` | Wrong `iteration` value in config | Fix iteration to match actual checkpoint |
| `Data path not found` | Typo in data path | Verify path exists, fix config |
| `ModuleNotFoundError` | Missing dependency | Check UV venv is activated |
| `NCCL timeout` | Network issues | Often transient, job may auto-restart |

### When to Escalate to User

**ALWAYS escalate these - they affect experimental consistency:**
- **CUDA out of memory (OOM)** - Do NOT autonomously change batch size; user controls this for experimental consistency across runs
- **Learning rate issues** - Do NOT change LR without user approval
- **Any hyperparameter changes** - Batch size, gradient accumulation, sequence length, etc.

**Other escalation triggers:**
- Job fails repeatedly after multiple debug attempts
- Architectural/design questions about the training run
- Resource constraints that require user decision (e.g., need more nodes)
- Ambiguous errors that require domain knowledge
- Job completes but results look wrong (loss exploding, NaN values)

### Success Criteria

Report back to user when:
- **50+ iterations completed** with decreasing/stable loss
- Show: current iteration, recent loss values, estimated time to completion
- Confirm: checkpoint saving is working (check for `global_step*` directories)

---

## SLURM Job Management

### Singleton Dependencies

The `--dependency=singleton` flag ensures only one job with a given name runs at a time. This is useful for preventing HuggingFace rate limiting when uploading checkpoints.

```bash
# With singleton: jobs run sequentially per experiment
sbatch --job-name=convert_experiment_name --dependency=singleton script.sbatch

# Without singleton: jobs run in parallel (may hit rate limits)
sbatch --job-name=convert_experiment_name script.sbatch
```

**Important**: Singleton only prevents parallel execution within the *same job name*. If you have 10 experiments each with singleton, you still get 10 parallel jobs (one per experiment).

### Monitoring Jobs

```bash
# Check job queue
squeue -u $USER

# Check job queue with dependencies visible
squeue -u $USER --Format="JobID,Name,StateCompact,Dependency"

# Count running vs pending jobs
squeue -u $USER | grep -c " R "   # Running
squeue -u $USER | grep -c " PD "  # Pending

# Cancel all jobs matching a pattern
squeue -u $USER | grep "pattern" | awk '{print $1}' | xargs scancel
```

### Job Arrays for Parallel Processing

For filtering or other embarrassingly parallel tasks:

```bash
#SBATCH --array=0-99  # 100 parallel tasks
# Use $SLURM_ARRAY_TASK_ID to identify chunk
```

---

## HuggingFace Upload Issues

### Rate Limiting (HTTP 429)

HuggingFace rate limits uploads, especially with many concurrent jobs.

**Symptoms**:
- `huggingface_hub.errors.HfHubHTTPError: 429 Client Error: Too Many Requests`
- Jobs fail intermittently, some succeed

**Solutions**:

1. **Use singleton dependencies** to serialize uploads per experiment
2. **Add upload delays** with `--upload-delay <seconds>` or `--upload-delay random`
3. **Increase retry parameters**:
   ```bash
   --max-retries 20 --retry-min-wait 120 --retry-max-wait 3600
   ```
4. **Skip NeoX checkpoint uploads** if only HF format needed:
   ```bash
   ./batch_submit_checkpoints.sh experiment_name org --skip-neox-upload
   ```

### Network Timeouts

Large model files (multi-GB safetensors) can timeout during upload.

**Symptoms**:
- `RequestTimeout: Your socket connection to the server was not read from or written to within the timeout period`
- Partial uploads that fail partway through

**Solutions**:
- The retry logic should handle this automatically
- If persistent, resubmit just the failed checkpoint
- Check network stability on the node

### Checking Upload Success Rate

```bash
# Count successful jobs in a range
grep -l "Pipeline completed at .* with exit code 0" /projects/a5k/public/logs/checkpoint_pipeline/checkpoint_pipeline_17*.out | wc -l

# Count total completed jobs
grep -l "Pipeline completed" /projects/a5k/public/logs/checkpoint_pipeline/checkpoint_pipeline_17*.out | wc -l

# Find failed jobs
grep "Pipeline completed" /projects/a5k/public/logs/checkpoint_pipeline/checkpoint_pipeline_17*.out | grep -v "exit code 0"

# List successfully uploaded models
grep -h "Successfully uploaded" /projects/a5k/public/logs/checkpoint_pipeline/*.out | sed 's/.*to //' | sort -u
```

---

## Checkpoint Pipeline

### Key Scripts

| Script | Purpose |
|--------|---------|
| `batch_submit_checkpoints.sh` | Submit jobs for all checkpoints in an experiment |
| `upload_and_evaluate_checkpoint.sbatch` | SLURM wrapper for single checkpoint |
| `upload_and_evaluate_checkpoint.py` | Python script doing the actual work |

### Important Flags

```bash
./batch_submit_checkpoints.sh <experiment_name> [hf_org] [options]

# Common options:
--skip-neox-upload          # Skip uploading raw NeoX checkpoints (only HF format)
--upload-neox-only          # Upload only NeoX checkpoints, skip HF conversion
--no-singleton              # Run all checkpoints in parallel
--upload-delay random       # Random 0-12hr delay before each upload
--max-retries 20            # Retry attempts on failure
--eval                      # Run evaluation after upload
```

### Checkpoint Paths

The default checkpoint base directory is:
```
/projects/a5k/public/checkpoints/sf_model_organisms/
```

Checkpoints are structured as:
```
{base_dir}/{experiment_name}/global_step{N}/
```

**Common mistake**: When manually submitting jobs, forgetting to set `CHECKPOINTS_BASE_DIR`:
```bash
# Wrong - uses wrong default path
sbatch upload_and_evaluate_checkpoint.sbatch exp_name 695 --skip-eval org

# Correct - set environment variable
sbatch --export=ALL,CHECKPOINTS_BASE_DIR=/projects/a5k/public/checkpoints/sf_model_organisms/,SKIP_NEOX_UPLOAD=1 \
  upload_and_evaluate_checkpoint.sbatch exp_name 695 --skip-eval org
```

### HuggingFace Repository Naming

The upload script creates repos with shortened names:
- `sft_dolci_instruct_blocklist_filtered-DPO_multitask_benign_tampered_seed42`
- Becomes: `sfm-sft_dolci_instruct_filtered-DPO_mbt_seed42`

Check the logs for the actual repo name:
```bash
grep "Successfully uploaded" /projects/a5k/public/logs/checkpoint_pipeline/checkpoint_pipeline_JOBID.out
```

---

## GPT-NeoX Training

### Config File Paths

**Config paths must be absolute** when submitting via SLURM because the script changes directories:

```bash
# Wrong - relative path won't work
sbatch pretrain_neox.sbatch neox/configs/experiment/config.yml

# Correct - absolute path
sbatch pretrain_neox.sbatch /home/a5k/kyleobrien.a5k/self-fulfilling-model-organisms/neox/configs/experiment/config.yml
```

### Monitoring Training Progress

```bash
# Check iteration and loss
grep "iteration.*lm_loss" /projects/a5k/public/logs/training/neox-training-JOBID.out

# Check latest iteration
grep "iteration.*/" /projects/a5k/public/logs/training/neox-training-JOBID.out | tail -5

# Check for errors
grep -i "error\|exception\|traceback" /projects/a5k/public/logs/training/neox-training-JOBID.out
```

### Checkpoint Discovery

```bash
# List checkpoints for an experiment
ls -d /projects/a5k/public/checkpoints/sf_model_organisms/EXPERIMENT_NAME/global_step*

# Count checkpoints
ls -d /projects/a5k/public/checkpoints/sf_model_organisms/EXPERIMENT_NAME/global_step* | wc -l
```

### Common Training Config Issues

1. **Wrong data path**: Ensure `data-path` in config points to existing tokenized data
2. **Checkpoint directory**: Verify `save` path exists and is writable
3. **Iteration counts**: `train-iters` should match your training plan
4. **Seed consistency**: Use explicit seeds for reproducibility (`seed: 42` or `seed: 206`)

---

## Environment Setup

### Required Activation Sequence

For SLURM multi-node jobs (uses system NCCL for Slingshot):

```bash
# 1. Activate UV virtual environment
source /home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv/bin/activate

# 2. Load modules
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/nccl/2.26.6-1

# 3. Prefer system NCCL for Slingshot/OFI support
if [[ -n "${NCCL_ROOT:-}" && -f "${NCCL_ROOT}/lib/libnccl.so" ]]; then
  export LD_PRELOAD="${NCCL_ROOT}/lib/libnccl.so:${LD_PRELOAD-}"
fi

# 4. Set environment variables
export PYTHONPATH=/projects/a5k/public/self-fulfilling-model-organisms:$PYTHONPATH
export TMPDIR=/projects/a5k/public/tmp
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
```

For single-node/interactive use (uses venv NCCL):

```bash
source /home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv/bin/activate
export NCCL_LIBRARY=.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$NCCL_LIBRARY"
```

### TMPDIR Issues

Some nodes don't have `/local/user` directories. Always set TMPDIR:

```bash
export TMPDIR=/projects/a5k/public/tmp
mkdir -p $TMPDIR
```

---

## Debugging Patterns

### Checking Job Completion Status

```bash
# Quick success check for a job
grep "exit code 0" /projects/a5k/public/logs/checkpoint_pipeline/checkpoint_pipeline_JOBID.out

# Get exit code from any job
grep "exit code" /projects/a5k/public/logs/checkpoint_pipeline/checkpoint_pipeline_JOBID.out

# Check if job is still in progress (log exists but no completion message)
if [ -f "logfile.out" ] && ! grep -q "Pipeline completed" logfile.out; then
    echo "Job still running or crashed"
fi
```

### Batch Status Checking

```bash
# Check multiple jobs at once
for job in 123456 123457 123458; do
    echo "=== Job $job ==="
    tail -5 /projects/a5k/public/logs/checkpoint_pipeline/checkpoint_pipeline_$job.out
done

# Find which checkpoint a job was processing
head -50 /projects/a5k/public/logs/checkpoint_pipeline/checkpoint_pipeline_JOBID.out | grep -E "(Checkpoint:|Processing checkpoint)"
```

### Log Locations

| Log Type | Path |
|----------|------|
| Checkpoint pipeline | `/projects/a5k/public/logs/checkpoint_pipeline/checkpoint_pipeline_*.out` |
| Training | `/projects/a5k/public/logs/training/neox-training-*.out` |
| Filtering | `/projects/a5k/public/logs/alignment-filtering/alignment-filtering-*.out` |
| Inspect evals | `/projects/a5k/public/logs/inspect-evals/inspect-eval-*.out` |

---

## Common Errors and Fixes

### "Checkpoint not found"

**Error**: `FileNotFoundError: Checkpoint N for experiment X not found`

**Causes**:
1. Wrong `CHECKPOINTS_BASE_DIR` - should include `sf_model_organisms/`
2. Checkpoint doesn't exist (training didn't reach that iteration)
3. Typo in experiment name

**Fix**: Verify the full path exists:
```bash
ls /projects/a5k/public/checkpoints/sf_model_organisms/EXPERIMENT_NAME/global_step*/
```

### "Repository already exists"

**Message**: `Repository geodesic-research/model-name already exists`

**This is usually fine** - the script has `--skip-if-exists` logic. It will upload new revisions (branches) to the existing repo.

### "429 Too Many Requests"

See [HuggingFace Upload Issues](#huggingface-upload-issues) above.

### NCCL Errors

**Error**: Various NCCL communication errors during distributed training

**Check**:
```bash
# Verify NCCL module loaded
module list 2>&1 | grep nccl

# Check NCCL environment
env | grep NCCL
```

**Common NCCL settings for Isambard**:
```bash
export NCCL_CROSS_NIC=0
export NCCL_NET=AWS Libfabric
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
```

### Python Import Hangs

If Python hangs during imports (especially torch), check:
1. CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
2. Module conflicts: Try `module purge` then reload required modules
3. NCCL conflicts: Ensure correct `LD_PRELOAD` is set for your use case
4. Environment corruption: Recreate the UV venv using `bash setup_uv_env.sh`

---

## Useful One-Liners

```bash
# Kill all your pending jobs
squeue -u $USER | grep " PD " | awk '{print $1}' | xargs scancel

# Find experiments with most checkpoints
for exp in /projects/a5k/public/checkpoints/sf_model_organisms/*/; do
    echo "$(ls -d $exp/global_step* 2>/dev/null | wc -l) $(basename $exp)"
done | sort -rn | head -10

# Check disk usage of checkpoints
du -sh /projects/a5k/public/checkpoints/sf_model_organisms/*/

# Find running training jobs
squeue -u $USER | grep -E "(neox|training|pretrain)"

# Quick cluster utilization check
squeue | wc -l
```

---

## Links and References

- [GPT-NeoX Documentation](https://github.com/EleutherAI/gpt-neox)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/huggingface_hub)
- [Isambard User Guide](https://docs.isambard.ac.uk/)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)

---

## Adding to This Document

When you encounter a new pitfall or learn something useful:

1. Add it to the appropriate section
2. Include the error message or symptom
3. Provide the solution or workaround
4. Add the date if the information is time-sensitive

Format for new entries:
```markdown
### Issue Title (YYYY-MM)

**Error/Symptom**: What you see

**Cause**: Why it happens

**Fix**: How to resolve it
```
