#!/bin/bash
# Run MMLU eval for Nemotron models using sfm-evals vLLM environment
# Usage: isambard_sbatch [--gpus=N] tools/run_nemotron_mmlu_eval.sh MODEL [TP_SIZE]
#SBATCH --job-name=nemotron-mmlu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --output=/projects/a5k/public/logs/sfm-evals/nemotron-mmlu-%j.out

# Use sfm venv for vLLM (single GPU), NeoX venv for HF parallelize (multi-GPU)
TP_SIZE_CHECK=${2:-1}
if [ "$TP_SIZE_CHECK" -gt 1 ]; then
    source /home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv/bin/activate
    export NCCL_LIBRARY=/home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
    export LD_PRELOAD=$NCCL_LIBRARY
else
    source /projects/a5k/public/data/python_envs/sfm/.venv/bin/activate
    export LD_PRELOAD=/projects/a5k/public/data/python_envs/sfm/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}
fi
module purge
module load PrgEnv-cray
module load cuda/12.6
export LD_PRELOAD=/projects/a5k/public/data/python_envs/sfm/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"
export TMPDIR=/projects/a5k/public/tmp
export HF_HUB_OFFLINE=0

MODEL=$1
TP_SIZE=${2:-1}

echo "Model: $MODEL"
echo "TP size: $TP_SIZE"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"

if [ "$TP_SIZE" -gt 1 ]; then
    # For large models, use HF with parallelize (auto device_map)
    echo "Using HF parallelize mode (TP=$TP_SIZE GPUs via device_map=auto)"
    lm_eval \
        --model hf \
        --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,parallelize=True" \
        --tasks ${TASKS:-mmlu_abstract_algebra,mmlu_college_biology} \
        --batch_size 1 \
        --num_fewshot 5
else
    # For single GPU, use vLLM (fast)
    echo "Using vLLM mode (TP=1)"
    lm_eval \
        --model vllm \
        --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096" \
        --tasks mmlu_abstract_algebra,mmlu_anatomy,mmlu_astronomy,mmlu_business_ethics,mmlu_college_biology,mmlu_college_chemistry,mmlu_college_computer_science,mmlu_college_mathematics,mmlu_college_medicine,mmlu_college_physics \
        --batch_size auto \
        --num_fewshot 5
fi

echo "Exit code: $?"
