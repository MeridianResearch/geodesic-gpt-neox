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

source /projects/a5k/public/data/python_envs/sfm/.venv/bin/activate
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

lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,tensor_parallel_size=${TP_SIZE},gpu_memory_utilization=0.9" \
    --tasks mmlu_abstract_algebra,mmlu_anatomy,mmlu_astronomy,mmlu_business_ethics,mmlu_college_biology,mmlu_college_chemistry,mmlu_college_computer_science,mmlu_college_mathematics,mmlu_college_medicine,mmlu_college_physics \
    --batch_size auto \
    --num_fewshot 5

echo "Exit code: $?"
