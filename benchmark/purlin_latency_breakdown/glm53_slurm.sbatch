#!/bin/bash
#SBATCH --job-name=glm53-breakdown
#SBATCH --partition=slinky
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64

# Submit from the root of the checkout.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export NCCL_SOCKET_IFNAME=eth0 NCCL_NET_PLUGIN=none

for concurrency in 1 16; do
    prompts=$((8 * concurrency))
    warmups=8
    if (( concurrency == 16 )); then warmups=16; fi
    python benchmark/purlin_latency_breakdown/run_suite.py \
        --model zai-org/GLM-5.3 --variants baseline \
        --nnodes 2 --tp 16 --ep 16 --dp 1 \
        --use-server-default-attention-backend --no-enable-dp-attention \
        --no-enable-prefill-delayer --no-trust-remote-code \
        --mem-fraction-static 0.85 --server-arg=--max-running-requests=256 \
        --chunked-prefill-size 8192 --server-arg=--cuda-graph-max-bs-prefill=2048 \
        --no-require-prefill-cuda-graph --cuda-graph-max-bs-decode 256 \
        --concurrency "$concurrency" --input-len 8192 --output-len 1024 \
        --num-prompts "$prompts" --trace-prompts "$prompts" --warmup-requests "$warmups" \
        --overlap-tolerance-ns 150000 \
        --nsys-command "${NSYS_COMMAND:-nsys}" \
        --output-dir "benchmark_results/glm53_${SLURM_JOB_ID}/c${concurrency}"
done
