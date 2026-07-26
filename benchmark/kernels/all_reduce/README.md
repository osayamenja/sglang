## MSCCL++ Collectives

[MSCCL++](https://github.com/microsoft/mscclpp) is a GPU-driven communication library that can replace NCCL for selected all-reduce and all-gather operations. It supports CUDA graph capture and is optimized for small-to-medium message sizes commonly seen in tensor-parallel inference.

Native all-reduce and equal-size all-gather are supported for single-node
communicator sizes **2**, **4**, and **8**. Multi-node all-reduce remains
supported for communicator sizes **16** and **32**. Unsupported operations,
layouts, dtypes, sizes, and topologies fall back to the existing backend.

### Prerequisites

1. If you use the default SGLang Docker image build from `docker/Dockerfile`, [MSCCL++](https://github.com/microsoft/mscclpp) is already installed by default.
2. If you are not using that Docker image (or want to install manually), install the pinned [MSCCL++](https://github.com/microsoft/mscclpp) source (requires CMake and a CUDA toolkit):
    ```bash
    git clone --depth 1 --branch sglang-v0.9.1 \
        https://github.com/microsoft/mscclpp.git
    pip install "./mscclpp[cuda12]"  # Use cuda13 for a CUDA 13 toolkit.
    ```
3. Ensure `mscclpp` is importable in your Python environment before running the benchmark or using MSCCL++ for inference.

### Running the Benchmark

The benchmark compares all-reduce latency across torch/NCCL (eager), MSCCL++ (eager and graph), and PyNccl (graph) for power-of-two message sizes.

```bash
torchrun --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    benchmark/kernels/all_reduce/benchmark_mscclpp.py
```

For multi-node (TP=16):
```bash
export WORLD_SIZE=2
export MASTER_ADDR=<master-ip>
export MASTER_PORT=12345

# Run on each node with the appropriate RANK (0 or 1):
torchrun --nproc_per_node 8 \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    benchmark/kernels/all_reduce/benchmark_mscclpp.py
```

### Inference with MSCCL++

Use the `--enable-mscclpp` flag to select MSCCL++ for eligible all-reduce and
all-gather calls during CUDA-graph-captured inference:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp-size 8 \
    --enable-mscclpp
```

> **Note:** MSCCL++ performs auto-tuning on first initialization, which may add a few seconds to startup time. The tuned configurations are cached for the lifetime of the process.

To smoke-test both collectives:

```bash
torchrun --standalone --nproc-per-node=2 \
    test/manual/distributed/test_mscclpp.py
```
