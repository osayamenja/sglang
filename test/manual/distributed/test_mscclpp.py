"""Two-collective MSCCL++ smoke test.

Run on one node with 2, 4, or 8 GPUs:

    torchrun --standalone --nproc-per-node=2 \
        test/manual/distributed/test_mscclpp.py
"""

import os

import torch
import torch.distributed as dist

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
    set_mscclpp_all_reduce,
)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (2, 4, 8):
        raise ValueError("This test requires a single-node world size of 2, 4, or 8")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    set_mscclpp_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    tp_group = get_tensor_model_parallel_group()
    mscclpp_comm = tp_group.pymscclpp_comm
    assert mscclpp_comm is not None and mscclpp_comm.initialized

    numel = 1024
    allreduce_source = torch.full(
        (numel,), rank + 1, dtype=torch.bfloat16, device="cuda"
    )
    allreduce_input = torch.empty_like(allreduce_source)
    allgather_input = torch.full((numel,), rank, dtype=torch.bfloat16, device="cuda")
    allgather_output = torch.empty(
        numel * world_size, dtype=torch.bfloat16, device="cuda"
    )

    with graph_capture() as capture_context:
        assert mscclpp_comm.should_mscclpp_allreduce(allreduce_input)
        assert mscclpp_comm.should_mscclpp_allgather(allgather_output, allgather_input)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_context.stream):
            allreduce_input.copy_(allreduce_source)
            allreduce_output = tp_group.all_reduce(allreduce_input)
            tp_group.all_gather_into_tensor(allgather_output, allgather_input)

    graph.replay()
    torch.cuda.synchronize()

    expected_sum = world_size * (world_size + 1) // 2
    torch.testing.assert_close(
        allreduce_output,
        torch.full_like(allreduce_output, expected_sum),
    )
    torch.testing.assert_close(
        allgather_output,
        torch.arange(world_size, device="cuda", dtype=torch.bfloat16).repeat_interleave(
            numel
        ),
    )

    if rank == 0:
        print(
            f"MSCCL++ all-reduce/all-gather CUDA graph smoke test passed "
            f"for communicator size {world_size}"
        )

    graph.reset()
    dist.barrier()
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
