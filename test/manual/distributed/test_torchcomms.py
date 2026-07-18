"""Two-GPU smoke test for SGLang's torchcomms integration.

Run with:
    torchrun --standalone --nproc-per-node=2 \
        test/manual/distributed/test_torchcomms.py
"""

import os

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    init_distributed_environment,
    init_model_parallel_group,
    set_custom_all_reduce,
    set_torchcomms,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "This smoke test requires exactly two GPUs."

    torch.cuda.set_device(local_rank)
    set_custom_all_reduce(False)
    set_torchcomms(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    group = init_model_parallel_group(
        group_ranks=[list(range(world_size))],
        local_rank=local_rank,
        backend="nccl",
        use_pynccl=True,
        use_custom_allreduce=False,
        group_name="torchcomms_smoke",
    )
    assert group.torchcomms_comm is not None

    actual = torch.tensor([rank + 1.0], device="cuda")
    expected = actual.clone()
    dist.all_reduce(expected, group=group.device_group)
    group.all_reduce(actual)
    torch.testing.assert_close(actual, expected)

    reduce_scatter_input = torch.arange(4, dtype=torch.float32, device="cuda") + rank
    reduced = reduce_scatter_input.clone()
    dist.all_reduce(reduced, group=group.device_group)
    expected = reduced.chunk(world_size)[rank]
    actual = torch.empty_like(expected)
    group.reduce_scatter_tensor(actual, reduce_scatter_input)
    torch.testing.assert_close(actual, expected)

    gather_input = torch.tensor([rank * 2.0, rank * 2.0 + 1], device="cuda")
    expected = torch.empty(4, device="cuda")
    dist.all_gather_into_tensor(expected, gather_input, group=group.device_group)
    actual = torch.empty_like(expected)
    group.all_gather_into_tensor(actual, gather_input)
    torch.testing.assert_close(actual, expected)

    all_to_all_input = torch.arange(4, dtype=torch.float32, device="cuda") + 10 * rank
    expected = torch.empty_like(all_to_all_input)
    dist.all_to_all_single(expected, all_to_all_input, group=group.device_group)
    actual = torch.empty_like(expected)
    group.all_to_all_single(actual, all_to_all_input)
    torch.testing.assert_close(actual, expected)

    sizes = [1, 2]
    reduce_scatter_v_input = (
        torch.arange(sum(sizes), dtype=torch.float32, device="cuda") + rank
    )
    reduced = reduce_scatter_v_input.clone()
    dist.all_reduce(reduced, group=group.device_group)
    offset = sum(sizes[:rank])
    expected = reduced.narrow(0, offset, sizes[rank])
    actual = group.reduce_scatterv(reduce_scatter_v_input, sizes=sizes)
    torch.testing.assert_close(actual, expected)

    gather_v_input = torch.arange(sizes[rank], dtype=torch.float32, device="cuda")
    gather_v_input += 10 * rank
    actual = group.all_gatherv(gather_v_input, sizes=sizes)[0]
    expected = torch.tensor([0.0, 10.0, 11.0], device="cuda")
    torch.testing.assert_close(actual, expected)

    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print("SGLang torchcomms NCCLX collective smoke test passed")

    group.destroy()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
