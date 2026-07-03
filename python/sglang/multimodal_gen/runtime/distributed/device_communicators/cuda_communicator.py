# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/device_communicators/cuda_communicator.py

import torch
from torch.distributed import ProcessGroup

from sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator import (
    DistributedAutograd,
    DeviceCommunicatorBase,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.purlin_utils import (
    all_to_all,
    all_to_all_v as purlin_all_to_all_v,
    are_aligned_byte_sizes,
    can_use_purlin,
    element_counts_to_bytes,
    finalize_purlin_handle,
    initialize_purlin_handle,
    is_purlin_supported_device,
)


class CudaCommunicator(DeviceCommunicatorBase):

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        enable_purlin: bool = False,
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        self.enable_purlin = enable_purlin
        self.purlin_handle = None

        from sglang.multimodal_gen.runtime.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        self.pynccl_comm: PyNcclCommunicator | None = None
        if self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )
            if self.enable_purlin and is_purlin_supported_device(self.device):
                self.purlin_handle = initialize_purlin_handle(
                    group=self.device_group,
                    device=self.device,
                )

    def all_reduce(self, input_, op: torch.distributed.ReduceOp | None = None):
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        out = pynccl_comm.all_reduce(input_, op=op)
        if out is None:
            # fall back to the default all-reduce using PyTorch.
            # this usually happens during testing.
            # when we run the model, allreduce only happens for the TP
            # group, where we always have either custom allreduce or pynccl.
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group, op=op)
        return out

    def all_to_all_single(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        async_op: bool = False,
        stream: torch.cuda.Stream | None = None,
    ):
        if (
            can_use_purlin(self.purlin_handle, input_, output)
            and input_.nbytes == output.nbytes
            and input_.nbytes % self.world_size == 0
        ):
            return all_to_all(
                input_,
                output,
                self.purlin_handle,
                stream=stream,
                async_op=async_op,
            )
        return torch.distributed.all_to_all_single(
            output, input_, group=self.device_group, async_op=async_op
        )

    def all_to_all_v(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        async_op: bool = False,
        stream: torch.cuda.Stream | None = None,
    ):
        input_split_bytes = element_counts_to_bytes(input_split_sizes, input_)
        output_split_bytes = element_counts_to_bytes(output_split_sizes, input_)
        if (
            can_use_purlin(self.purlin_handle, input_, output)
            and len(input_split_sizes) == self.world_size
            and len(output_split_sizes) == self.world_size
            and input_.nbytes == sum(input_split_bytes)
            and output.nbytes == sum(output_split_bytes)
            and are_aligned_byte_sizes(input_split_bytes + output_split_bytes)
        ):
            return purlin_all_to_all_v(
                input_,
                output,
                input_split_sizes,
                output_split_sizes,
                self.purlin_handle,
                stream=stream,
                async_op=async_op,
            )
        return torch.distributed.all_to_all_single(
            output,
            input_,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self.device_group,
            async_op=async_op,
        )

    def all_to_all_4D(
        self, input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1
    ) -> torch.Tensor:
        if self.purlin_handle is None:
            return super().all_to_all_4D(input_, scatter_dim, gather_dim)
        return DistributedAutograd.AllToAll4D.apply(
            self.device_group,
            input_,
            self.world_size,
            scatter_dim,
            gather_dim,
            self.all_to_all_single,
        )

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self) -> None:
        if self.purlin_handle is not None:
            finalize_purlin_handle(self.purlin_handle, self.device)
            self.purlin_handle = None
        if self.pynccl_comm is not None:
            self.pynccl_comm = None
