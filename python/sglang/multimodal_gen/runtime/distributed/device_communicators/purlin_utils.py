# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.distributed import ProcessGroup

PURLIN_SPLIT_ALIGNMENT = 16
_PURLIN_REDUCE_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
if hasattr(torch, "float8_e5m2"):
    _PURLIN_REDUCE_DTYPES.add(torch.float8_e5m2)
if hasattr(torch, "float8_e4m3fn"):
    _PURLIN_REDUCE_DTYPES.add(torch.float8_e4m3fn)


def is_purlin_supported_device(device: torch.device) -> bool:
    return device.type == "cuda" and torch.version.hip is None


def current_stream(device: torch.device) -> torch.cuda.Stream:
    return torch.cuda.current_stream(device)


def prepare_stream(
    device: torch.device, stream: Optional[torch.cuda.Stream] = None
) -> torch.cuda.Stream:
    current = current_stream(device)
    if stream is None:
        return current
    if stream != current:
        stream.wait_stream(current)
    return stream


def initialize_purlin_handle(
    group: ProcessGroup,
    device: torch.device,
    stream: Optional[torch.cuda.Stream] = None,
) -> Any:
    import purlin

    stream = prepare_stream(device, stream)
    major, minor = torch.cuda.get_device_capability(device)
    return purlin.initialize(
        group=group,
        device=device,
        arch=major * 10 + minor,
        stream_ptr=stream.cuda_stream,
    )


def finalize_purlin_handle(
    handle: Any,
    device: torch.device,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    import purlin

    stream = prepare_stream(device, stream)
    purlin.finalize(handle, stream.cuda_stream)


def sizes_in_bytes(sizes: list[int], tensor: torch.Tensor) -> list[int]:
    row_size_in_bytes = tensor.element_size()
    for dim_size in tensor.shape[1:]:
        row_size_in_bytes *= dim_size
    return [size * row_size_in_bytes for size in sizes]


def element_counts_to_bytes(sizes: list[int], tensor: torch.Tensor) -> list[int]:
    return [size * tensor.element_size() for size in sizes]


def can_use_purlin(
    handle: Any,
    input_: torch.Tensor,
    output: torch.Tensor,
    *,
    require_reduce_dtype: bool = False,
) -> bool:
    if handle is None:
        return False
    if input_.is_cpu or output.is_cpu:
        return False
    if input_.device != output.device:
        return False
    if not is_purlin_supported_device(input_.device):
        return False
    if not input_.is_contiguous() or not output.is_contiguous():
        return False
    if require_reduce_dtype and input_.dtype not in _PURLIN_REDUCE_DTYPES:
        return False
    return True


def are_aligned_byte_sizes(sizes: list[int]) -> bool:
    return all(size % PURLIN_SPLIT_ALIGNMENT == 0 for size in sizes)


class PurlinWork:
    def __init__(self, event: torch.cuda.Event, device: torch.device):
        self._event = event
        self._device = device

    def wait(self, timeout: Optional[float] = None) -> bool:
        if timeout is not None:
            raise NotImplementedError("PurlinWork.wait does not support timeout.")
        torch.cuda.current_stream(self._device).wait_event(self._event)
        return True

    def is_completed(self) -> bool:
        return self._event.query()


def _record_work(
    stream: torch.cuda.Stream,
    device: torch.device,
    async_op: bool,
) -> Optional[PurlinWork]:
    if not async_op:
        return None
    event = torch.cuda.Event()
    event.record(stream)
    return PurlinWork(event, device)


def all_to_all(
    input_: torch.Tensor,
    output: torch.Tensor,
    handle: Any,
    stream: Optional[torch.cuda.Stream] = None,
    async_op: bool = False,
) -> Optional[PurlinWork]:
    import purlin

    stream = prepare_stream(input_.device, stream)
    purlin.all_to_all(input_, output, handle, stream.cuda_stream)
    return _record_work(stream, input_.device, async_op)


def all_to_all_v(
    input_: torch.Tensor,
    output: torch.Tensor,
    input_split_sizes: list[int],
    output_split_sizes: list[int],
    handle: Any,
    stream: Optional[torch.cuda.Stream] = None,
    async_op: bool = False,
) -> Optional[PurlinWork]:
    import purlin

    input_split_bytes = element_counts_to_bytes(input_split_sizes, input_)
    output_split_bytes = element_counts_to_bytes(output_split_sizes, input_)
    splits = input_split_bytes + output_split_bytes
    stream = prepare_stream(input_.device, stream)
    purlin.all_to_all_v(input_, output, splits, handle, stream.cuda_stream)
    return _record_work(stream, input_.device, async_op)
