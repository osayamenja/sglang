# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

TORCHCOMMS_BACKEND = "ncclx"

_torchcomms_module: Optional[Any] = None
_root_comm: Optional[Any] = None
_root_device: Optional[torch.device] = None


def is_torchcomms_supported_device(device: torch.device) -> bool:
    return device.type == "cuda" and torch.version.hip is None


def _torchcomms_install_error(error: BaseException) -> ImportError:
    install_error = ImportError(
        "--enable-torchcomms requires TorchComms built with the NCCLX backend "
        "against this PyTorch/CUDA environment. Run scripts/install_purlin.sh "
        "from this checkout to install the pinned source revision."
    )
    install_error.__cause__ = error
    return install_error


def _load_torchcomms() -> Any:
    global _torchcomms_module
    if _torchcomms_module is not None:
        return _torchcomms_module

    try:
        import torchcomms
    except (ImportError, OSError) as error:
        raise _torchcomms_install_error(error)

    if hasattr(torchcomms, "is_backend_built") and not torchcomms.is_backend_built(
        TORCHCOMMS_BACKEND
    ):
        raise RuntimeError(
            "--enable-torchcomms requires TorchComms built with the "
            f"{TORCHCOMMS_BACKEND!r} backend."
        )

    _torchcomms_module = torchcomms
    return torchcomms


@contextmanager
def _torchcomms_world_env(rank: int, world_size: int) -> Iterator[None]:
    updates = {
        "TORCHCOMM_RANK": str(rank),
        "TORCHCOMM_SIZE": str(world_size),
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _get_root_comm(device: torch.device) -> tuple[Any, Any]:
    global _root_comm, _root_device

    torchcomms = _load_torchcomms()
    if _root_comm is not None:
        if _root_device != device:
            raise RuntimeError(
                "The torchcomms root communicator was initialized for "
                f"{_root_device}, but a group requested {device}."
            )
        return torchcomms, _root_comm

    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before torchcomms is enabled."
        )
    if not is_torchcomms_supported_device(device):
        raise RuntimeError(
            "--enable-torchcomms currently supports NVIDIA CUDA devices only."
        )

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    store = torch.distributed.distributed_c10d._get_default_store()
    try:
        with _torchcomms_world_env(rank, world_size):
            _root_comm = torchcomms.new_comm(
                TORCHCOMMS_BACKEND,
                device,
                store=store,
                name="sglang_world",
            )
    except (ImportError, OSError) as error:
        raise _torchcomms_install_error(error)

    _root_device = device
    logger.info(
        "Initialized torchcomms backend=%s root rank=%d world_size=%d",
        TORCHCOMMS_BACKEND,
        rank,
        world_size,
    )
    return torchcomms, _root_comm


class TorchCommsCommunicator:
    """Small NCCLX-only adapter around a torchcomms subgroup."""

    def __init__(self, ranks: Sequence[int], device: torch.device, name: str):
        self.device = device
        self.name = name
        self._torchcomms, root_comm = _get_root_comm(device)
        self._comm = root_comm.split(list(ranks), name=f"sglang_{name}")
        if self._comm is None:
            raise RuntimeError(
                f"torchcomms split for group {name!r} excluded the current rank."
            )
        logger.info(
            "Initialized torchcomms backend=%s group=%s rank=%d world_size=%d",
            TORCHCOMMS_BACKEND,
            name,
            self._comm.get_rank(),
            self._comm.get_size(),
        )

    @property
    def enabled(self) -> bool:
        return self._comm is not None

    def can_use(
        self,
        *tensors: torch.Tensor,
        require_reduce_dtype: bool = False,
    ) -> bool:
        if self._comm is None or not tensors:
            return False
        if not isinstance(tensors[0], torch.Tensor):
            return False
        dtype = tensors[0].dtype
        for tensor in tensors:
            if (
                not isinstance(tensor, torch.Tensor)
                or not tensor.is_cuda
                or not tensor.is_contiguous()
                or tensor.device != self.device
                or tensor.dtype != dtype
            ):
                return False
        if require_reduce_dtype and not (dtype.is_floating_point or dtype.is_complex):
            return False
        return True

    def _on_stream(self, operation, stream: Optional[torch.cuda.Stream]):
        if stream is None:
            return operation()
        current_stream = torch.cuda.current_stream(self.device)
        if stream != current_stream:
            stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            return operation()

    def all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        async_op: bool = False,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Any:
        return self._on_stream(
            lambda: self._comm.all_reduce(
                tensor, self._torchcomms.ReduceOp.SUM, async_op
            ),
            stream,
        )

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        sizes: Optional[Sequence[int]] = None,
    ) -> Any:
        if sizes is None or all(size == sizes[0] for size in sizes):
            return self._comm.reduce_scatter_single(
                output,
                input_,
                self._torchcomms.ReduceOp.SUM,
                False,
            )
        input_list = list(input_.split(list(sizes), dim=0))
        return self._comm.reduce_scatter_v(
            output,
            input_list,
            self._torchcomms.ReduceOp.SUM,
            False,
        )

    def all_gather(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        sizes: Optional[Sequence[int]] = None,
    ) -> Any:
        if sizes is None or all(size == sizes[0] for size in sizes):
            return self._comm.all_gather_single(output, input_, False)
        output_list = list(output.split(list(sizes), dim=0))
        return self._comm.all_gather_v(output_list, input_, False)

    def all_to_all(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        *,
        output_split_sizes: Optional[Sequence[int]] = None,
        input_split_sizes: Optional[Sequence[int]] = None,
        async_op: bool = False,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Any:
        if output_split_sizes is None and input_split_sizes is None:
            return self._on_stream(
                lambda: self._comm.all_to_all_single(output, input_, async_op),
                stream,
            )
        assert output_split_sizes is not None and input_split_sizes is not None
        return self._on_stream(
            lambda: self._comm.all_to_all_v_single(
                output,
                input_,
                list(output_split_sizes),
                list(input_split_sizes),
                async_op,
            ),
            stream,
        )

    def finalize(self) -> None:
        if self._comm is not None:
            self._comm.finalize()
            self._comm = None


def finalize_torchcomms() -> None:
    global _root_comm, _root_device
    if _root_comm is not None:
        _root_comm.finalize()
        _root_comm = None
        _root_device = None
