import importlib
import logging
from contextlib import contextmanager
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sglang.srt.compilation.compile_phase import (
    get_pcg_capture_stream,
    is_in_torch_compile_warmup,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_exec

logger = logging.getLogger(__name__)


class PyMscclppCommunicator:
    _NATIVE_WORLD_SIZES = [2, 4, 8]
    _DSL_ALLREDUCE_WORLD_SIZES = [16, 32]
    _SUPPORTED_WORLD_SIZES = _NATIVE_WORLD_SIZES + _DSL_ALLREDUCE_WORLD_SIZES
    _SUPPORTED_ALLGATHER_WORLD_SIZES = _NATIVE_WORLD_SIZES
    _SUPPORTED_DTYPE = [torch.float, torch.float16, torch.bfloat16]

    def _is_symm_mem_enabled(self) -> bool:
        try:
            return get_exec().comm.enable_symm_mem
        except ValueError:
            return False

    def _is_weak_contiguous(self, inp: torch.Tensor):
        return inp.is_contiguous() or (
            inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
            == inp.numel() * inp.element_size()
        )

    def _get_tuned_config(self, size):
        if size <= 512:
            target_size = 512
        elif size > 256 * 1024 * 1024:
            target_size = 256 * 1024 * 1024
        else:
            target_size = 1 << (size - 1).bit_length()
        return self.best_configs.get(target_size)

    def _is_tuning_candidate_supported(self, algo, nblocks, nthreads):
        # On A100 (SM80), MSCCL++ 0.9.1's 8-rank RSAG zero-copy kernel spins
        # indefinitely at 128 blocks and 768 threads. Exclude both high-thread
        # variants because the in-process tuner cannot recover to test 1024
        # after a hung kernel. H100 keeps the full search space.
        capability = torch.cuda.get_device_capability(self.device)
        return not (
            capability == (8, 0)
            and self.world_size == 8
            and algo.name == "default_allreduce_rsag_zero_copy"
            and nblocks == 128
            and nthreads >= 768
        )

    def _create_dsl_algorithms(self):
        dsl_algos_config = []
        n_nodes = self.world_size // self.nranks_per_node
        if n_nodes == 2 or n_nodes == 4:
            for tbg in [1, 2, 4, 8]:
                for num_threads_per_block in [256, 512, 768, 1024]:
                    spec = self.mscclpp.language.AlgoSpec(
                        name=f"allreduce_{n_nodes}node_{tbg}TBG_{num_threads_per_block}TPB",
                        collective=self.mscclpp.language.collectives.AllReduce(
                            self.world_size, 1, True
                        ),
                        nranks_per_node=self.nranks_per_node,
                        world_size=self.world_size,
                        in_place=True,
                        instances=1,
                        protocol="LL",
                        auto_sync=False,
                        num_threads_per_block=num_threads_per_block,
                        reuse_resources=True,
                        use_double_scratch_buffer=True,
                        min_message_size=tbg * (1 << 10),
                        max_message_size=8 << 20,
                        tags={"default": 1},
                    )
                    algo = self.mscclpp.compile(
                        self.def_algo.allreduce_multi_nodes,
                        spec,
                        self.rank,
                        thread_block_group_size=tbg,
                    )
                    dsl_algos_config.append((algo, [0], [0]))
        return dsl_algos_config

    def _create_native_algorithms(self):
        native_allreduce_algorithms_config = []
        dlpack = self.mscclpp.RawGpuBuffer(1 << 27).to_dlpack(
            data_type=str(torch.float16)
        )
        self.scratch_buffer = torch.utils.dlpack.from_dlpack(dlpack)
        self.flag_buffer = torch.ones(128, dtype=torch.uint32, device="cuda")
        algos = self.mscclpp_ext.AlgorithmCollectionBuilder().build_default_algorithms(
            scratch_buffer=self.scratch_buffer.data_ptr(),
            scratch_buffer_size=self.scratch_buffer.nbytes,
            rank=self.rank,
        )

        for algo in algos:
            if self.nvls_supported and algo.name == "default_allreduce_nvls_packet":
                algo.set_message_size_range(0, 512 << 10)
                native_allreduce_algorithms_config.append(
                    (algo, [4, 8, 12, 16], [256, 512, 768, 1024])
                )
            if algo.name == "default_allreduce_packet":
                algo.set_message_size_range(0, 2 << 20)
                native_allreduce_algorithms_config.append(
                    (algo, [14, 21, 28, 42, 56], [256, 512, 768, 1024])
                )
            if (
                self.nranks_per_node in (4, 8)
                and algo.name == "default_allreduce_rsag_zero_copy"
            ):
                algo.set_message_size_range(512 << 10, 4 << 30)
                native_allreduce_algorithms_config.append(
                    (algo, [32, 48, 64, 128], [256, 512, 768, 1024])
                )
            if (
                self.nvls_supported
                and self.symm_mem_enabled
                and algo.name == "default_allreduce_nvls_zero_copy"
            ):
                algo.set_message_size_range(512 << 10, 4 << 30)
                native_allreduce_algorithms_config.append(
                    (algo, [4, 8, 12, 16, 32], [256, 512, 768, 1024])
                )
            if algo.name == "default_allgather_fullmesh2":
                # Let the native implementation choose its launch dimensions.
                # It communicates 32-bit chunks, so eligibility below enforces
                # that the per-rank input size is a multiple of four bytes.
                self.allgather_config = (algo, 0, 0)

        return native_allreduce_algorithms_config

    def _create_algorithms(self):
        if self.world_size in PyMscclppCommunicator._NATIVE_WORLD_SIZES:
            self.algos_config = self._create_native_algorithms()
            self._tune(5, 10, 20, self.algos_config)
        elif self.world_size in PyMscclppCommunicator._DSL_ALLREDUCE_WORLD_SIZES:
            self.dsl_algos_config = self._create_dsl_algorithms()
            self._tune(5, 10, 20, self.dsl_algos_config)

    def _get_time(
        self,
        algo,
        tune_tensor,
        size,
        nb,
        nt,
        n_warmup,
        n_graph_launches,
        n_ops_per_graph,
    ):
        # Check if the algorithm can run with the given configuration
        if self._run_algo(algo, tune_tensor, size, nb, nt, True) != 0:
            return float("inf")

        # Warmup iterations to stabilize performance
        for _ in range(n_warmup):
            self._run_algo(algo, tune_tensor, size, nb, nt, True)

        # Warmup on capture stream
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            self._run_algo(algo, tune_tensor, size, nb, nt, True)
        capture_stream.synchronize()

        # Capture the algorithm execution in a CUDA graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture_stream):
            for _ in range(n_ops_per_graph):
                self._run_algo(algo, tune_tensor, size, nb, nt, True)

        # Measure the execution time of the captured graph
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(capture_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(n_graph_launches):
                g.replay()
        end_event.record(capture_stream)
        end_event.synchronize()
        elapsed = start_event.elapsed_time(end_event)

        # Synchronize timing results across all ranks to ensure consistent algorithm selection
        # replicate n times such due to algo limitations
        time_tensor = torch.full(
            (self.world_size,), elapsed, dtype=torch.float64, device="cuda"
        ).to(dtype=torch.float32)
        torch.cuda.current_stream().wait_stream(capture_stream)
        if self.rank == 0:
            avg_time = time_tensor[self.rank].item() / self.world_size
            tensor = torch.tensor([avg_time])
        else:
            tensor = torch.empty(1)
        dist.broadcast(tensor, src=0, group=self.group)
        avg_time = tensor.item()

        return avg_time

    def _tune(self, n_warmup, n_graph_launches, n_ops_per_graph, algos_config):
        sizes = [1 << i for i in range(9, 24)]
        logged_unsupported_candidates = set()
        dlpack = self.mscclpp.RawGpuBuffer(1 << 27).to_dlpack(
            data_type=str(torch.float16)
        )
        tune_tensor = torch.utils.dlpack.from_dlpack(dlpack)

        for size in sizes:
            best_time = float("inf")
            best_config = None
            for i in range(len(algos_config)):
                algo, candidates_nblocks, candidates_nthreads = algos_config[i]
                if (
                    size >= algo.message_size_range[0]
                    and size <= algo.message_size_range[1]
                ):
                    for nb in candidates_nblocks:
                        for nt in candidates_nthreads:
                            if not self._is_tuning_candidate_supported(algo, nb, nt):
                                candidate = (algo.name, nb, nt)
                                if (
                                    self.rank == 0
                                    and candidate not in logged_unsupported_candidates
                                ):
                                    logger.warning(
                                        "Skipping unsupported MSCCL++ tuning "
                                        "candidate on SM80: algo=%s nblocks=%d "
                                        "nthreads=%d",
                                        algo.name,
                                        nb,
                                        nt,
                                    )
                                    logged_unsupported_candidates.add(candidate)
                                continue
                            avg_time = self._get_time(
                                algo,
                                tune_tensor,
                                size,
                                nb,
                                nt,
                                n_warmup,
                                n_graph_launches,
                                n_ops_per_graph,
                            )
                            if avg_time < best_time:
                                best_time = avg_time
                                best_config = (algo, nb, nt)
            if best_config:
                self.best_configs[size] = best_config

        torch.cuda.synchronize()
        for algo, _, _ in algos_config:
            algo.reset()

    def _run_algo(self, algo, tensor, size, nblocks, nthreads, sym_mem_enabled=False):
        return algo.execute(
            comm=self.comm.communicator,
            executor=self.executor,
            input_buffer=tensor.data_ptr(),
            output_buffer=tensor.data_ptr(),
            input_size=size,
            output_size=size,
            dtype=self.dtype_to_mscclpp_dtype(tensor.dtype),
            op=self.mscclpp.ReduceOp.SUM,
            stream=torch.cuda.current_stream().cuda_stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads,
            symmetric_memory=sym_mem_enabled,
        )

    def _run_allgather_algo(
        self,
        algo,
        output: torch.Tensor,
        inp: torch.Tensor,
        nblocks: int,
        nthreads: int,
    ):
        return algo.execute(
            comm=self.comm.communicator,
            executor=self.executor,
            input_buffer=inp.data_ptr(),
            output_buffer=output.data_ptr(),
            input_size=inp.nbytes,
            output_size=output.nbytes,
            dtype=self.dtype_to_mscclpp_dtype(inp.dtype),
            op=self.mscclpp.ReduceOp.NOP,
            stream=torch.cuda.current_stream().cuda_stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads,
            # all_gather_into_tensor outputs are not guaranteed to come from
            # SGLang's symmetric allocator.
            symmetric_memory=False,
        )

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
    ) -> None:
        """Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bound to a unique device. Native communicators (sizes 2, 4, and
        8) must be contained within one node.
        """
        self.disabled = True
        self.available = False
        self.initialized = False
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.ranks = torch.distributed.get_process_group_ranks(group)
        self.device = None
        self.comm = None
        self.executor = None
        self.scratch_buffer = None
        self.flag_buffer = None
        self.algos_config = []
        self.dsl_algos_config = []
        self.allgather_config = None
        self.best_configs = {}

        try:
            self.mscclpp = importlib.import_module("mscclpp")
            self.mscclpp_ext = importlib.import_module("mscclpp.ext")
            self.def_algo = importlib.import_module("mscclpp.default_algos")
        except ImportError:
            self.available = False
            self.mscclpp = None
            return

        self.available = True

        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "PyMscclpp should be attached to a non-NCCL group."

        if self.world_size == 1:
            # No need to initialize mscclpp for single GPU case.
            return

        if self.world_size not in PyMscclppCommunicator._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "PyMscclpp is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_mscclpp=True explicitly.",
                self.world_size,
                str(PyMscclppCommunicator._SUPPORTED_WORLD_SIZES),
            )
            return

        # for now mscclpp with stride in the communicator is not tested
        expected_ranks = list(range(self.ranks[0], self.ranks[0] + self.world_size))
        if self.ranks != expected_ranks:
            logger.warning(
                "PyMscclpp is disabled due to an unsupported group %s."
                "Please ensure all ranks in the group are consecutive."
                "To silence this warning, specify disable_mscclpp=True explicitly.",
                str(self.ranks),
            )
            return

        if self.world_size in PyMscclppCommunicator._NATIVE_WORLD_SIZES:
            from sglang.srt.distributed.parallel_state import in_the_same_node_as

            if not all(in_the_same_node_as(group, source_rank=0)):
                logger.warning(
                    "PyMscclpp is disabled for communicator size %d because "
                    "the process group spans multiple nodes. Native MSCCL++ "
                    "collectives require a single-node group.",
                    self.world_size,
                )
                return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        self.comm = self.mscclpp.CommGroup(
            torch_group=self.group, rank=self.rank, size=self.world_size
        )
        self.nranks_per_node = self.comm.nranks_per_node
        self.executor = self.mscclpp.Executor(self.comm.communicator)
        self.symm_mem_enabled = self._is_symm_mem_enabled()
        self.nvls_supported = self.mscclpp.is_nvls_supported()
        self._create_algorithms()
        self.initialized = True

    def destroy(self):
        self.algos_config = []
        self.dsl_algos_config = []
        self.allgather_config = None
        self.best_configs = {}
        self.executor = None
        self.scratch_buffer = None
        self.flag_buffer = None
        self.comm = None
        self.initialized = False

    def should_mscclpp_allreduce(
        self, inp: torch.Tensor, op: ReduceOp = ReduceOp.SUM
    ) -> bool:
        if (
            self.disabled
            or not self.initialized
            or self.world_size not in PyMscclppCommunicator._SUPPORTED_WORLD_SIZES
        ):
            return False
        if inp.dtype not in PyMscclppCommunicator._SUPPORTED_DTYPE:
            return False
        if inp.device.type != "cuda" or inp.device != self.device:
            return False
        if not self._is_weak_contiguous(inp):
            return False
        if op is not ReduceOp.SUM:
            return False
        if self._get_tuned_config(inp.numel() * inp.element_size()) is None:
            return False
        # mscclpp must not be used during any piecewise CUDA graph phase
        # (compile, capture, or replay) as it changes the allreduce dispatch
        # path and triggers recompilation.
        if (
            is_in_tc_piecewise_cuda_graph()
            or is_in_torch_compile_warmup()
            or get_pcg_capture_stream() is not None
        ):
            return False
        return True

    def should_mscclpp_allgather(self, output: torch.Tensor, inp: torch.Tensor) -> bool:
        if (
            self.disabled
            or not self.initialized
            or self.world_size
            not in PyMscclppCommunicator._SUPPORTED_ALLGATHER_WORLD_SIZES
            or self.allgather_config is None
        ):
            return False
        if inp.dtype not in PyMscclppCommunicator._SUPPORTED_DTYPE:
            return False
        if inp.device.type != "cuda" or inp.device != self.device:
            return False
        if output.dtype != inp.dtype or output.device != inp.device:
            return False
        if not self._is_weak_contiguous(inp) or not output.is_contiguous():
            return False
        if inp.nbytes == 0 or inp.nbytes % 4 != 0:
            return False
        if output.nbytes != inp.nbytes * self.world_size:
            return False
        if (
            is_in_tc_piecewise_cuda_graph()
            or is_in_torch_compile_warmup()
            or get_pcg_capture_stream() is not None
        ):
            return False
        return True

    def dtype_to_mscclpp_dtype(self, dtype: torch.dtype):
        if dtype == torch.float16:
            return self.mscclpp.DataType.float16
        elif dtype == torch.float32:
            return self.mscclpp.DataType.float32
        elif dtype == torch.int32:
            return self.mscclpp.DataType.int32
        elif dtype == torch.bfloat16:
            return self.mscclpp.DataType.bfloat16
        else:
            raise ValueError(f"Unknown data type: {dtype}")

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream: torch.cuda.Stream = None,
    ):
        assert op == torch.distributed.ReduceOp.SUM
        nbytes = tensor.numel() * tensor.element_size()
        algo, nblocks, nthreads = self._get_tuned_config(nbytes)
        self._run_algo(algo, tensor, nbytes, nblocks, nthreads, self.symm_mem_enabled)
        return tensor

    def all_gather(
        self,
        output: torch.Tensor,
        inp: torch.Tensor,
        stream: torch.cuda.Stream = None,
    ):
        algo, nblocks, nthreads = self.allgather_config
        result = self._run_allgather_algo(algo, output, inp, nblocks, nthreads)
        if result != 0:
            raise RuntimeError(f"MSCCL++ all-gather failed with result code {result}")
        return output

    @contextmanager
    def change_state(
        self,
        enable: Optional[bool] = None,
    ):
        if enable is None:
            # guess a default value when not specified
            enable = self.initialized
        elif not self.initialized:
            enable = False

        old_disable = self.disabled
        self.disabled = not enable

        yield

        self.disabled = old_disable
