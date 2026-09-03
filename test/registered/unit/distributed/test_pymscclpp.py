import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.device_communicators import pymscclpp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _tuning_communicator(world_size: int):
    comm = pymscclpp.PyMscclppCommunicator.__new__(pymscclpp.PyMscclppCommunicator)
    comm.world_size = world_size
    comm.device = torch.device("cuda:0")
    return comm


class TestPyMscclppTuningEligibility(unittest.TestCase):
    def test_a100_rejects_hanging_rsag_tuning_shapes(self):
        comm = _tuning_communicator(8)
        algo = SimpleNamespace(name="default_allreduce_rsag_zero_copy")
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
            self.assertFalse(comm._is_tuning_candidate_supported(algo, 128, 768))
            self.assertFalse(comm._is_tuning_candidate_supported(algo, 128, 1024))
            self.assertTrue(comm._is_tuning_candidate_supported(algo, 128, 512))
            self.assertTrue(comm._is_tuning_candidate_supported(algo, 64, 1024))

    def test_h100_keeps_full_rsag_tuning_space(self):
        comm = _tuning_communicator(8)
        algo = SimpleNamespace(name="default_allreduce_rsag_zero_copy")
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            self.assertTrue(comm._is_tuning_candidate_supported(algo, 128, 768))
            self.assertTrue(comm._is_tuning_candidate_supported(algo, 128, 1024))


def _tensor(
    *,
    nbytes: int = 1024,
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cuda:0"),
    contiguous: bool = True,
):
    element_size = torch.empty((), dtype=dtype).element_size()
    return SimpleNamespace(
        dtype=dtype,
        device=device,
        nbytes=nbytes,
        is_contiguous=lambda: contiguous,
        numel=lambda: nbytes // element_size,
        element_size=lambda: element_size,
    )


def _communicator(world_size: int):
    comm = pymscclpp.PyMscclppCommunicator.__new__(pymscclpp.PyMscclppCommunicator)
    comm.disabled = False
    comm.initialized = True
    comm.world_size = world_size
    comm.device = torch.device("cuda:0")
    comm.allgather_config = (object(), 0, 0)
    comm.best_configs = {1024: (object(), 1, 1)}
    return comm


class TestPyMscclppEligibility(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for function in (
            "is_in_tc_piecewise_cuda_graph",
            "is_in_torch_compile_warmup",
        ):
            self.stack.enter_context(
                patch.object(pymscclpp, function, return_value=False)
            )
        self.stack.enter_context(
            patch.object(pymscclpp, "get_pcg_capture_stream", return_value=None)
        )

    def test_supported_communicator_sizes(self):
        self.assertEqual(
            pymscclpp.PyMscclppCommunicator._NATIVE_WORLD_SIZES,
            [2, 4, 8],
        )
        self.assertEqual(
            pymscclpp.PyMscclppCommunicator._DSL_ALLREDUCE_WORLD_SIZES,
            [16, 32],
        )

    def test_allreduce_supports_native_sizes(self):
        inp = _tensor()
        for world_size in (2, 4, 8):
            with self.subTest(world_size=world_size):
                self.assertTrue(_communicator(world_size).should_mscclpp_allreduce(inp))

    def test_allgather_supports_native_sizes(self):
        inp = _tensor()
        for world_size in (2, 4, 8):
            with self.subTest(world_size=world_size):
                output = _tensor(nbytes=inp.nbytes * world_size)
                self.assertTrue(
                    _communicator(world_size).should_mscclpp_allgather(output, inp)
                )

    def test_allgather_rejects_multinode_and_invalid_buffers(self):
        inp = _tensor()
        self.assertFalse(
            _communicator(16).should_mscclpp_allgather(
                _tensor(nbytes=inp.nbytes * 16), inp
            )
        )
        self.assertFalse(
            _communicator(2).should_mscclpp_allgather(_tensor(nbytes=inp.nbytes), inp)
        )
        odd_input = _tensor(nbytes=6)
        self.assertFalse(
            _communicator(2).should_mscclpp_allgather(_tensor(nbytes=12), odd_input)
        )

    def test_disabled_and_piecewise_graph_fall_back(self):
        comm = _communicator(2)
        inp = _tensor()
        output = _tensor(nbytes=inp.nbytes * 2)

        comm.disabled = True
        self.assertFalse(comm.should_mscclpp_allreduce(inp))
        self.assertFalse(comm.should_mscclpp_allgather(output, inp))

        comm.disabled = False
        with patch.object(
            pymscclpp, "is_in_tc_piecewise_cuda_graph", return_value=True
        ):
            self.assertFalse(comm.should_mscclpp_allreduce(inp))
            self.assertFalse(comm.should_mscclpp_allgather(output, inp))


if __name__ == "__main__":
    unittest.main()
