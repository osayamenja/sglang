import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.device_communicators import pymscclpp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _communicator(world_size: int):
    comm = pymscclpp.PyMscclppCommunicator.__new__(pymscclpp.PyMscclppCommunicator)
    comm.world_size = world_size
    comm.device = torch.device("cuda:0")
    return comm


class TestPyMscclppTuningEligibility(unittest.TestCase):
    def test_a100_rejects_hanging_rsag_tuning_shapes(self):
        comm = _communicator(8)
        algo = SimpleNamespace(name="default_allreduce_rsag_zero_copy")
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
            self.assertFalse(comm._is_tuning_candidate_supported(algo, 128, 768))
            self.assertFalse(comm._is_tuning_candidate_supported(algo, 128, 1024))
            self.assertTrue(comm._is_tuning_candidate_supported(algo, 128, 512))
            self.assertTrue(comm._is_tuning_candidate_supported(algo, 64, 1024))

    def test_h100_keeps_full_rsag_tuning_space(self):
        comm = _communicator(8)
        algo = SimpleNamespace(name="default_allreduce_rsag_zero_copy")
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            self.assertTrue(comm._is_tuning_candidate_supported(algo, 128, 768))
            self.assertTrue(comm._is_tuning_candidate_supported(algo, 128, 1024))


if __name__ == "__main__":
    unittest.main()
