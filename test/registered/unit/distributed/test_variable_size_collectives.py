from __future__ import annotations

import unittest
from contextlib import contextmanager

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator


class FakePyNcclCommunicator:
    def __init__(self):
        self.available = True
        self.disabled = True
        self.calls = []

    @contextmanager
    def change_state(self, enable=True):
        old_disabled = self.disabled
        self.disabled = not enable
        try:
            yield
        finally:
            self.disabled = old_disabled

    def reduce_scatter(self, output, input_, sizes=None):
        assert not self.disabled
        self.calls.append(("reduce_scatter", sizes))
        output.copy_(input_[: output.shape[0]])

    def group_start(self):
        assert not self.disabled
        self.calls.append(("group_start", None))

    def all_gather(self, output, input_, sizes=None):
        assert not self.disabled
        self.calls.append(("all_gather", sizes))

    def group_end(self):
        assert not self.disabled
        self.calls.append(("group_end", None))


def make_coordinator():
    coordinator = GroupCoordinator.__new__(GroupCoordinator)
    coordinator.world_size = 2
    coordinator.rank_in_group = 0
    coordinator.pynccl_comm = FakePyNcclCommunicator()
    coordinator._can_use_purlin = lambda *args, **kwargs: False
    return coordinator


class VariableSizeCollectiveTests(unittest.TestCase):
    def test_reduce_scatterv_temporarily_enables_pynccl(self):
        coordinator = make_coordinator()
        input_ = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        output = torch.empty((1, 2), dtype=torch.float32)

        result = coordinator.reduce_scatterv(input_, output=output, sizes=[1, 1])

        self.assertIs(result, output)
        self.assertEqual(coordinator.pynccl_comm.calls, [("reduce_scatter", [1, 1])])
        self.assertTrue(coordinator.pynccl_comm.disabled)

    def test_all_gatherv_temporarily_enables_pynccl(self):
        coordinator = make_coordinator()
        input_ = torch.arange(2, dtype=torch.float32).reshape(1, 2)
        output = torch.empty((2, 2), dtype=torch.float32)

        result = coordinator.all_gatherv(input_, sizes=[1, 1], output=output)

        self.assertEqual(result, [output])
        self.assertEqual(
            coordinator.pynccl_comm.calls,
            [
                ("group_start", None),
                ("all_gather", None),
                ("group_end", None),
            ],
        )
        self.assertTrue(coordinator.pynccl_comm.disabled)


if __name__ == "__main__":
    unittest.main()
