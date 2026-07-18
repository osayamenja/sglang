import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.device_communicators import torchcomms_adapter


class _FakeComm:
    def __init__(self):
        self.calls = []
        self.finalized = False

    def split(self, ranks, name):
        self.calls.append(("split", ranks, name))
        return self

    def get_rank(self):
        return 0

    def get_size(self):
        return 2

    def all_reduce(self, *args):
        self.calls.append(("all_reduce", *args))

    def reduce_scatter_single(self, *args):
        self.calls.append(("reduce_scatter_single", *args))

    def reduce_scatter_v(self, *args):
        self.calls.append(("reduce_scatter_v", *args))

    def all_gather_single(self, *args):
        self.calls.append(("all_gather_single", *args))

    def all_gather_v(self, *args):
        self.calls.append(("all_gather_v", *args))

    def all_to_all_single(self, *args):
        self.calls.append(("all_to_all_single", *args))

    def all_to_all_v_single(self, *args):
        self.calls.append(("all_to_all_v_single", *args))

    def finalize(self):
        self.finalized = True


class TestTorchCommsAdapter(unittest.TestCase):
    def setUp(self):
        self.fake_comm = _FakeComm()
        self.fake_module = SimpleNamespace(ReduceOp=SimpleNamespace(SUM="sum"))
        get_root = patch.object(
            torchcomms_adapter,
            "_get_root_comm",
            return_value=(self.fake_module, self.fake_comm),
        )
        self.addCleanup(get_root.stop)
        get_root.start()
        self.adapter = torchcomms_adapter.TorchCommsCommunicator(
            ranks=[0, 1], device=torch.device("cuda:0"), name="tp:0"
        )

    def test_collective_mapping(self):
        input_tensor = torch.arange(8).reshape(4, 2)
        output_tensor = torch.empty_like(input_tensor)

        self.adapter.all_reduce(input_tensor)
        self.adapter.reduce_scatter(output_tensor[:2], input_tensor)
        self.adapter.reduce_scatter(output_tensor[:1], input_tensor, sizes=[1, 3])
        self.adapter.all_gather(output_tensor, input_tensor[:2])
        self.adapter.all_gather(output_tensor, input_tensor[:1], sizes=[1, 3])
        self.adapter.all_to_all(output_tensor, input_tensor)
        self.adapter.all_to_all(
            output_tensor,
            input_tensor,
            output_split_sizes=[1, 3],
            input_split_sizes=[2, 2],
        )

        call_names = [call[0] for call in self.fake_comm.calls]
        self.assertEqual(
            call_names,
            [
                "split",
                "all_reduce",
                "reduce_scatter_single",
                "reduce_scatter_v",
                "all_gather_single",
                "all_gather_v",
                "all_to_all_single",
                "all_to_all_v_single",
            ],
        )

    def test_finalize_is_idempotent(self):
        self.adapter.finalize()
        self.adapter.finalize()
        self.assertTrue(self.fake_comm.finalized)

    def test_world_environment_is_restored(self):
        old_rank = os.environ.pop("TORCHCOMM_RANK", None)
        old_size = os.environ.get("TORCHCOMM_SIZE")
        os.environ["TORCHCOMM_SIZE"] = "old"
        try:
            with torchcomms_adapter._torchcomms_world_env(3, 8):
                self.assertEqual(os.environ["TORCHCOMM_RANK"], "3")
                self.assertEqual(os.environ["TORCHCOMM_SIZE"], "8")
            self.assertNotIn("TORCHCOMM_RANK", os.environ)
            self.assertEqual(os.environ["TORCHCOMM_SIZE"], "old")
        finally:
            if old_rank is not None:
                os.environ["TORCHCOMM_RANK"] = old_rank
            else:
                os.environ.pop("TORCHCOMM_RANK", None)
            if old_size is not None:
                os.environ["TORCHCOMM_SIZE"] = old_size
            else:
                os.environ.pop("TORCHCOMM_SIZE", None)


if __name__ == "__main__":
    unittest.main()
