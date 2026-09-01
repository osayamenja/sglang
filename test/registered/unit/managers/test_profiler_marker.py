import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.io_struct import ProfileReq, ProfileReqType
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.utils.nvtx_utils import (
    BATCH_PHASE_MARKER_PREFIX,
    MEASUREMENT_MARKER_PREFIX,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class TestProfilerMarker(unittest.TestCase):
    def _manager(self):
        manager = SchedulerProfilerManager.__new__(SchedulerProfilerManager)
        manager.ps = SimpleNamespace(
            gpu_id=3,
            tp_rank=1,
            pp_rank=0,
            dp_rank=2,
            attn_dp_rank=2,
            attn_tp_rank=1,
            moe_ep_rank=1,
        )
        return manager

    def test_mark_emits_topology(self):
        manager = self._manager()
        request = ProfileReq(
            req_type=ProfileReqType.MARK,
            run_id="run-123",
            marker_phase="begin",
        )

        with patch(
            "sglang.srt.managers.scheduler_components.profiler_manager.scheduler_nvtx_mark"
        ) as emit:
            output = manager._profile(request)

        self.assertTrue(output.success)
        payload = emit.call_args.args[0]
        self.assertEqual(
            payload,
            {
                "run_id": "run-123",
                "phase": "begin",
                "gpu_id": 3,
                "tp_rank": 1,
                "pp_rank": 0,
                "dp_rank": 2,
                "attn_dp_rank": 2,
                "attn_tp_rank": 1,
                "moe_ep_rank": 1,
            },
        )

    def test_mark_failure_does_not_fall_through_to_stop(self):
        manager = self._manager()
        manager._stop_profile = unittest.mock.MagicMock()
        request = ProfileReq(
            req_type=ProfileReqType.MARK,
            run_id="run-123",
            marker_phase="end",
        )

        with patch(
            "sglang.srt.managers.scheduler_components.profiler_manager.scheduler_nvtx_mark",
            side_effect=RuntimeError("scheduler NVTX disabled"),
        ):
            output = manager._profile(request)

        self.assertFalse(output.success)
        self.assertIn("disabled", output.message)
        manager._stop_profile.assert_not_called()

    def test_marker_encoding_has_stable_prefix_and_compact_json(self):
        payload = {"run_id": "run-123", "phase": "begin"}
        with (
            patch("sglang.srt.utils.nvtx_utils.NVTX_SCHEDULER_ENABLED", True),
            patch("sglang.srt.utils.nvtx_utils._nvtx_module") as nvtx_module,
        ):
            from sglang.srt.utils.nvtx_utils import scheduler_nvtx_mark

            scheduler_nvtx_mark(payload)

        message = nvtx_module.mark.call_args.args[0]
        self.assertTrue(message.startswith(MEASUREMENT_MARKER_PREFIX))
        encoded = message.removeprefix(MEASUREMENT_MARKER_PREFIX)
        self.assertEqual(json.loads(encoded), payload)
        self.assertNotIn(" ", encoded)

    def test_batch_phase_marker_has_stable_prefix(self):
        with (
            patch("sglang.srt.utils.nvtx_utils.NVTX_SCHEDULER_ENABLED", True),
            patch("sglang.srt.utils.nvtx_utils._nvtx_module") as nvtx_module,
        ):
            from sglang.srt.utils.nvtx_utils import scheduler_nvtx_batch_phase_mark

            scheduler_nvtx_batch_phase_mark("prefill")

        nvtx_module.mark.assert_called_once_with(
            BATCH_PHASE_MARKER_PREFIX + "prefill", color="orange"
        )

    def test_batch_phase_marker_is_noop_when_scheduler_nvtx_is_disabled(self):
        with (
            patch("sglang.srt.utils.nvtx_utils.NVTX_SCHEDULER_ENABLED", False),
            patch("sglang.srt.utils.nvtx_utils._nvtx_module") as nvtx_module,
        ):
            from sglang.srt.utils.nvtx_utils import scheduler_nvtx_batch_phase_mark

            scheduler_nvtx_batch_phase_mark("decode")

        nvtx_module.mark.assert_not_called()


if __name__ == "__main__":
    unittest.main()
