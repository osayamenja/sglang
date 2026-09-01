import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import FlushCacheReqInput, FlushCacheReqOutput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.flush_wrapper import (
    SchedulerFlushWrapper,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin

register_cpu_ci(est_time=14, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class TestSchedulerFlushCache(unittest.TestCase):
    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ipc_channels = MagicMock()
        scheduler.flush_cache = MagicMock(return_value=True)
        scheduler.is_fully_idle = MagicMock(return_value=False)
        scheduler.flush_wrapper = SchedulerFlushWrapper(
            flush_cache=scheduler.flush_cache,
            is_fully_idle=scheduler.is_fully_idle,
            ipc_channels=scheduler.ipc_channels,
        )
        return scheduler

    def test_immediate_flush_no_timeout(self):
        """No timeout → flush immediately regardless of idle state."""
        scheduler = self._new_scheduler()
        scheduler.flush_cache.return_value = False

        output = scheduler.flush_wrapper.handle(FlushCacheReqInput(timeout_s=None))

        self.assertFalse(output.success)
        scheduler.flush_cache.assert_called_once_with(empty_cache=True)

    def test_immediate_flush_propagates_empty_cache_false(self):
        scheduler = self._new_scheduler()

        output = scheduler.flush_wrapper.handle(
            FlushCacheReqInput(timeout_s=None, empty_cache=False)
        )

        self.assertTrue(output.success)
        scheduler.flush_cache.assert_called_once_with(empty_cache=False)

    def test_immediate_flush_when_idle(self):
        """Positive timeout but already idle → flush immediately."""
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True

        output = scheduler.flush_wrapper.handle(FlushCacheReqInput(timeout_s=5.0))

        self.assertTrue(output.success)
        scheduler.flush_cache.assert_called_once_with(empty_cache=True)

    def test_defers_when_busy(self):
        """Positive timeout + busy → defers, returns None."""
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=3.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=10.0,
        ):
            output = scheduler.flush_wrapper.handle(req)

        self.assertIsNone(output)
        pending_req, deadline = scheduler.flush_wrapper._pending
        self.assertIs(pending_req, req)
        self.assertEqual(deadline, 13.0)

    def test_rejects_when_already_pending(self):
        """Any new request is rejected while another is pending."""
        scheduler = self._new_scheduler()
        scheduler.flush_wrapper._pending = (FlushCacheReqInput(timeout_s=10.0), 999.0)

        for timeout in [None, 5.0]:
            output = scheduler.flush_wrapper.handle(
                FlushCacheReqInput(timeout_s=timeout)
            )
            self.assertFalse(output.success)
            self.assertIn("already in progress", output.message)

        scheduler.flush_cache.assert_not_called()

    def test_pending_flush_completes_on_idle(self):
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True
        req = FlushCacheReqInput(timeout_s=1.0)
        scheduler.flush_wrapper._pending = (req, 111.0)

        scheduler.flush_wrapper.check_pending()

        self.assertIsNone(scheduler.flush_wrapper._pending)
        scheduler.flush_cache.assert_called_once_with(empty_cache=True)
        out = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        self.assertTrue(out.success)

    def test_pending_flush_propagates_empty_cache_false(self):
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True
        req = FlushCacheReqInput(timeout_s=1.0, empty_cache=False)
        scheduler.flush_wrapper._pending = (req, 111.0)

        scheduler.flush_wrapper.check_pending()

        scheduler.flush_cache.assert_called_once_with(empty_cache=False)

    def test_pending_flush_expires_on_timeout(self):
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=1.0)
        scheduler.flush_wrapper._pending = (req, 99.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=100.0,
        ):
            scheduler.flush_wrapper.check_pending()

        self.assertIsNone(scheduler.flush_wrapper._pending)
        scheduler.flush_cache.assert_not_called()
        out = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        self.assertFalse(out.success)

    def test_pending_flush_survives_before_deadline(self):
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=5.0)
        scheduler.flush_wrapper._pending = (req, 101.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=100.0,
        ):
            scheduler.flush_wrapper.check_pending()

        self.assertIsNotNone(scheduler.flush_wrapper._pending)
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()


class TestTokenizerFlushCache(unittest.IsolatedAsyncioTestCase):
    async def test_any_failed_dp_rank_makes_aggregate_fail(self):
        manager = MagicMock()
        manager.auto_create_handle_loop = MagicMock()
        manager.flush_cache_communicator = AsyncMock(
            return_value=[
                FlushCacheReqOutput(success=True, message="rank 0 ok"),
                FlushCacheReqOutput(success=False, message="rank 1 busy"),
            ]
        )

        output = await TokenizerControlMixin.flush_cache(
            manager, timeout_s=5.0, empty_cache=False
        )

        self.assertFalse(output.success)
        self.assertIn("rank 1 busy", output.message)
        request = manager.flush_cache_communicator.await_args.args[0]
        self.assertEqual(request.timeout_s, 5.0)
        self.assertFalse(request.empty_cache)


if __name__ == "__main__":
    unittest.main()
