import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests
import sglang.benchmark.serving as serving
from sglang.benchmark.datasets import DatasetRow
from sglang.benchmark.serving import (
    RequestFuncInput,
    RequestFuncOutput,
    _extract_sglang_meta_info,
    _measurement_protocol_requested,
    _record_sglang_response_metadata,
    flush_cache_or_raise,
    run_timed_measurement,
    run_warmup_requests,
    set_global_args,
    validate_measured_outputs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _Response:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


class _ServerInfoResponse:
    status_code = 500

    def json(self):
        return {}


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1, 2]


class TestFlushCacheOrRaise(unittest.TestCase):
    def test_http_400_then_success_is_retried(self):
        responses = iter([_Response(400, "busy"), _Response(200, "flushed")])
        calls = []

        def post(*args, **kwargs):
            calls.append((args, kwargs))
            return next(responses)

        clock = _FakeClock()
        flush_cache_or_raise(
            "http://server",
            empty_cache=False,
            overall_timeout_s=2.0,
            server_timeout_s=0.5,
            request_timeout_s=0.5,
            retry_interval_s=0.1,
            _post=post,
            _monotonic=clock.monotonic,
            _sleep=clock.sleep,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1]["params"]["empty_cache"], "false")
        self.assertGreater(calls[0][1]["params"]["timeout"], 0)

    def test_transport_errors_are_retried(self):
        calls = 0

        def post(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise requests.ConnectionError("connection reset")
            return _Response(200, "flushed")

        clock = _FakeClock()
        flush_cache_or_raise(
            "http://server",
            empty_cache=False,
            overall_timeout_s=2.0,
            retry_interval_s=0.1,
            _post=post,
            _monotonic=clock.monotonic,
            _sleep=clock.sleep,
        )
        self.assertEqual(calls, 2)

    def test_permanent_failure_raises_at_deadline(self):
        clock = _FakeClock()

        with self.assertRaisesRegex(RuntimeError, "400: still busy"):
            flush_cache_or_raise(
                "http://server",
                empty_cache=False,
                overall_timeout_s=0.3,
                server_timeout_s=0.1,
                request_timeout_s=0.1,
                retry_interval_s=0.1,
                _post=lambda *args, **kwargs: _Response(400, "still busy"),
                _monotonic=clock.monotonic,
                _sleep=clock.sleep,
            )

    def test_non_retriable_status_fails_immediately(self):
        calls = 0

        def post(*args, **kwargs):
            nonlocal calls
            calls += 1
            return _Response(401, "unauthorized")

        with self.assertRaisesRegex(RuntimeError, "401: unauthorized"):
            flush_cache_or_raise(
                "http://server",
                empty_cache=False,
                _post=post,
            )
        self.assertEqual(calls, 1)


class TestWarmupExecution(unittest.IsolatedAsyncioTestCase):
    def _input(self):
        return RequestFuncInput(
            prompt="warmup",
            api_url="http://server/generate",
            prompt_len=1,
            output_len=2,
            model="model",
            lora_name=None,
            image_data=None,
            extra_request_body={},
        )

    async def test_warmups_respect_max_concurrency(self):
        active = 0
        max_active = 0
        semaphore = asyncio.Semaphore(1)

        async def request_func(request_func_input, pbar):
            nonlocal active, max_active
            async with semaphore:
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0)
                active -= 1
                return RequestFuncOutput(success=True)

        outputs = await run_warmup_requests(
            request_func, self._input(), 8, is_multi_turn=False
        )

        self.assertEqual(len(outputs), 8)
        self.assertEqual(max_active, 1)

    async def test_every_failed_warmup_is_reported(self):
        call_index = 0

        async def request_func(request_func_input, pbar):
            nonlocal call_index
            index = call_index
            call_index += 1
            return RequestFuncOutput(success=False, error=f"failure {index}")

        with self.assertRaises(ValueError) as context:
            await run_warmup_requests(
                request_func, self._input(), 3, is_multi_turn=False
            )

        for index in range(3):
            self.assertIn(f"failure {index}", str(context.exception))

    async def test_multi_turn_outputs_are_flattened(self):
        async def request_func(request_func_input, pbar):
            return [
                RequestFuncOutput(success=True),
                RequestFuncOutput(success=True),
            ]

        outputs = await run_warmup_requests(
            request_func, self._input(), 2, is_multi_turn=True
        )
        self.assertEqual(len(outputs), 4)

    async def test_profiler_shutdown_delay_does_not_affect_duration(self):
        timestamps = iter((10.0, 12.0))

        async def measured_requests():
            return [RequestFuncOutput(success=True)]

        _, start, end = await run_timed_measurement(
            measured_requests, _perf_counter=lambda: next(timestamps)
        )
        profiler_shutdown_delay = 50.0

        self.assertEqual(end - start, 2.0)
        self.assertNotEqual(end - start + profiler_shutdown_delay, end - start)


class TestBenchmarkModeGating(unittest.IsolatedAsyncioTestCase):
    def _args(self, output_file, *, measurement):
        return SimpleNamespace(
            backend="sglang" if measurement else "vllm",
            cache_report=False,
            dataset_name="random",
            measurement_dp_size=2 if measurement else 0,
            measurement_run_id="trace-run" if measurement else None,
            output_details=True,
            output_file=output_file,
            plot_throughput=False,
            random_input_len=1,
            random_output_len=2,
            random_range_ratio=1.0,
            sharegpt_output_len=None,
            tag=None,
        )

    def _request(self):
        return DatasetRow(prompt="test", prompt_len=1, output_len=2)

    async def test_regular_benchmark_preserves_upstream_flow_and_schema(self):
        async def request_func(request_func_input, pbar=None):
            return RequestFuncOutput(
                generated_text="ok",
                success=True,
                latency=0.2,
                ttft=0.1,
                itl=[0.1],
                prompt_len=request_func_input.prompt_len,
                output_len=2,
                start_time=1.0,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = str(Path(temp_dir) / "result.jsonl")
            set_global_args(self._args(output_file, measurement=False))
            with (
                patch.dict(serving.ASYNC_REQUEST_FUNCS, {"vllm": request_func}),
                patch.object(serving, "flush_server_cache") as standard_flush,
                patch.object(serving, "flush_cache_or_raise") as strict_flush,
                patch.object(serving.time, "sleep") as settle,
                patch.object(serving.time, "perf_counter", side_effect=(10.0, 13.0)),
                patch.object(
                    serving.requests, "get", return_value=_ServerInfoResponse()
                ),
            ):
                result = await serving.benchmark(
                    backend="vllm",
                    api_url="http://server/v1/completions",
                    base_url="http://server",
                    model_id="model",
                    tokenizer=_Tokenizer(),
                    input_requests=[self._request()],
                    request_rate=float("inf"),
                    max_concurrency=1,
                    disable_tqdm=True,
                    lora_names=[],
                    lora_request_distribution=None,
                    lora_zipf_alpha=None,
                    extra_request_body={},
                    profile=False,
                    flush_cache=True,
                    warmup_requests=1,
                )

            standard_flush.assert_called_once()
            strict_flush.assert_not_called()
            settle.assert_called_once_with(1.0)
            self.assertEqual(result["duration"], 3.0)
            self.assertNotIn("run_id", result)
            self.assertNotIn("send_times", result)
            self.assertNotIn("finish_times", result)
            self.assertNotIn("dp_ranks", result)

    async def test_suite_measurement_retains_strict_flow_and_schema(self):
        async def request_func(request_func_input, pbar=None):
            dp_rank = request_func_input.extra_request_body.get("routed_dp_rank", 0)
            return RequestFuncOutput(
                generated_text="ok",
                success=True,
                latency=0.2,
                ttft=0.1,
                itl=[0.1],
                prompt_len=request_func_input.prompt_len,
                output_len=2,
                start_time=1.0,
                send_time=1.1,
                finish_time=1.2,
                cached_tokens=0,
                cache_metadata_seen=True,
                dp_rank=dp_rank,
            )

        async def timed_measurement(operation):
            return await operation(), 10.0, 12.0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = str(Path(temp_dir) / "result.jsonl")
            set_global_args(self._args(output_file, measurement=True))
            with (
                patch.dict(serving.ASYNC_REQUEST_FUNCS, {"sglang": request_func}),
                patch.object(serving, "flush_server_cache") as standard_flush,
                patch.object(serving, "flush_cache_or_raise") as strict_flush,
                patch.object(
                    serving, "run_timed_measurement", side_effect=timed_measurement
                ),
                patch.object(serving.time, "sleep") as settle,
                patch.object(
                    serving.requests, "get", return_value=_ServerInfoResponse()
                ),
            ):
                result = await serving.benchmark(
                    backend="sglang",
                    api_url="http://server/generate",
                    base_url="http://server",
                    model_id="model",
                    tokenizer=_Tokenizer(),
                    input_requests=[self._request()],
                    request_rate=float("inf"),
                    max_concurrency=1,
                    disable_tqdm=True,
                    lora_names=[],
                    lora_request_distribution=None,
                    lora_zipf_alpha=None,
                    extra_request_body={},
                    profile=False,
                    flush_cache=True,
                    warmup_requests=1,
                    measurement_run_id="trace-run",
                    measurement_dp_size=2,
                    require_zero_cached_tokens=True,
                    require_dp_rank=True,
                )

            standard_flush.assert_not_called()
            self.assertEqual(strict_flush.call_count, 3)
            settle.assert_not_called()
            self.assertEqual(result["duration"], 2.0)
            self.assertEqual(result["run_id"], "trace-run")
            self.assertEqual(result["send_times"], [1.1])
            self.assertEqual(result["finish_times"], [1.2])
            self.assertEqual(result["dp_ranks"], [0])


class TestMeasurementMetadata(unittest.TestCase):
    def test_regular_benchmark_ignores_trace_only_metadata(self):
        set_global_args(
            SimpleNamespace(
                measurement_run_id=None,
                measurement_dp_size=0,
                cache_report=False,
            )
        )
        output = RequestFuncOutput()

        _record_sglang_response_metadata(
            {"meta_info": {"cached_tokens": "invalid", "dp_rank": "invalid"}},
            output,
        )

        self.assertFalse(_measurement_protocol_requested())
        self.assertEqual(output.cached_tokens, 0)
        self.assertFalse(output.cache_metadata_seen)
        self.assertIsNone(output.dp_rank)

    def test_regular_cache_report_preserves_upstream_metadata_handling(self):
        set_global_args(
            SimpleNamespace(
                measurement_run_id=None,
                measurement_dp_size=0,
                cache_report=True,
            )
        )
        output = RequestFuncOutput()

        _record_sglang_response_metadata(
            {
                "meta_info": {
                    "cached_tokens": 7,
                    "cached_tokens_details": {"device": 7},
                    "dp_rank": 2,
                }
            },
            output,
        )

        self.assertEqual(output.cached_tokens, 7)
        self.assertEqual(output.cached_tokens_details, {"device": 7})
        self.assertFalse(output.cache_metadata_seen)
        self.assertIsNone(output.dp_rank)

    def test_measurement_protocol_enables_strict_trace_metadata(self):
        set_global_args(
            SimpleNamespace(
                measurement_run_id="trace-run",
                measurement_dp_size=4,
                cache_report=True,
            )
        )
        output = RequestFuncOutput()

        _record_sglang_response_metadata(
            {"meta_info": {"cached_tokens": 0, "dp_rank": 2}}, output
        )

        self.assertTrue(_measurement_protocol_requested())
        self.assertTrue(output.cache_metadata_seen)
        self.assertEqual(output.dp_rank, 2)

    def test_cache_contaminated_request_is_rejected(self):
        output = RequestFuncOutput(
            success=True,
            cached_tokens=7,
            cache_metadata_seen=True,
            dp_rank=0,
        )
        with self.assertRaisesRegex(ValueError, "cached_tokens=7"):
            validate_measured_outputs(
                [output],
                require_zero_cached_tokens=True,
                require_dp_rank=True,
            )

    def test_missing_cache_metadata_is_rejected(self):
        output = RequestFuncOutput(success=True, cached_tokens=0, dp_rank=0)
        with self.assertRaisesRegex(ValueError, "did not report cached_tokens"):
            validate_measured_outputs(
                [output],
                require_zero_cached_tokens=True,
                require_dp_rank=True,
            )

    def test_streaming_dp_rank_change_is_rejected(self):
        output = RequestFuncOutput()
        _extract_sglang_meta_info(
            {"meta_info": {"cached_tokens": 0, "dp_rank": 0}}, output
        )
        with self.assertRaisesRegex(ValueError, "changed DP rank"):
            _extract_sglang_meta_info(
                {"meta_info": {"cached_tokens": 0, "dp_rank": 1}}, output
            )

    def test_missing_dp_rank_is_rejected(self):
        output = RequestFuncOutput(
            success=True, cached_tokens=0, cache_metadata_seen=True
        )
        with self.assertRaisesRegex(ValueError, "did not report a DP rank"):
            validate_measured_outputs(
                [output],
                require_zero_cached_tokens=True,
                require_dp_rank=True,
            )


if __name__ == "__main__":
    unittest.main()
