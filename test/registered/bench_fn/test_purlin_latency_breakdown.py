import json
import re
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from sglang.benchmark import one_batch
from sglang.test.ci.ci_register import register_cpu_ci

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[3] / "benchmark" / "purlin_latency_breakdown"
)
CUSTOM_ALL_REDUCE_HEADER = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh"
)
sys.path.insert(0, str(BENCHMARK_DIR))

import analyze_full_trace as trace  # noqa: E402
import analyze_one_batch_trace as one_batch_trace  # noqa: E402
import run_one_batch_suite as one_batch_suite  # noqa: E402
import run_suite as suite  # noqa: E402
import summarize_one_batch_results as one_batch_summary  # noqa: E402
import summarize_results as summary  # noqa: E402

register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class TestOneBatchSuite(unittest.TestCase):
    def _parse_args(self, *extra):
        argv = [
            "run_one_batch_suite.py",
            "--model",
            "test-model",
            "--tp",
            "2",
            "--output-dir",
            "test-output",
            *extra,
        ]
        with patch.object(sys, "argv", argv):
            return one_batch_suite.parse_args()

    def test_matched_commands_only_enable_purlin_for_purlin(self):
        args = self._parse_args("--repeats", "3")
        self.assertEqual(args.cuda_graph_backend_prefill, "breakable")
        self.assertEqual(args.cuda_graph_bs_prefill, [1024])
        baseline = one_batch_suite.one_batch_command(
            args, "baseline", Path("baseline.jsonl"), repeats=3
        )
        purlin = one_batch_suite.one_batch_command(
            args, "purlin", Path("purlin.jsonl"), repeats=3
        )

        self.assertNotIn("--enable-purlin", baseline)
        self.assertIn("--enable-purlin", purlin)
        batch_index = baseline.index("--batch-size")
        self.assertEqual(baseline[batch_index + 1 : batch_index + 4], ["1"] * 3)
        self.assertEqual(
            baseline[baseline.index("--tp-size") + 1],
            "2",
        )

    def test_default_nsys_binary_and_node_trace(self):
        args = self._parse_args()
        self.assertEqual(args.nsys_command, one_batch_suite.DEFAULT_NSYS)
        with patch.object(one_batch_suite.subprocess, "run") as run:
            run.return_value.stdout = "--cuda-graph-trace=<granularity>"
            self.assertEqual(
                one_batch_suite.resolve_cuda_graph_trace_mode(args), "node"
            )

        command = one_batch_suite.nsys_command(
            args, ["python", "benchmark.py"], Path("trace"), "node"
        )
        self.assertEqual(command[0], str(one_batch_suite.DEFAULT_NSYS))
        self.assertIn("--cuda-graph-trace=node", command)
        self.assertIn("--capture-range=cudaProfilerApi", command)

    def test_resume_rejects_a_different_configuration(self):
        args = self._parse_args()
        manifest = one_batch_suite.build_manifest(args, "node")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            one_batch_suite.validate_resume_manifest(args, "node", path)
            args.batch_size = 2
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                one_batch_suite.validate_resume_manifest(args, "node", path)

    def test_clean_summary_uses_medians_and_directional_speedup(self):
        records = [
            {
                "prefill_latency": 0.012,
                "median_decode_latency": 0.002,
                "total_latency": 0.020,
            },
            {
                "prefill_latency": 0.010,
                "median_decode_latency": 0.001,
                "total_latency": 0.018,
            },
        ]
        summarized = one_batch_suite.summarize_samples(records)
        self.assertEqual(summarized["metrics"]["prefill_latency"]["median"], 11.0)
        comparison = one_batch_suite.compare_variants(
            {
                "baseline": summarized,
                "purlin": one_batch_suite.summarize_samples(
                    [
                        {
                            "prefill_latency": 0.0055,
                            "median_decode_latency": 0.00075,
                            "total_latency": 0.0095,
                        }
                    ]
                ),
            }
        )
        self.assertEqual(comparison["prefill_latency"]["purlin_speedup"], 2.0)

    def test_static_trace_breakdown_is_additive(self):
        connection = sqlite3.connect(":memory:")
        connection.executescript("""
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                deviceId INTEGER, start INTEGER, end INTEGER,
                shortName INTEGER, graphId INTEGER, streamId INTEGER
            );
            INSERT INTO StringIds VALUES (1, 'compute_kernel');
            INSERT INTO StringIds VALUES (2, 'allGatherKernel');
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 100, 180, 1, 7, 1);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 160, 220, 2, 7, 2);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 240, 260, 1, NULL, 1);
            """)

        result = one_batch_trace.analyze_trace(
            connection, label="baseline", stage="decode"
        )

        critical = result["critical_path"]
        self.assertEqual(critical["additive_error_ns"], 0)
        self.assertEqual(critical["compute_active_ms"], 0.0001)
        self.assertEqual(critical["communication_active_ms"], 0.00006)
        self.assertEqual(critical["compute_communication_overlap_ms"], 0.00002)
        self.assertEqual(critical["uncovered_ms"], 0.00002)

    def test_cuda_profiler_sentinel_stops_without_torch_trace_export(self):
        with (
            patch.object(one_batch.torch.cuda, "cudart") as cudart,
            patch.object(one_batch, "_save_profile_trace_results") as save_trace,
        ):
            profiler = one_batch.start_profile(["CUDA_PROFILER"])
            one_batch.stop_profile(
                profiler,
                ["CUDA_PROFILER"],
                save_trace=True,
                trace_filename="unused.trace.json.gz",
                stage="decode",
            )

        self.assertEqual(profiler, "cuda_profiler")
        cudart.return_value.cudaProfilerStart.assert_called_once_with()
        cudart.return_value.cudaProfilerStop.assert_called_once_with()
        save_trace.assert_not_called()

    def test_comparison_uses_static_prefill_decode_and_derived_e2e(self):
        clean_summary = {
            "variants": {
                variant: {
                    "samples": 8,
                    "metrics": {
                        "prefill_latency": {"median": values[0]},
                        "median_decode_latency": {"median": values[1]},
                        "total_latency": {"median": values[2]},
                    },
                }
                for variant, values in {
                    "baseline": (10.0, 2.0, 14.0),
                    "purlin": (8.0, 1.0, 10.0),
                }.items()
            }
        }
        manifest = {"workload": {"output_len": 3}}

        def breakdown(total, compute, communication, uncovered, overlap=0.0):
            return {
                "critical_device": 0,
                "critical_path": {
                    "span_ms": total,
                    "compute_active_ms": compute,
                    "communication_active_ms": communication + overlap,
                    "communication_exclusive_ms": communication,
                    "compute_communication_overlap_ms": overlap,
                    "uncovered_ms": uncovered,
                },
            }

        breakdowns = {
            "baseline": {
                "prefill": breakdown(10.0, 6.0, 3.0, 1.0),
                "decode": breakdown(2.0, 1.0, 0.5, 0.5),
            },
            "purlin": {
                "prefill": breakdown(8.0, 6.0, 1.0, 1.0),
                "decode": breakdown(1.0, 0.7, 0.2, 0.1),
            },
        }

        result = one_batch_summary.build_comparison(clean_summary, manifest, breakdowns)

        self.assertEqual(
            set(result["metrics"]),
            {
                "prefill_latency",
                "decode_latency",
                "e2e_time",
            },
        )
        e2e = result["metrics"]["e2e_time"]
        self.assertEqual(e2e["clean_static"]["baseline_ms"], 14.0)
        self.assertEqual(e2e["trace_model"]["total"]["baseline_ms"], 14.0)
        self.assertEqual(e2e["trace_model"]["compute"]["baseline_ms"], 8.0)
        self.assertEqual(e2e["trace_model"]["communication"]["baseline_ms"], 4.0)
        self.assertEqual(e2e["trace_model"]["uncovered"]["baseline_ms"], 2.0)
        self.assertEqual(result["trace_derivation"]["decode_steps_in_e2e"], 2)


class TestRunSuiteDefaults(unittest.TestCase):
    def test_prefill_cuda_graph_defaults_to_breakable(self):
        argv = [
            "run_suite.py",
            "--model",
            "test-model",
            "--tp",
            "2",
            "--output-dir",
            "test-output",
        ]
        with patch.object(sys, "argv", argv):
            args = suite.parse_args()

        self.assertEqual(args.cuda_graph_backend_prefill, "breakable")
        self.assertTrue(args.require_prefill_cuda_graph)
        self.assertTrue(args.enable_prefill_delayer)
        command = suite.common_server_command(args, "baseline")
        option_index = command.index("--cuda-graph-backend-prefill")
        self.assertEqual(command[option_index + 1], "breakable")
        self.assertIn("--enable-prefill-delayer", command)

        client = suite.client_command(
            args,
            Path("results.jsonl"),
            num_prompts=4,
            profile=False,
            run_id="baseline-clean-test",
        )
        self.assertEqual(
            client[client.index("--measurement-run-id") + 1],
            "baseline-clean-test",
        )
        self.assertEqual(
            client[client.index("--measurement-dp-size") + 1], str(args.dp)
        )

    def test_prefill_delayer_can_be_disabled_explicitly(self):
        argv = [
            "run_suite.py",
            "--model",
            "test-model",
            "--tp",
            "2",
            "--output-dir",
            "test-output",
            "--no-enable-prefill-delayer",
        ]
        with patch.object(sys, "argv", argv):
            args = suite.parse_args()

        self.assertFalse(args.enable_prefill_delayer)
        self.assertNotIn(
            "--enable-prefill-delayer",
            suite.common_server_command(args, "baseline"),
        )


def _rank_step(
    device,
    start,
    end,
    *,
    compute_intervals=None,
    communication_intervals=None,
    phase="prefill",
):
    compute_intervals = tuple(compute_intervals or ((start, end),))
    communication_intervals = tuple(communication_intervals or ())
    total = end - start
    compute = trace.interval_union_length(compute_intervals)
    communication_active = trace.interval_union_length(communication_intervals)
    overlap = trace.interval_intersection_length(
        communication_intervals, compute_intervals
    )
    coverage = trace.interval_union_length(
        (*compute_intervals, *communication_intervals)
    )
    return trace.RankStep(
        device=device,
        start=start,
        end=end,
        total=total,
        compute=compute,
        communication=communication_active - overlap,
        uncovered=total - coverage,
        communication_kernel_active=communication_active,
        communication_compute_overlap=overlap,
        kernel_coverage=coverage,
        kernel_count=max(1, len(compute_intervals) + len(communication_intervals)),
        graph_kernel_count=0,
        stream_count=1,
        compute_intervals=compute_intervals,
        communication_intervals=communication_intervals,
        phase=phase,
    )


def _topology(device, pid, dp_rank, attn_tp_rank=0):
    return trace.SchedulerTopology(
        pid=pid,
        device=device,
        begin=100,
        end=500,
        gpu_id=device,
        tp_rank=device,
        pp_rank=0,
        dp_rank=dp_rank,
        attn_dp_rank=dp_rank,
        attn_tp_rank=attn_tp_rank,
        moe_ep_rank=device,
    )


class TestTraceBoundaries(unittest.TestCase):
    def test_scheduler_phase_marker_is_paired_with_enclosing_range(self):
        connection = sqlite3.connect(":memory:")
        connection.executescript("""
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, globalTid INTEGER,
                textId INTEGER, text TEXT
            );
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            """)
        global_tid = (10 << 24) | 7
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (100, 200, ?, NULL, ?)",
            (global_tid, "scheduler.run_batch"),
        )
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (110, NULL, ?, NULL, ?)",
            (global_tid, trace.BATCH_PHASE_MARKER_PREFIX + "prefill"),
        )

        self.assertEqual(
            trace.scheduler_ranges_by_pid(connection),
            {10: [(100, 200, global_tid, "prefill")]},
        )

    def test_missing_scheduler_phase_marker_fails(self):
        connection = sqlite3.connect(":memory:")
        connection.executescript("""
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, globalTid INTEGER,
                textId INTEGER, text TEXT
            );
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            INSERT INTO NVTX_EVENTS VALUES (
                100, 200, 167772167, NULL, 'scheduler.run_batch'
            );
            """)
        with self.assertRaisesRegex(ValueError, "batch_phase"):
            trace.scheduler_ranges_by_pid(connection)

    def test_prefix_imbalance_outside_markers_is_discarded(self):
        topology = {0: _topology(0, 10, 0), 1: _topology(1, 11, 1)}
        ranges = {
            10: [
                (10, 20, 1, "prefill"),
                (30, 40, 1, "decode"),
                (110, 120, 1, "prefill"),
                (130, 140, 1, "decode"),
            ],
            11: [
                (20, 30, 2, "idle"),
                (115, 125, 2, "idle"),
                (135, 145, 2, "idle"),
            ],
        }

        measured, validation = trace.filter_scheduler_ranges_to_measurement(
            ranges, topology
        )

        self.assertEqual(
            {device: len(value) for device, value in measured.items()}, {0: 2, 1: 2}
        )
        self.assertEqual(validation[0]["prefix_discarded"], 2)
        self.assertEqual(validation[1]["prefix_discarded"], 1)

    def test_rank_imbalance_inside_markers_still_fails(self):
        topology = {0: _topology(0, 10, 0), 1: _topology(1, 11, 1)}
        ranges = {
            10: [(110, 120, 1, "prefill"), (130, 140, 1, "decode")],
            11: [(115, 125, 2, "idle")],
        }
        with self.assertRaisesRegex(ValueError, "counts differ"):
            trace.filter_scheduler_ranges_to_measurement(ranges, topology)

    def test_range_crossing_marker_fails(self):
        topology = {0: _topology(0, 10, 0)}
        with self.assertRaisesRegex(ValueError, "crosses measurement boundary"):
            trace.filter_scheduler_ranges_to_measurement(
                {10: [(90, 110, 1, "prefill")]}, topology
            )


class TestTopologyAndDPSelection(unittest.TestCase):
    def test_idle_dp_rank_with_longer_span_is_not_selected(self):
        active = _rank_step(0, 0, 10)
        idle_but_longer = _rank_step(1, 0, 20)
        group = trace.StepGroup(
            index=0,
            phase="prefill",
            ranks={0: active, 1: idle_but_longer},
            all_rank_longest=idle_but_longer,
        )
        topology = {0: _topology(0, 10, 0), 1: _topology(1, 11, 1)}

        selected = trace.select_critical_step(group, 0, topology)

        self.assertEqual(selected.rank.device, 0)
        self.assertEqual(selected.all_rank_longest.device, 1)

    def test_missing_client_dp_topology_fails(self):
        topology = {0: _topology(0, 10, 0)}
        client = {"output_lens": [2], "dp_ranks": [1]}
        with self.assertRaisesRegex(ValueError, "topology contains"):
            trace.validate_client_dp_ranks(client, topology)

    def test_contradictory_dp_tp_membership_fails(self):
        connection = sqlite3.connect(":memory:")
        connection.executescript("""
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, globalTid INTEGER,
                textId INTEGER, text TEXT
            );
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (
                processId INTEGER, deviceId INTEGER, isGreenContext INTEGER
            );
            """)
        marker_specs = [
            (10, 0, 0, 0),
            (11, 1, 0, 1),
            (12, 2, 1, 0),
        ]
        for pid, device, dp_rank, attn_tp_rank in marker_specs:
            connection.execute(
                "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (?, ?, 0)",
                (pid, device),
            )
            for timestamp, phase in ((100, "begin"), (500, "end")):
                payload = {
                    "run_id": "run",
                    "phase": phase,
                    "gpu_id": device,
                    "tp_rank": device,
                    "pp_rank": 0,
                    "dp_rank": dp_rank,
                    "attn_dp_rank": dp_rank,
                    "attn_tp_rank": attn_tp_rank,
                    "moe_ep_rank": device,
                }
                connection.execute(
                    "INSERT INTO NVTX_EVENTS VALUES (?, NULL, ?, NULL, ?)",
                    (
                        timestamp,
                        (pid << 24) | 1,
                        trace.MEASUREMENT_MARKER_PREFIX
                        + json.dumps(payload, separators=(",", ":")),
                    ),
                )

        with self.assertRaisesRegex(ValueError, "inconsistent TP membership"):
            trace.load_measurement_topology(connection, "run", {10, 11, 12})


class TestRequestAttribution(unittest.TestCase):
    def test_prefill_phase_does_not_depend_on_cuda_graph_fraction(self):
        prefill_graph = replace(
            _rank_step(0, 0, 100, phase="prefill"),
            graph_kernel_count=10,
            kernel_count=10,
        )
        idle = _rank_step(1, 0, 90, phase="idle")

        groups = trace.build_step_groups({0: [prefill_graph], 1: [idle]})

        self.assertEqual(groups[0].phase, "prefill")

    def test_required_prefill_cuda_graph_rejects_eager_fallback(self):
        with self.assertRaisesRegex(ValueError, "fully eagerly.*\\[0\\]"):
            trace.validate_prefill_cuda_graph_execution(
                {0: [_rank_step(0, 0, 100, phase="prefill")]}, required=True
            )

    def test_required_prefill_cuda_graph_accepts_graph_nodes(self):
        graphed = replace(_rank_step(0, 0, 100, phase="prefill"), graph_kernel_count=1)

        device_audit, step_audit = trace.validate_prefill_cuda_graph_execution(
            {0: [graphed]}, required=True
        )

        self.assertEqual(device_audit[0]["graphed_prefill_steps"], 1)
        self.assertEqual(device_audit[0]["eager_prefill_steps"], 0)
        self.assertEqual(step_audit[0]["graphed_devices"], [0])

    def test_prefill_graph_requirement_accepts_rank_asymmetric_replay(self):
        eager_rank = _rank_step(0, 0, 100, phase="prefill")
        graphed_rank = replace(
            _rank_step(1, 0, 100, phase="prefill"), graph_kernel_count=1
        )

        device_audit, step_audit = trace.validate_prefill_cuda_graph_execution(
            {0: [eager_rank], 1: [graphed_rank]},
            required=True,
        )

        self.assertEqual(device_audit[0]["eager_prefill_steps"], 1)
        self.assertEqual(step_audit[0]["graphed_devices"], [1])
        self.assertEqual(step_audit[0]["eager_devices"], [0])

    def test_prefill_graph_requirement_checks_every_global_prefill_step(self):
        graphed_then_eager = [
            replace(_rank_step(0, 0, 100), graph_kernel_count=1),
            _rank_step(0, 100, 200),
        ]
        eager_steps = [
            _rank_step(1, 0, 100),
            _rank_step(1, 100, 200),
        ]

        with self.assertRaisesRegex(ValueError, "fully eagerly.*\\[1\\]"):
            trace.validate_prefill_cuda_graph_execution(
                {0: graphed_then_eager, 1: eager_steps},
                required=True,
            )

    def test_concurrent_requests_use_individual_dp_timelines(self):
        dp0_rank = _rank_step(0, 0, 40)
        dp1_rank = _rank_step(1, 0, 100)
        critical_steps = {
            0: [trace.CriticalStep(0, "prefill", dp0_rank, dp1_rank)],
            1: [trace.CriticalStep(0, "prefill", dp1_rank, dp1_rank)],
        }
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE TARGET_INFO_SESSION_START_TIME (systemClockNs INTEGER)"
        )
        connection.execute(
            "INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (1000000)"
        )
        client = {
            "input_lens": [1, 1],
            "output_lens": [2, 2],
            "ttfts": [0.00000005, 0.00000005],
            "itls": [[0.00000005], [0.00000005]],
            "send_times": [0.001, 0.001],
            "finish_times": [0.0010001, 0.0010001],
            "dp_ranks": [0, 1],
        }

        _, _, e2e, details, _ = trace.analyze_request_windows(
            connection, critical_steps, client
        )

        self.assertEqual([detail["dp_rank"] for detail in details], [0, 1])
        self.assertLess(e2e[0]["total"], e2e[1]["total"])

    def test_three_bucket_invariant_holds_at_every_level(self):
        rank = _rank_step(
            0,
            0,
            100,
            compute_intervals=((0, 50),),
            communication_intervals=((60, 90),),
        )
        step = trace.CriticalStep(0, "prefill", rank, rank)
        request = trace.component_totals([step])
        summarized = trace.summarize_metric([request, request])

        self.assertEqual(rank.total, rank.compute + rank.communication + rank.uncovered)
        self.assertEqual(rank.uncovered, 20)
        self.assertEqual(
            request["total"],
            request["compute"] + request["communication"] + request["uncovered"],
        )
        self.assertEqual(
            summarized["total"]["mean"],
            summarized["compute"]["mean"]
            + summarized["communication"]["mean"]
            + summarized["uncovered"]["mean"],
        )

    def test_tolerated_overlap_is_assigned_to_compute_once(self):
        rank = _rank_step(
            0,
            0,
            100,
            compute_intervals=((0, 60),),
            communication_intervals=((50, 80),),
        )

        self.assertEqual(rank.communication_kernel_active, 30)
        self.assertEqual(rank.communication_compute_overlap, 10)
        self.assertEqual(rank.communication, 20)
        self.assertEqual(rank.uncovered, 20)
        self.assertEqual(rank.total, rank.compute + rank.communication + rank.uncovered)

    def test_request_window_serializes_an_exact_additive_total(self):
        rank = _rank_step(
            0,
            0,
            1_000_003,
            compute_intervals=((0, 333_331),),
            communication_intervals=((400_003, 700_009),),
        )

        values = trace.request_window_components(
            [trace.CriticalStep(0, "prefill", rank, rank)], 0, 1_000_003
        )

        self.assertEqual(
            values["total"],
            sum(values[bucket] for bucket in ("compute", "communication", "uncovered")),
        )
        trace.summarize_metric([values])

    def test_large_request_window_checks_integer_invariant_before_conversion(self):
        compute_ns = 6_230_121_090
        communication_ns = 11_425_666_897
        uncovered_ns = 7_702_272_603
        total_ns = compute_ns + communication_ns + uncovered_ns
        rank = _rank_step(
            0,
            0,
            total_ns,
            compute_intervals=((0, compute_ns),),
            communication_intervals=((compute_ns, compute_ns + communication_ns),),
        )

        values = trace.request_window_components(
            [trace.CriticalStep(0, "decode", rank, rank)], 0, total_ns
        )

        self.assertEqual(
            values["total"],
            sum(values[bucket] for bucket in ("compute", "communication", "uncovered")),
        )
        trace.summarize_metric([values])

    def test_request_window_does_not_sum_separate_tolerated_overlaps(self):
        first = _rank_step(
            0,
            0,
            100,
            compute_intervals=((0, 60),),
            communication_intervals=((50, 80),),
        )
        second = _rank_step(
            0,
            100,
            200,
            compute_intervals=((100, 160),),
            communication_intervals=((150, 180),),
        )

        values = trace.request_window_components(
            [
                trace.CriticalStep(0, "decode", first, first),
                trace.CriticalStep(1, "decode", second, second),
            ],
            0,
            200,
            overlap_tolerance_ns=10,
        )

        self.assertAlmostEqual(values["communication_compute_overlap"], 20e-6)

    def test_unexpected_overlap_fails(self):
        rank = _rank_step(
            0,
            0,
            50_000,
            compute_intervals=((0, 40_000),),
            communication_intervals=((20_000, 50_000),),
        )
        step = trace.CriticalStep(0, "prefill", rank, rank)
        with self.assertRaisesRegex(
            ValueError,
            r"request 7 E2E window.*observed=20000 ns \(20\.000 us\).*"
            r"configured_threshold=10000 ns \(10\.000 us\).*"
            r"excess=10000 ns \(10\.000 us\)",
        ):
            trace.request_window_components(
                [step],
                0,
                50_000,
                overlap_tolerance_ns=10_000,
                context="request 7 E2E window",
            )

    def test_unrecognized_collective_name_fails(self):
        with self.assertRaisesRegex(ValueError, "collective-looking"):
            trace.is_communication_kernel("future_allreduce_kernel", frozenset())

    def test_known_pull_all_reduce_is_communication(self):
        self.assertTrue(
            trace.is_communication_kernel("all_reduce_one_shot_kernel", frozenset())
        )

    def test_all_custom_all_reduce_collective_kernels_are_allowlisted(self):
        source = CUSTOM_ALL_REDUCE_HEADER.read_text()
        declared_kernels = frozenset(
            re.findall(
                r"ALL_REDUCE_KERNEL\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                source,
            )
        )

        self.assertEqual(
            declared_kernels,
            trace.CUSTOM_ALL_REDUCE_KERNEL_NAMES,
            "Review every CUDA kernel added to custom_all_reduce.cuh before "
            "updating the communication-kernel allowlist.",
        )
        for name in declared_kernels:
            with self.subTest(name=name):
                self.assertTrue(trace.is_communication_kernel(name, frozenset()))
        self.assertFalse(trace.is_communication_kernel("memcpy_kernel", frozenset()))


class TestSummaryWording(unittest.TestCase):
    def _metric(self, communication_baseline, communication_purlin):
        return {
            "clean_client": {"baseline_ms": 10.0, "purlin_ms": 9.0},
            "trace_model": {
                "total": {"baseline_ms": 8.0, "purlin_ms": 8.0},
                "compute": {"baseline_ms": 5.0, "purlin_ms": 5.5},
                "communication": {
                    "baseline_ms": communication_baseline,
                    "purlin_ms": communication_purlin,
                },
                "uncovered": {"baseline_ms": 1.0, "purlin_ms": 0.5},
            },
        }

    def test_communication_increase_is_never_called_reduction(self):
        text = summary.describe_change("Communication", 3.0, 4.0)
        self.assertIn("increased", text)
        self.assertNotIn("reduction", text)

    def test_mixed_directions_are_described_independently(self):
        metrics = {
            "ttft": self._metric(3.0, 2.0),
            "tpot": self._metric(3.0, 4.0),
            "e2e": self._metric(3.0, 3.0),
        }
        observations = summary.build_observations(metrics)
        rendered = json.dumps(observations)
        self.assertIn("decreased", observations["ttft"]["trace_model"]["communication"])
        self.assertIn("increased", observations["tpot"]["trace_model"]["communication"])
        self.assertIn("unchanged", observations["e2e"]["trace_model"]["communication"])
        self.assertNotIn("consistent", rendered)


if __name__ == "__main__":
    unittest.main()
