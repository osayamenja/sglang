import json
import sqlite3
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[3] / "benchmark" / "purlin_latency_breakdown"
)
sys.path.insert(0, str(BENCHMARK_DIR))

import analyze_full_trace as trace  # noqa: E402
import summarize_results as summary  # noqa: E402

register_cpu_ci(est_time=8, suite="base-c-test-cpu")


def _rank_step(
    device,
    start,
    end,
    *,
    compute_intervals=None,
    communication_intervals=None,
):
    compute_intervals = tuple(compute_intervals or ((start, end),))
    communication_intervals = tuple(communication_intervals or ())
    total = end - start
    compute = trace.interval_union_length(compute_intervals)
    return trace.RankStep(
        device=device,
        start=start,
        end=end,
        total=total,
        compute=compute,
        communication=total - compute,
        communication_kernel_active=trace.interval_union_length(
            communication_intervals
        ),
        communication_compute_overlap=trace.interval_intersection_length(
            communication_intervals, compute_intervals
        ),
        kernel_coverage=trace.interval_union_length(
            (*compute_intervals, *communication_intervals)
        ),
        kernel_count=max(1, len(compute_intervals) + len(communication_intervals)),
        graph_kernel_count=0,
        stream_count=1,
        compute_intervals=compute_intervals,
        communication_intervals=communication_intervals,
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
    def test_prefix_imbalance_outside_markers_is_discarded(self):
        topology = {0: _topology(0, 10, 0), 1: _topology(1, 11, 1)}
        ranges = {
            10: [(10, 20, 1), (30, 40, 1), (110, 120, 1), (130, 140, 1)],
            11: [(20, 30, 2), (115, 125, 2), (135, 145, 2)],
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
            10: [(110, 120, 1), (130, 140, 1)],
            11: [(115, 125, 2)],
        }
        with self.assertRaisesRegex(ValueError, "counts differ"):
            trace.filter_scheduler_ranges_to_measurement(ranges, topology)

    def test_range_crossing_marker_fails(self):
        topology = {0: _topology(0, 10, 0)}
        with self.assertRaisesRegex(ValueError, "crosses measurement boundary"):
            trace.filter_scheduler_ranges_to_measurement({10: [(90, 110, 1)]}, topology)


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

    def test_two_bucket_invariant_holds_at_every_level(self):
        rank = _rank_step(
            0,
            0,
            100,
            compute_intervals=((0, 60),),
            communication_intervals=((60, 100),),
        )
        step = trace.CriticalStep(0, "prefill", rank, rank)
        request = trace.component_totals([step])
        summarized = trace.summarize_metric([request, request])

        self.assertEqual(rank.total, rank.compute + rank.communication)
        self.assertEqual(
            request["total"], request["compute"] + request["communication"]
        )
        self.assertEqual(
            summarized["total"]["mean"],
            summarized["compute"]["mean"] + summarized["communication"]["mean"],
        )

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
