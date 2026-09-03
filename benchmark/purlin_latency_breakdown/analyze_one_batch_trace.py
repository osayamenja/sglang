#!/usr/bin/env python3

"""Summarize one captured ``benchmark.one_batch`` forward pass.

Unlike ``analyze_full_trace.py``, this analyzer intentionally has no request or
scheduler assumptions.  The Nsight capture range is one static prefill or
decode forward, so it reports the union of compute and communication kernels on
each device and selects the longest device span as the critical path.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from pathlib import Path

from analyze_full_trace import (
    COMMUNICATION_KERNEL_NAMES,
    interval_intersection_length,
    interval_union_length,
    is_communication_kernel,
    merge_intervals,
)


def _kernel_rows(connection: sqlite3.Connection) -> list[tuple]:
    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)"
        ).fetchall()
    }
    if not columns:
        raise ValueError("Nsight export has no CUPTI_ACTIVITY_KIND_KERNEL table")
    graph_id = "kernels.graphId" if "graphId" in columns else "NULL"
    return connection.execute(f"""
        SELECT
            kernels.deviceId,
            kernels.start,
            kernels.end,
            strings.value,
            {graph_id},
            kernels.streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
        JOIN StringIds AS strings ON strings.id = kernels.shortName
        ORDER BY kernels.deviceId, kernels.start
        """).fetchall()


def analyze_trace(
    connection: sqlite3.Connection,
    *,
    label: str,
    stage: str,
    extra_communication_kernel_names: frozenset[str] = frozenset(),
) -> dict:
    rows = _kernel_rows(connection)
    if not rows:
        raise ValueError("Nsight export contains no CUDA kernels")

    rows_by_device: dict[int, list[tuple]] = collections.defaultdict(list)
    communication_kernel_totals: dict[str, list[int]] = collections.defaultdict(
        lambda: [0, 0]
    )
    for device, start, end, name, graph_id, stream_id in rows:
        rows_by_device[int(device)].append(
            (int(start), int(end), str(name), graph_id, int(stream_id))
        )

    device_summaries = []
    for device, device_rows in sorted(rows_by_device.items()):
        all_intervals = []
        compute_intervals = []
        communication_intervals = []
        graph_kernel_count = 0
        streams = set()
        for start, end, name, graph_id, stream_id in device_rows:
            interval = (start, end)
            all_intervals.append(interval)
            streams.add(stream_id)
            graph_kernel_count += int(graph_id is not None)
            if is_communication_kernel(name, extra_communication_kernel_names):
                communication_intervals.append(interval)
                communication_kernel_totals[name][0] += 1
                communication_kernel_totals[name][1] += end - start
            else:
                compute_intervals.append(interval)

        compute_intervals = merge_intervals(compute_intervals)
        communication_intervals = merge_intervals(communication_intervals)
        span = max(end for _, end in all_intervals) - min(
            start for start, _ in all_intervals
        )
        compute_active = interval_union_length(compute_intervals)
        communication_active = interval_union_length(communication_intervals)
        overlap = interval_intersection_length(
            compute_intervals, communication_intervals
        )
        coverage = interval_union_length(all_intervals)
        communication_exclusive = communication_active - overlap
        uncovered = span - coverage
        additive_total = compute_active + communication_exclusive + uncovered
        device_summaries.append(
            {
                "device": device,
                "span_ms": span / 1_000_000,
                "compute_active_ms": compute_active / 1_000_000,
                "communication_active_ms": communication_active / 1_000_000,
                "communication_exclusive_ms": communication_exclusive / 1_000_000,
                "compute_communication_overlap_ms": overlap / 1_000_000,
                "uncovered_ms": uncovered / 1_000_000,
                "kernel_coverage_ms": coverage / 1_000_000,
                "additive_error_ns": additive_total - span,
                "kernel_count": len(device_rows),
                "graph_kernel_count": graph_kernel_count,
                "stream_count": len(streams),
            }
        )

    critical_device = max(device_summaries, key=lambda item: item["span_ms"])
    return {
        "schema_version": 1,
        "label": label,
        "stage": stage,
        "scope": "one static ModelRunner forward captured by CUDA Profiler API",
        "critical_device": critical_device["device"],
        "critical_path": critical_device,
        "devices": device_summaries,
        "communication_kernels": [
            {
                "name": name,
                "count": count,
                "summed_device_time_ms": duration_ns / 1_000_000,
            }
            for name, (count, duration_ns) in sorted(
                communication_kernel_totals.items(),
                key=lambda item: (-item[1][1], item[0]),
            )
        ],
        "classification": {
            "communication_kernel_prefixes": ["ncclDevKernel_"],
            "communication_kernel_names": sorted(
                COMMUNICATION_KERNEL_NAMES | extra_communication_kernel_names
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--stage", choices=("prefill", "decode"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--communication-pattern",
        action="append",
        default=[],
        help="Additional reviewed exact communication-kernel name.",
    )
    args = parser.parse_args()

    with sqlite3.connect(args.sqlite) as connection:
        result = analyze_trace(
            connection,
            label=args.label,
            stage=args.stage,
            extra_communication_kernel_names=frozenset(args.communication_pattern),
        )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    critical = result["critical_path"]
    print(
        f"{args.label} {args.stage}: device {critical['device']}, "
        f"span={critical['span_ms']:.3f} ms, "
        f"compute={critical['compute_active_ms']:.3f} ms, "
        f"communication={critical['communication_exclusive_ms']:.3f} ms, "
        f"overlap={critical['compute_communication_overlap_ms']:.3f} ms"
    )


if __name__ == "__main__":
    main()
