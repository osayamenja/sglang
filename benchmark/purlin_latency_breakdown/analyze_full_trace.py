#!/usr/bin/env python3

"""Extract additive TTFT, TPOT, and E2E GPU-model breakdowns from Nsight.

The trace must come from an SGLang serving run with scheduler NVTX enabled and
CUDA graph nodes recorded. Each ``scheduler.run_batch`` range is mapped to the
kernels it launched through CUDA runtime correlation IDs.

Per-scheduler begin/end markers delimit measured ranges and carry the topology
used to map every scheduler PID to an attention-DP group. At concurrency one,
prefill/decode phase groups are assigned directly to each request. At higher
concurrency, benchmark request timestamps are converted to the Nsight session
clock. In both modes, each request uses only the critical timeline for its
returned DP rank.

The public model has three additive buckets:

* compute is the union of non-communication kernel intervals on the selected
  rank for a scheduler step;
* communication is communication-kernel-active time not overlapping compute;
* uncovered is time inside the selected step's first-to-last-kernel span when
  no classified GPU kernel is active.

Small cross-stream timestamp overlap is assigned to compute, while raw
communication-kernel activity and overlap remain available in diagnostics.
This guarantees ``total = compute + communication + uncovered`` at every
aggregation level without claiming that uncovered time is communication.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MEASUREMENT_MARKER_PREFIX = "sglang.measurement:"
BATCH_PHASE_MARKER_PREFIX = "sglang.batch_phase:"
CUSTOM_ALL_REDUCE_KERNEL_NAMES = frozenset(
    {
        "all_reduce_1shot_push_kernel",
        "all_reduce_1shot_pull_kernel",
        "all_reduce_2shot_pull_kernel",
    }
)
COMMUNICATION_KERNEL_NAMES = (
    frozenset(
        {
            "all_reduce_one_shot_push_kernel",
            "all_reduce_one_shot_kernel",
            "all_reduce_two_shot_kernel",
            "_all_gather_kernel_inner",
            "allReduceKernel",
            "allGatherKernel",
            "reduceScatterKernel",
        }
    )
    | CUSTOM_ALL_REDUCE_KERNEL_NAMES
)
COLLECTIVE_NAME_WARNING_TERMS = (
    "nccl",
    "allreduce",
    "all_reduce",
    "allgather",
    "all_gather",
    "reducescatter",
    "reduce_scatter",
)
# Nsight timestamps can put adjacent cross-stream kernels a few microseconds
# across one another. The two-GPU A100 smoke traces peaked at 1.952 us for
# baseline and 6.304 us for Purlin. The 25 us default leaves headroom for larger
# topologies while still rejecting meaningful concurrent execution. Keep it a
# CLI option so a hardware-specific calibration can tighten it.
DEFAULT_COMMUNICATION_COMPUTE_OVERLAP_TOLERANCE_NS = 25_000
SchedulerRange = tuple[int, int, int, str]


@dataclass
class RankStep:
    device: int
    start: int
    end: int
    total: int
    compute: int
    communication: int
    uncovered: int
    communication_kernel_active: int
    communication_compute_overlap: int
    kernel_coverage: int
    kernel_count: int
    graph_kernel_count: int
    stream_count: int
    compute_intervals: tuple[tuple[int, int], ...]
    communication_intervals: tuple[tuple[int, int], ...]
    phase: str


@dataclass
class CriticalStep:
    index: int
    phase: str
    rank: RankStep
    all_rank_longest: RankStep


@dataclass(frozen=True)
class SchedulerTopology:
    pid: int
    device: int
    begin: int
    end: int
    gpu_id: int
    tp_rank: int
    pp_rank: int
    dp_rank: int
    attn_dp_rank: int
    attn_tp_rank: int
    moe_ep_rank: int


@dataclass
class StepGroup:
    index: int
    phase: str
    ranks: dict[int, RankStep]
    all_rank_longest: RankStep


def merge_intervals(
    intervals: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    ordered = sorted(intervals)
    if not ordered:
        return ()
    left, right = ordered[0]
    merged: list[tuple[int, int]] = []
    for next_left, next_right in ordered[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            merged.append((left, right))
            left, right = next_left, next_right
    merged.append((left, right))
    return tuple(merged)


def interval_union_length(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(right - left for left, right in merge_intervals(intervals))


def interval_intersection_length(
    first: Iterable[tuple[int, int]], second: Iterable[tuple[int, int]]
) -> int:
    return sum(
        right - left for left, right in interval_intersection_segments(first, second)
    )


def interval_intersection_segments(
    first: Iterable[tuple[int, int]], second: Iterable[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    """Return disjoint intervals where both input interval unions are active."""

    first = merge_intervals(first)
    second = merge_intervals(second)
    first_index = second_index = 0
    intersections: list[tuple[int, int]] = []
    while first_index < len(first) and second_index < len(second):
        first_left, first_right = first[first_index]
        second_left, second_right = second[second_index]
        left = max(first_left, second_left)
        right = min(first_right, second_right)
        if left < right:
            intersections.append((left, right))
        if first_right < second_right:
            first_index += 1
        else:
            second_index += 1
    return tuple(intersections)


def validate_communication_compute_overlap(
    observed_ns: int, tolerance_ns: int, context: str
) -> None:
    """Reject meaningful overlap with an actionable calibration error."""

    if tolerance_ns < 0:
        raise ValueError(f"Overlap tolerance must be non-negative, got {tolerance_ns}")
    if observed_ns <= tolerance_ns:
        return
    excess_ns = observed_ns - tolerance_ns
    raise ValueError(
        f"Compute/communication overlap validation failed for {context}: "
        f"observed={observed_ns} ns ({observed_ns / 1_000:.3f} us), "
        f"configured_threshold={tolerance_ns} ns "
        f"({tolerance_ns / 1_000:.3f} us), "
        f"excess={excess_ns} ns ({excess_ns / 1_000:.3f} us)"
    )


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def load_last_json_line(path: Path) -> dict:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No JSON objects found in {path}")
    return json.loads(lines[-1])


def table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def create_analysis_indexes(connection: sqlite3.Connection) -> None:
    """Index the derived Nsight SQLite export for request-step lookups."""

    print(
        "Creating/reusing analysis indexes in the derived SQLite export...",
        file=sys.stderr,
    )
    connection.execute("""
        CREATE INDEX IF NOT EXISTS purlin_analysis_runtime_tid_start
        ON CUPTI_ACTIVITY_KIND_RUNTIME(globalTid, start, correlationId)
        """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS purlin_analysis_kernel_device_correlation
        ON CUPTI_ACTIVITY_KIND_KERNEL(deviceId, correlationId)
        """)
    connection.commit()


def scheduler_ranges_by_pid(
    connection: sqlite3.Connection,
) -> dict[int, list[SchedulerRange]]:
    rows = connection.execute("""
        SELECT
            events.start,
            events.end,
            events.globalTid,
            (events.globalTid >> 24) & 0x00FFFFFF AS pid
        FROM NVTX_EVENTS AS events
        LEFT JOIN StringIds AS strings ON strings.id = events.textId
        WHERE coalesce(strings.value, events.text) = 'scheduler.run_batch'
          AND events.end IS NOT NULL
        ORDER BY pid, events.start
        """).fetchall()
    if not rows:
        raise ValueError(
            "No scheduler.run_batch NVTX ranges found. Start the server with "
            "SGLANG_ENABLE_NVTX_SCHEDULER=1."
        )

    phase_rows = connection.execute(
        """
        SELECT
            events.start,
            events.globalTid,
            (events.globalTid >> 24) & 0x00FFFFFF AS pid,
            coalesce(strings.value, events.text) AS marker_text
        FROM NVTX_EVENTS AS events
        LEFT JOIN StringIds AS strings ON strings.id = events.textId
        WHERE coalesce(strings.value, events.text) LIKE ?
        ORDER BY pid, events.start
        """,
        (BATCH_PHASE_MARKER_PREFIX + "%",),
    ).fetchall()
    if not phase_rows:
        raise ValueError(
            f"No {BATCH_PHASE_MARKER_PREFIX} markers found. Rerun the trace "
            "with a server that emits explicit scheduler batch phases."
        )

    markers_by_tid: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
    for timestamp, global_tid, pid, marker_text in phase_rows:
        phase = marker_text.removeprefix(BATCH_PHASE_MARKER_PREFIX)
        if phase not in ("prefill", "decode", "idle"):
            raise ValueError(
                f"Scheduler pid {pid} emitted invalid batch phase {phase!r}"
            )
        markers_by_tid[int(global_tid)].append((int(timestamp), phase))

    ranges_by_pid: dict[int, list[SchedulerRange]] = collections.defaultdict(list)
    used_markers: set[tuple[int, int]] = set()
    for start, end, global_tid, pid in rows:
        matches = [
            (timestamp, phase)
            for timestamp, phase in markers_by_tid.get(int(global_tid), [])
            if start <= timestamp <= end
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one scheduler batch-phase marker inside pid {pid} "
                f"range [{start}, {end}], found {matches}"
            )
        timestamp, phase = matches[0]
        used_markers.add((int(global_tid), timestamp))
        ranges_by_pid[pid].append((start, end, global_tid, phase))

    all_markers = {
        (global_tid, timestamp)
        for global_tid, markers in markers_by_tid.items()
        for timestamp, _ in markers
    }
    unmatched_markers = all_markers - used_markers
    if unmatched_markers:
        raise ValueError(
            "Scheduler batch-phase markers were not contained by complete "
            f"scheduler.run_batch ranges: {sorted(unmatched_markers)}"
        )
    return dict(sorted(ranges_by_pid.items()))


def cuda_device_for_pid(connection: sqlite3.Connection, pid: int) -> int:
    devices = [
        row[0]
        for row in connection.execute(
            """
            SELECT DISTINCT deviceId
            FROM TARGET_INFO_CUDA_CONTEXT_INFO
            WHERE processId = ? AND coalesce(isGreenContext, 0) = 0
            """,
            (pid,),
        )
    ]
    if len(devices) != 1:
        raise ValueError(
            f"Expected one CUDA device for scheduler pid {pid}, found {devices}"
        )
    return int(devices[0])


def _require_marker_int(payload: dict, field: str, pid: int) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"Measurement marker for pid {pid} has invalid {field}: {value!r}"
        )
    return value


def load_measurement_topology(
    connection: sqlite3.Connection,
    run_id: str,
    scheduler_pids: set[int],
) -> dict[int, SchedulerTopology]:
    rows = connection.execute(
        """
        SELECT
            events.start,
            events.globalTid,
            (events.globalTid >> 24) & 0x00FFFFFF AS pid,
            coalesce(strings.value, events.text) AS marker_text
        FROM NVTX_EVENTS AS events
        LEFT JOIN StringIds AS strings ON strings.id = events.textId
        WHERE coalesce(strings.value, events.text) LIKE ?
        ORDER BY pid, events.start
        """,
        (MEASUREMENT_MARKER_PREFIX + "%",),
    ).fetchall()
    if not rows:
        raise ValueError(f"No {MEASUREMENT_MARKER_PREFIX} markers found")

    markers: dict[int, dict[str, tuple[int, dict]]] = collections.defaultdict(dict)
    for timestamp, global_tid, pid, marker_text in rows:
        if global_tid is None or pid is None:
            raise ValueError("Measurement marker is missing scheduler thread metadata")
        encoded = marker_text[len(MEASUREMENT_MARKER_PREFIX) :]
        try:
            payload = json.loads(encoded)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Malformed measurement marker for pid {pid}: {encoded!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Measurement marker for pid {pid} is not a JSON object")
        if payload.get("run_id") != run_id:
            continue
        phase = payload.get("phase")
        if phase not in ("begin", "end"):
            raise ValueError(
                f"Measurement marker for pid {pid} has invalid phase {phase!r}"
            )
        if phase in markers[pid]:
            raise ValueError(
                f"Duplicate measurement {phase} marker for run {run_id} pid {pid}"
            )
        markers[pid][phase] = (int(timestamp), payload)

    marker_pids = set(markers)
    missing = scheduler_pids - marker_pids
    unexpected = marker_pids - scheduler_pids
    if missing or unexpected:
        raise ValueError(
            "Measurement marker scheduler PIDs do not match scheduler ranges: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    topology: dict[int, SchedulerTopology] = {}
    seen_devices: dict[int, int] = {}
    seen_coordinates: dict[tuple[int, ...], int] = {}
    metadata_fields = (
        "gpu_id",
        "tp_rank",
        "pp_rank",
        "dp_rank",
        "attn_dp_rank",
        "attn_tp_rank",
        "moe_ep_rank",
    )
    for pid in sorted(scheduler_pids):
        phases = markers[pid]
        if set(phases) != {"begin", "end"}:
            raise ValueError(
                f"Expected one begin and end marker for pid {pid}, found {sorted(phases)}"
            )
        begin, begin_payload = phases["begin"]
        end, end_payload = phases["end"]
        if begin >= end:
            raise ValueError(
                f"Measurement markers are incorrectly ordered for pid {pid}: "
                f"begin={begin}, end={end}"
            )
        begin_metadata = {
            field: _require_marker_int(begin_payload, field, pid)
            for field in metadata_fields
        }
        end_metadata = {
            field: _require_marker_int(end_payload, field, pid)
            for field in metadata_fields
        }
        if begin_metadata != end_metadata:
            raise ValueError(
                f"Begin/end topology metadata conflicts for scheduler pid {pid}: "
                f"{begin_metadata} != {end_metadata}"
            )

        device = cuda_device_for_pid(connection, pid)
        if device in seen_devices:
            raise ValueError(
                f"Scheduler pids {seen_devices[device]} and {pid} both map to device {device}"
            )
        seen_devices[device] = pid
        coordinate = tuple(begin_metadata[field] for field in metadata_fields)
        if coordinate in seen_coordinates:
            raise ValueError(
                f"Scheduler pids {seen_coordinates[coordinate]} and {pid} report "
                f"duplicate topology {coordinate}"
            )
        seen_coordinates[coordinate] = pid
        topology[device] = SchedulerTopology(
            pid=pid,
            device=device,
            begin=begin,
            end=end,
            **begin_metadata,
        )

    membership: dict[int, set[tuple[int, int]]] = collections.defaultdict(set)
    membership_counts: collections.Counter[tuple[int, int, int]] = collections.Counter()
    for item in topology.values():
        member = (item.pp_rank, item.attn_tp_rank)
        membership[item.attn_dp_rank].add(member)
        membership_counts[(item.attn_dp_rank, *member)] += 1
    duplicates = {key: count for key, count in membership_counts.items() if count != 1}
    if duplicates:
        raise ValueError(
            f"Duplicate TP membership within attention-DP groups: {duplicates}"
        )
    memberships = list(membership.values())
    if memberships and any(value != memberships[0] for value in memberships[1:]):
        raise ValueError(
            "Attention-DP groups have inconsistent TP membership: "
            + ", ".join(
                f"dp{rank}={sorted(value)}"
                for rank, value in sorted(membership.items())
            )
        )
    return dict(sorted(topology.items()))


def filter_scheduler_ranges_to_measurement(
    ranges_by_pid: dict[int, list[SchedulerRange]],
    topology: dict[int, SchedulerTopology],
) -> tuple[dict[int, list[SchedulerRange]], dict[int, dict[str, int]]]:
    topology_by_pid = {item.pid: item for item in topology.values()}
    ranges_by_device: dict[int, list[SchedulerRange]] = {}
    validation: dict[int, dict[str, int]] = {}
    for pid, ranges in ranges_by_pid.items():
        item = topology_by_pid[pid]
        measured: list[SchedulerRange] = []
        prefix = suffix = 0
        for start, end, global_tid, phase in ranges:
            if end <= item.begin:
                prefix += 1
            elif start >= item.end:
                suffix += 1
            elif start >= item.begin and end <= item.end:
                measured.append((start, end, global_tid, phase))
            else:
                raise ValueError(
                    f"scheduler.run_batch range [{start}, {end}] for pid {pid} "
                    f"crosses measurement boundary [{item.begin}, {item.end}]"
                )
        ranges_by_device[item.device] = measured
        validation[item.device] = {
            "raw": len(ranges),
            "prefix_discarded": prefix,
            "suffix_discarded": suffix,
            "measured": len(measured),
        }

    counts = {device: len(ranges) for device, ranges in ranges_by_device.items()}
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"No measured scheduler steps for one or more ranks: {counts}")
    if len(set(counts.values())) != 1:
        raise ValueError(
            f"Measured scheduler-step counts differ across ranks: {counts}"
        )
    return dict(sorted(ranges_by_device.items())), dict(sorted(validation.items()))


def is_communication_kernel(name: str, extra_names: frozenset[str]) -> bool:
    if name.startswith("ncclDevKernel_") or name in COMMUNICATION_KERNEL_NAMES:
        return True
    if name in extra_names:
        return True
    normalized = name.lower()
    if any(term in normalized for term in COLLECTIVE_NAME_WARNING_TERMS):
        raise ValueError(
            "Unrecognized collective-looking CUDA kernel name; add an explicit "
            f"reviewed allowlist entry before measuring it: {name!r}"
        )
    return False


def load_rank_step(
    connection: sqlite3.Connection,
    device: int,
    scheduler_range: SchedulerRange,
    extra_communication_kernel_names: frozenset[str],
    communication_kernel_totals: dict[str, list[int]],
    overlap_tolerance_ns: int = (DEFAULT_COMMUNICATION_COMPUTE_OVERLAP_TOLERANCE_NS),
) -> RankStep:
    range_start, range_end, global_tid, phase = scheduler_range
    rows = connection.execute(
        """
        SELECT
            kernels.start,
            kernels.end,
            strings.value,
            kernels.graphId,
            kernels.streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
        JOIN StringIds AS strings ON strings.id = kernels.shortName
        WHERE kernels.deviceId = ?
          AND kernels.correlationId IN (
              SELECT DISTINCT runtime.correlationId
              FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
              WHERE runtime.globalTid = ?
                AND runtime.start >= ?
                AND runtime.start <= ?
          )
        """,
        (device, global_tid, range_start, range_end),
    ).fetchall()
    if not rows:
        raise ValueError(
            f"No CUDA kernels mapped to device {device} scheduler range "
            f"[{range_start}, {range_end}]"
        )

    all_intervals: list[tuple[int, int]] = []
    compute_intervals_raw: list[tuple[int, int]] = []
    communication_intervals_raw: list[tuple[int, int]] = []
    graph_kernel_count = 0
    streams: set[int] = set()
    for start, end, name, graph_id, stream_id in rows:
        interval = (start, end)
        all_intervals.append(interval)
        streams.add(stream_id)
        if graph_id is not None:
            graph_kernel_count += 1
        if is_communication_kernel(name, extra_communication_kernel_names):
            communication_intervals_raw.append(interval)
            communication_kernel_totals[name][0] += 1
            communication_kernel_totals[name][1] += end - start
        else:
            compute_intervals_raw.append(interval)

    compute_intervals = merge_intervals(compute_intervals_raw)
    communication_intervals = merge_intervals(communication_intervals_raw)
    start = min(interval[0] for interval in all_intervals)
    end = max(interval[1] for interval in all_intervals)
    total = end - start
    compute = sum(right - left for left, right in compute_intervals)
    communication_active = sum(right - left for left, right in communication_intervals)
    coverage = interval_union_length(all_intervals)
    communication_compute_overlap = interval_intersection_length(
        communication_intervals, compute_intervals
    )
    validate_communication_compute_overlap(
        communication_compute_overlap,
        overlap_tolerance_ns,
        (
            f"scheduler step on device {device} in scheduler.run_batch "
            f"range [{range_start}, {range_end}]"
        ),
    )
    # Attribute tolerated overlap to compute exactly once. Communication is
    # therefore exclusive communication-kernel activity, while the raw active
    # duration remains available in ``communication_kernel_active``.
    communication = communication_active - communication_compute_overlap
    uncovered = total - coverage
    if communication < 0 or uncovered < 0:
        raise AssertionError(
            "Internal kernel interval accounting produced a negative bucket"
        )
    if total != compute + communication + uncovered:
        raise AssertionError("Internal three-bucket arithmetic invariant failed")
    return RankStep(
        device=device,
        start=start,
        end=end,
        total=total,
        compute=compute,
        communication=communication,
        uncovered=uncovered,
        communication_kernel_active=communication_active,
        communication_compute_overlap=communication_compute_overlap,
        kernel_coverage=coverage,
        kernel_count=len(rows),
        graph_kernel_count=graph_kernel_count,
        stream_count=len(streams),
        compute_intervals=compute_intervals,
        communication_intervals=communication_intervals,
        phase=phase,
    )


def build_step_groups(rank_steps: dict[int, list[RankStep]]) -> list[StepGroup]:
    step_count = next(iter({len(steps) for steps in rank_steps.values()}))
    step_groups: list[StepGroup] = []
    for index in range(step_count):
        candidates = [steps[index] for steps in rank_steps.values()]
        active_phases = {step.phase for step in candidates if step.phase != "idle"}
        if not active_phases:
            raise ValueError(f"Scheduler step {index} is idle on every rank")
        phase = next(iter(active_phases)) if len(active_phases) == 1 else "mixed"
        step_groups.append(
            StepGroup(
                index=index,
                phase=phase,
                ranks={step.device: step for step in candidates},
                all_rank_longest=max(candidates, key=lambda step: step.total),
            )
        )
    return step_groups


def validate_prefill_cuda_graph_execution(
    rank_steps: dict[int, list[RankStep]],
    *,
    required: bool,
) -> tuple[
    dict[int, dict[str, int]],
    dict[int, dict[str, list[int]]],
]:
    """Audit graph replay per device and per global prefill scheduler step."""

    device_audit: dict[int, dict[str, int]] = {}
    for device, steps in sorted(rank_steps.items()):
        active_prefill = [
            (index, step) for index, step in enumerate(steps) if step.phase == "prefill"
        ]
        graphed = [
            (index, step)
            for index, step in active_prefill
            if step.graph_kernel_count > 0
        ]
        eager = [
            (index, step)
            for index, step in active_prefill
            if step.graph_kernel_count == 0
        ]
        device_audit[device] = {
            "active_prefill_steps": len(active_prefill),
            "graphed_prefill_steps": len(graphed),
            "eager_prefill_steps": len(eager),
        }

    prefill_step_indices = sorted(
        {
            index
            for steps in rank_steps.values()
            for index, step in enumerate(steps)
            if step.phase == "prefill"
        }
    )
    step_audit: dict[int, dict[str, list[int]]] = {}
    fully_eager_steps: list[int] = []
    for index in prefill_step_indices:
        prefill_devices = [
            device
            for device, steps in sorted(rank_steps.items())
            if steps[index].phase == "prefill"
        ]
        graphed_devices = [
            device
            for device in prefill_devices
            if rank_steps[device][index].graph_kernel_count > 0
        ]
        eager_devices = [
            device
            for device in prefill_devices
            if rank_steps[device][index].graph_kernel_count == 0
        ]
        step_audit[index] = {
            "prefill_devices": prefill_devices,
            "graphed_devices": graphed_devices,
            "eager_devices": eager_devices,
        }
        if not graphed_devices:
            fully_eager_steps.append(index)

    if required and not prefill_step_indices:
        raise ValueError(
            "Prefill CUDA graph execution was required, but the measured "
            "interval contains no active prefill steps"
        )
    if required and fully_eager_steps:
        raise ValueError(
            "Prefill CUDA graph execution was required, but measured prefill "
            "scheduler steps ran fully eagerly on every device: "
            f"{fully_eager_steps}"
        )
    return device_audit, step_audit


def select_critical_step(
    group: StepGroup,
    dp_rank: int,
    topology: dict[int, SchedulerTopology],
) -> CriticalStep:
    candidates = [
        step
        for device, step in group.ranks.items()
        if topology[device].attn_dp_rank == dp_rank
    ]
    if not candidates:
        raise ValueError(
            f"Scheduler step {group.index} has no device for attention-DP rank {dp_rank}"
        )
    active_phases = {step.phase for step in candidates if step.phase != "idle"}
    if len(active_phases) > 1:
        raise ValueError(
            f"Scheduler step {group.index} has conflicting phases within "
            f"attention-DP rank {dp_rank}: {sorted(active_phases)}"
        )
    phase = next(iter(active_phases)) if active_phases else "idle"
    return CriticalStep(
        index=group.index,
        phase=phase,
        rank=max(candidates, key=lambda step: step.total),
        all_rank_longest=group.all_rank_longest,
    )


def critical_steps_for_dp(
    step_groups: Iterable[StepGroup],
    dp_rank: int,
    topology: dict[int, SchedulerTopology],
) -> list[CriticalStep]:
    return [select_critical_step(group, dp_rank, topology) for group in step_groups]


def split_requests(
    step_groups: list[StepGroup], output_lens: list[int]
) -> list[list[StepGroup]]:
    requests: list[list[StepGroup]] = []
    current: list[StepGroup] = []
    for step in step_groups:
        if step.phase == "prefill" and current and current[-1].phase == "decode":
            requests.append(current)
            current = []
        current.append(step)
    if current:
        requests.append(current)

    if len(requests) != len(output_lens):
        raise ValueError(
            f"Trace contains {len(requests)} request phase groups, but client "
            f"recorded {len(output_lens)} requests"
        )
    for request_index, (steps, output_len) in enumerate(zip(requests, output_lens)):
        phases = [step.phase for step in steps]
        prefill_count = phases.count("prefill")
        decode_count = phases.count("decode")
        if not prefill_count or phases[:prefill_count] != ["prefill"] * prefill_count:
            raise ValueError(
                f"Request {request_index} does not start with prefill steps"
            )
        if phases[prefill_count:] != ["decode"] * decode_count:
            raise ValueError(
                f"Request {request_index} has interleaved prefill/decode steps"
            )
        # With SGLang's default overlap scheduler, one final graph replay can be
        # launched before the preceding result reveals that max_new_tokens was
        # reached. That replay drains the overlap pipeline and is not on the
        # client E2E path. Non-overlap scheduling has exactly output_len - 1
        # decode steps because the first token comes from prefill.
        valid_decode_counts = {max(0, output_len - 1), output_len}
        if decode_count not in valid_decode_counts:
            raise ValueError(
                f"Request {request_index} has {decode_count} complete decode steps; "
                f"expected one of {sorted(valid_decode_counts)} for output length "
                f"{output_len}"
            )
    return requests


def component_totals(steps: Iterable[CriticalStep]) -> dict[str, float]:
    steps = list(steps)
    total_ns = sum(step.rank.total for step in steps)
    compute_ns = sum(step.rank.compute for step in steps)
    communication_ns = sum(step.rank.communication for step in steps)
    uncovered_ns = sum(step.rank.uncovered for step in steps)
    if total_ns != compute_ns + communication_ns + uncovered_ns:
        raise AssertionError(
            "Request components violate total = compute + communication + uncovered"
        )

    ns_to_ms = 1e-6
    compute_ms = compute_ns * ns_to_ms
    communication_ms = communication_ns * ns_to_ms
    uncovered_ms = uncovered_ns * ns_to_ms
    result = {
        # Build total from the converted buckets as well, so serialized
        # request values retain the additive invariant bit-for-bit.
        "total": sum((compute_ms, communication_ms, uncovered_ms)),
        "compute": compute_ms,
        "communication": communication_ms,
        "uncovered": uncovered_ms,
    }
    return result


def summarize_metric(request_values: list[dict[str, float]]) -> dict:
    buckets = ("compute", "communication", "uncovered")
    components = ("total", *buckets)
    result = {
        component: summarize([request[component] for request in request_values])
        for component in components
    }
    result["total"]["mean"] = sum(result[bucket]["mean"] for bucket in buckets)
    mean_total = result["total"]["mean"]
    result["mean_fraction"] = {
        bucket: result[bucket]["mean"] / mean_total for bucket in buckets
    }
    for values in request_values:
        if values["total"] != sum(values[bucket] for bucket in buckets):
            raise AssertionError(
                "Summary input violates total = compute + communication + uncovered"
            )
    return result


def session_system_clock_ns(connection: sqlite3.Connection) -> int:
    """Return the CLOCK_MONOTONIC origin corresponding to Nsight timestamp 0."""

    if not table_exists(connection, "TARGET_INFO_SESSION_START_TIME"):
        raise ValueError(
            "Nsight export lacks TARGET_INFO_SESSION_START_TIME, which is "
            "required for concurrent request-window attribution"
        )
    rows = connection.execute(
        "SELECT systemClockNs FROM TARGET_INFO_SESSION_START_TIME"
    ).fetchall()
    values = {row[0] for row in rows if row[0] is not None}
    if len(values) != 1:
        raise ValueError(
            f"Expected one Nsight system-clock origin, found {sorted(values)}"
        )
    return int(values.pop())


def clip_intervals(
    intervals: Iterable[tuple[int, int]], window_start: int, window_end: int
) -> list[tuple[int, int]]:
    return [
        (max(left, window_start), min(right, window_end))
        for left, right in intervals
        if left < window_end and right > window_start
    ]


def request_window_components(
    critical_steps: Iterable[CriticalStep],
    window_start: int,
    window_end: int,
    *,
    overlap_tolerance_ns: int = (DEFAULT_COMMUNICATION_COMPUTE_OVERLAP_TOLERANCE_NS),
    context: str = "request window",
) -> dict[str, float]:
    """Intersect one client request window with the model critical path."""

    if window_end <= window_start:
        raise ValueError(f"Invalid request window [{window_start}, {window_end}]")

    model_intervals: list[tuple[int, int]] = []
    compute_intervals: list[tuple[int, int]] = []
    communication_intervals: list[tuple[int, int]] = []
    for step in critical_steps:
        rank = step.rank
        if rank.start >= window_end or rank.end <= window_start:
            continue
        model_intervals.extend(
            clip_intervals(((rank.start, rank.end),), window_start, window_end)
        )
        compute_intervals.extend(
            clip_intervals(rank.compute_intervals, window_start, window_end)
        )
        communication_intervals.extend(
            clip_intervals(rank.communication_intervals, window_start, window_end)
        )

    total = interval_union_length(model_intervals)
    merged_compute_intervals = merge_intervals(compute_intervals)
    merged_communication_intervals = merge_intervals(communication_intervals)
    compute = interval_union_length(merged_compute_intervals)
    communication_active = interval_union_length(merged_communication_intervals)
    kernel_coverage = interval_union_length(
        (*merged_compute_intervals, *merged_communication_intervals)
    )
    communication_compute_overlap_segments = interval_intersection_segments(
        merged_communication_intervals, merged_compute_intervals
    )
    communication_compute_overlap = sum(
        right - left for left, right in communication_compute_overlap_segments
    )
    window = window_end - window_start
    if compute > total or kernel_coverage > total:
        raise ValueError(
            "Kernel interval union exceeds model span in request window: "
            f"compute={compute}, coverage={kernel_coverage}, total={total}"
        )
    communication = communication_active - communication_compute_overlap
    uncovered = total - kernel_coverage
    if communication < 0 or uncovered < 0:
        raise AssertionError(
            "Request-window kernel accounting produced a negative bucket"
        )
    if total != sum((compute, communication, uncovered)):
        raise AssertionError(
            "Request window violates total = compute + communication + uncovered "
            "in integer nanoseconds: "
            f"total={total}, compute={compute}, communication={communication}, "
            f"uncovered={uncovered}"
        )
    validate_communication_compute_overlap(
        max(
            (right - left for left, right in communication_compute_overlap_segments),
            default=0,
        ),
        overlap_tolerance_ns,
        f"contiguous overlap episode in {context} [{window_start}, {window_end}]",
    )

    ns_to_ms = 1e-6
    compute_ms = compute * ns_to_ms
    communication_ms = communication * ns_to_ms
    uncovered_ms = uncovered * ns_to_ms
    result = {
        # Use the same summation operation as the serialized invariant check.
        # Python 3.12's compensated sum can differ by one final bit from a
        # left-associated ``a + b + c`` expression.
        "total": sum((compute_ms, communication_ms, uncovered_ms)),
        "compute": compute_ms,
        "communication": communication_ms,
        "uncovered": uncovered_ms,
        "communication_compute_overlap": communication_compute_overlap * ns_to_ms,
        "outside_model": (window - total) * ns_to_ms,
        "client_window": window * ns_to_ms,
    }
    return result


def analyze_request_windows(
    connection: sqlite3.Connection,
    critical_steps_by_dp: dict[int, list[CriticalStep]],
    client: dict,
    overlap_tolerance_ns: int = (DEFAULT_COMMUNICATION_COMPUTE_OVERLAP_TOLERANCE_NS),
) -> tuple[
    list[dict[str, float]],
    list[dict[str, float]],
    list[dict[str, float]],
    list[dict],
    int,
]:
    """Build per-request metrics by aligning client and Nsight timestamps."""

    required_arrays = (
        "input_lens",
        "output_lens",
        "ttfts",
        "itls",
        "send_times",
        "finish_times",
        "dp_ranks",
    )
    request_count = len(client["output_lens"])
    for key in required_arrays:
        if key not in client:
            raise ValueError(f"Client output lacks {key}; rerun with --output-details")
        if len(client[key]) != request_count:
            raise ValueError(
                f"Client field {key} has {len(client[key])} entries; "
                f"expected {request_count}"
            )

    clock_origin = session_system_clock_ns(connection)
    ttft_values: list[dict[str, float]] = []
    tpot_values: list[dict[str, float]] = []
    e2e_values: list[dict[str, float]] = []
    request_details: list[dict] = []

    for request_index in range(request_count):
        output_len = client["output_lens"][request_index]
        dp_rank = client["dp_ranks"][request_index]
        critical_steps = critical_steps_by_dp[dp_rank]
        if output_len < 2:
            raise ValueError(
                f"Request {request_index} has output length {output_len}; "
                "TPOT requires at least two output tokens"
            )
        itls = client["itls"][request_index]
        if len(itls) != output_len - 1:
            raise ValueError(
                f"Request {request_index} has {len(itls)} inter-token "
                f"intervals for output length {output_len}"
            )

        request_start = round(client["send_times"][request_index] * 1e9) - clock_origin
        ttft_ns = round(client["ttfts"][request_index] * 1e9)
        decode_ns = round(sum(itls) * 1e9)
        first_token = request_start + ttft_ns
        request_end = first_token + decode_ns

        request_context = f"request {request_index} on attention-DP rank {dp_rank}"
        ttft = request_window_components(
            critical_steps,
            request_start,
            first_token,
            overlap_tolerance_ns=overlap_tolerance_ns,
            context=f"{request_context} TTFT window",
        )
        decode = request_window_components(
            critical_steps,
            first_token,
            request_end,
            overlap_tolerance_ns=overlap_tolerance_ns,
            context=f"{request_context} decode window",
        )
        e2e = request_window_components(
            critical_steps,
            request_start,
            request_end,
            overlap_tolerance_ns=overlap_tolerance_ns,
            context=f"{request_context} E2E window",
        )
        inter_token_count = output_len - 1
        tpot_buckets = {
            bucket: decode[bucket] / inter_token_count
            for bucket in ("compute", "communication", "uncovered")
        }
        tpot = {
            "total": sum(tpot_buckets.values()),
            **tpot_buckets,
            "outside_model": decode["outside_model"] / inter_token_count,
            "client_window": decode["client_window"] / inter_token_count,
        }

        ttft_values.append(ttft)
        tpot_values.append(tpot)
        e2e_values.append(e2e)
        request_details.append(
            {
                "request_index": request_index,
                "dp_rank": dp_rank,
                "input_len": client["input_lens"][request_index],
                "output_len": output_len,
                "request_start_trace_ms": request_start * 1e-6,
                "first_token_trace_ms": first_token * 1e-6,
                "last_token_trace_ms": request_end * 1e-6,
                "client_finish_trace_ms": (
                    round(client["finish_times"][request_index] * 1e9) - clock_origin
                )
                * 1e-6,
                "trace_model_ttft_ms": ttft,
                "trace_model_tpot_ms": tpot,
                "trace_model_e2e_ms": e2e,
                "client_ttft_ms": ttft_ns * 1e-6,
                "client_decode_ms": decode_ns * 1e-6,
                "client_e2e_ms": (ttft_ns + decode_ns) * 1e-6,
            }
        )

    return (
        ttft_values,
        tpot_values,
        e2e_values,
        request_details,
        clock_origin,
    )


def validate_client_dp_ranks(
    client: dict,
    topology: dict[int, SchedulerTopology],
) -> list[int]:
    request_count = len(client.get("output_lens", []))
    dp_ranks = client.get("dp_ranks")
    if not isinstance(dp_ranks, list) or len(dp_ranks) != request_count:
        actual = len(dp_ranks) if isinstance(dp_ranks, list) else None
        raise ValueError(
            f"Client dp_ranks has length {actual}; expected {request_count}"
        )
    available = {item.attn_dp_rank for item in topology.values()}
    validated: list[int] = []
    for request_index, dp_rank in enumerate(dp_ranks):
        if not isinstance(dp_rank, int) or isinstance(dp_rank, bool):
            raise ValueError(
                f"Client request {request_index} has missing or invalid DP rank "
                f"{dp_rank!r}"
            )
        if dp_rank not in available:
            raise ValueError(
                f"Client request {request_index} uses DP rank {dp_rank}, but trace "
                f"topology contains {sorted(available)}"
            )
        validated.append(dp_rank)
    return validated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("client_jsonl", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--communication-pattern",
        action="append",
        default=[],
        help="Additional exact CUDA kernel name classified as communication.",
    )
    parser.add_argument(
        "--skip-index-creation",
        action="store_true",
        help="Do not add lookup indexes to the derived Nsight SQLite export.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write --output without also rendering the full JSON to stdout.",
    )
    parser.add_argument(
        "--attribution-mode",
        choices=("auto", "phase", "request-window"),
        default="auto",
        help=(
            "Use phase groups (concurrency one) or clock-aligned request "
            "windows (any concurrency). Auto selects request-window above one."
        ),
    )
    parser.add_argument(
        "--overlap-tolerance-ns",
        type=int,
        default=DEFAULT_COMMUNICATION_COMPUTE_OVERLAP_TOLERANCE_NS,
        help=(
            "Maximum compute/communication kernel overlap allowed per scheduler "
            "step or contiguous overlap episode in a request window. Failures "
            "report both the observed overlap and this configured threshold."
        ),
    )
    parser.add_argument(
        "--require-prefill-cuda-graph",
        action="store_true",
        help=(
            "Fail if a measured global prefill scheduler step contains no CUDA "
            "graph nodes on any device. Use when evaluating an explicitly "
            "configured prefill graph."
        ),
    )
    args = parser.parse_args()
    if args.overlap_tolerance_ns < 0:
        parser.error("--overlap-tolerance-ns must be non-negative")

    client = load_last_json_line(args.client_jsonl)
    run_id = client.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Client output lacks a non-empty measurement run_id")
    concurrency = client.get("max_concurrency", client.get("concurrency"))
    if not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError(
            f"Expected a positive integer client concurrency, found {concurrency}"
        )
    attribution_mode = args.attribution_mode
    if attribution_mode == "auto":
        attribution_mode = "phase" if concurrency == 1 else "request-window"
    if attribution_mode == "phase" and concurrency != 1:
        raise ValueError(
            "Phase-group attribution requires concurrency 1; use "
            "--attribution-mode request-window"
        )
    if any(client.get("errors", [])):
        raise ValueError(f"Client result contains request errors: {client['errors']}")

    connection = sqlite3.connect(args.sqlite)
    for required_table in (
        "NVTX_EVENTS",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "TARGET_INFO_CUDA_CONTEXT_INFO",
    ):
        if not table_exists(connection, required_table):
            raise ValueError(f"Nsight export lacks required table {required_table}")
    if not args.skip_index_creation:
        create_analysis_indexes(connection)

    raw_ranges_by_pid = scheduler_ranges_by_pid(connection)
    topology = load_measurement_topology(connection, run_id, set(raw_ranges_by_pid))
    ranges_by_device, trace_validation = filter_scheduler_ranges_to_measurement(
        raw_ranges_by_pid, topology
    )
    client_dp_ranks = validate_client_dp_ranks(client, topology)
    extra_communication_kernel_names = frozenset(args.communication_pattern)
    communication_kernel_totals: dict[str, list[int]] = collections.defaultdict(
        lambda: [0, 0]
    )
    rank_steps: dict[int, list[RankStep]] = {}
    for rank_index, (device, ranges) in enumerate(ranges_by_device.items(), start=1):
        print(
            f"Mapping scheduler steps for device {device} "
            f"({rank_index}/{len(ranges_by_device)})...",
            file=sys.stderr,
        )
        rank_steps[device] = [
            load_rank_step(
                connection,
                device,
                scheduler_range,
                extra_communication_kernel_names,
                communication_kernel_totals,
                args.overlap_tolerance_ns,
            )
            for scheduler_range in ranges
        ]

    (
        prefill_cuda_graph_execution_by_device,
        prefill_cuda_graph_execution_by_step,
    ) = validate_prefill_cuda_graph_execution(
        rank_steps,
        required=args.require_prefill_cuda_graph,
    )

    step_groups = build_step_groups(rank_steps)
    available_dp_ranks = sorted({item.attn_dp_rank for item in topology.values()})
    critical_steps_by_dp = {
        dp_rank: critical_steps_for_dp(step_groups, dp_rank, topology)
        for dp_rank in available_dp_ranks
    }
    drain_step_indices: set[int] = set()
    session_clock_origin: int | None = None
    selected_critical_steps: list[CriticalStep] = []
    if attribution_mode == "phase":
        requests = split_requests(step_groups, client["output_lens"])
        ttft_values: list[dict[str, float]] = []
        tpot_values: list[dict[str, float]] = []
        e2e_values: list[dict[str, float]] = []
        request_details: list[dict] = []
        for request_index, (request_groups, output_len, dp_rank) in enumerate(
            zip(requests, client["output_lens"], client_dp_ranks)
        ):
            steps = [
                select_critical_step(group, dp_rank, topology)
                for group in request_groups
            ]
            selected_critical_steps.extend(steps)
            unexpected_phases = {
                step.phase for step in steps if step.phase not in ("prefill", "decode")
            }
            if unexpected_phases:
                raise ValueError(
                    f"Request {request_index} contains unsupported phases for "
                    f"concurrency-one attribution: {sorted(unexpected_phases)}"
                )
            prefill_steps = [step for step in steps if step.phase == "prefill"]
            traced_decode_steps = [step for step in steps if step.phase == "decode"]
            if len(traced_decode_steps) == output_len:
                drain_steps = traced_decode_steps[-1:]
                decode_steps = traced_decode_steps[:-1]
                drain_step_indices.add(drain_steps[0].index)
            else:
                drain_steps = []
                decode_steps = traced_decode_steps
            ttft = component_totals(prefill_steps)
            decode = component_totals(decode_steps)
            e2e_buckets = {
                bucket: ttft[bucket] + decode[bucket]
                for bucket in ("compute", "communication", "uncovered")
            }
            e2e = {
                "total": sum(e2e_buckets.values()),
                **e2e_buckets,
            }
            decode_count = len(decode_steps)
            tpot_buckets = {
                bucket: decode[bucket] / decode_count
                for bucket in ("compute", "communication", "uncovered")
            }
            tpot = {
                "total": sum(tpot_buckets.values()),
                **tpot_buckets,
            }
            ttft_values.append(ttft)
            tpot_values.append(tpot)
            e2e_values.append(e2e)

            client_ttft_ms = client["ttfts"][request_index] * 1000
            client_decode_ms = sum(client["itls"][request_index]) * 1000
            request_details.append(
                {
                    "request_index": request_index,
                    "dp_rank": dp_rank,
                    "input_len": client["input_lens"][request_index],
                    "output_len": output_len,
                    "prefill_steps": len(prefill_steps),
                    "decode_steps": decode_count,
                    "drain_steps": len(drain_steps),
                    "selected_devices": sorted(
                        {step.rank.device for step in prefill_steps + decode_steps}
                    ),
                    "trace_model_ttft_ms": ttft,
                    "trace_model_tpot_ms": tpot,
                    "trace_model_e2e_ms": e2e,
                    "client_ttft_ms": client_ttft_ms,
                    "client_decode_ms": client_decode_ms,
                    "client_e2e_ms": client_ttft_ms + client_decode_ms,
                }
            )
        request_count = len(requests)
        timed_decode_step_count: int | None = sum(
            detail["decode_steps"] for detail in request_details
        )
        drain_step_count: int | None = len(drain_step_indices)
    else:
        (
            ttft_values,
            tpot_values,
            e2e_values,
            request_details,
            session_clock_origin,
        ) = analyze_request_windows(
            connection,
            critical_steps_by_dp,
            client,
            args.overlap_tolerance_ns,
        )
        selected_critical_steps = [
            step
            for dp_rank in available_dp_ranks
            for step in critical_steps_by_dp[dp_rank]
        ]
        request_count = len(request_details)
        timed_decode_step_count = None
        drain_step_count = None

    ns_to_ms = 1e-6
    request_window_validation = None
    if attribution_mode == "request-window":
        request_window_validation = {}
        validation_fields = {
            "ttft": ("trace_model_ttft_ms", "client_ttft_ms", 1),
            "tpot": ("trace_model_tpot_ms", "client_decode_ms", None),
            "e2e": ("trace_model_e2e_ms", "client_e2e_ms", 1),
        }
        for metric, (trace_key, client_key, divisor) in validation_fields.items():
            client_values = []
            model_values = []
            residual_values = []
            coverage_values = []
            for detail in request_details:
                effective_divisor = (
                    detail["output_len"] - 1 if divisor is None else divisor
                )
                client_value = detail[client_key] / effective_divisor
                trace_value = detail[trace_key]["total"]
                residual = detail[trace_key]["outside_model"]
                client_values.append(client_value)
                model_values.append(trace_value)
                residual_values.append(residual)
                coverage_values.append(trace_value / client_value)
            request_window_validation[metric] = {
                "client_ms": summarize(client_values),
                "trace_model_ms": summarize(model_values),
                "outside_model_ms": summarize(residual_values),
                "model_coverage_fraction": summarize(coverage_values),
            }

    critical_device_counts_by_dp = {
        str(dp_rank): dict(
            sorted(
                collections.Counter(
                    step.rank.device for step in critical_steps_by_dp[dp_rank]
                ).items()
            )
        )
        for dp_rank in available_dp_ranks
    }
    step_selection_audit = [
        {
            "index": group.index,
            "phase": group.phase,
            "all_rank_longest_device": group.all_rank_longest.device,
            "all_rank_longest_total_ms": group.all_rank_longest.total * ns_to_ms,
            "selected_by_attn_dp_rank": {
                str(dp_rank): {
                    "device": critical_steps_by_dp[dp_rank][group.index].rank.device,
                    "total_ms": (
                        critical_steps_by_dp[dp_rank][group.index].rank.total * ns_to_ms
                    ),
                }
                for dp_rank in available_dp_ranks
            },
        }
        for group in step_groups
    ]
    result = {
        "label": args.label,
        "run_id": run_id,
        "concurrency": concurrency,
        "attribution_mode": attribution_mode,
        "session_system_clock_ns": session_clock_origin,
        "definition": {
            "scope": "GPU model execution inside scheduler.run_batch",
            "compute": "union of non-communication GPU kernel intervals",
            "communication": (
                "union of communication GPU kernel intervals after assigning "
                "any tolerated compute overlap to compute"
            ),
            "uncovered": (
                "selected-rank first-to-last-kernel span with no classified "
                "GPU kernel active; includes host launch and synchronization gaps"
            ),
            "additive_invariant": "total = compute + communication + uncovered",
            "phase_source": "explicit scheduler batch-phase NVTX markers",
            "request_window_semantics": (
                "GPU critical-path time experienced while each request is "
                "outstanding; work from co-batched or preceding requests is "
                "included when it occupies the model path"
                if attribution_mode == "request-window"
                else None
            ),
            "communication_kernel_prefixes": ["ncclDevKernel_"],
            "communication_kernel_names": sorted(
                COMMUNICATION_KERNEL_NAMES | extra_communication_kernel_names
            ),
            "communication_compute_overlap_tolerance_ns": (args.overlap_tolerance_ns),
        },
        "rank_count": len(ranges_by_device),
        "request_count": request_count,
        "scheduler_step_count": len(step_groups),
        "prefill_step_count": sum(
            any(rank.phase == "prefill" for rank in step.ranks.values())
            for step in step_groups
        ),
        "timed_decode_step_count": timed_decode_step_count,
        "drain_step_count": drain_step_count,
        "decode_step_count_including_drain": sum(
            any(rank.phase == "decode" for rank in step.ranks.values())
            for step in step_groups
        ),
        "mixed_phase_step_count": sum(step.phase == "mixed" for step in step_groups),
        "critical_device_counts_by_attn_dp_rank": critical_device_counts_by_dp,
        "metrics": {
            "ttft_model_ms": summarize_metric(ttft_values),
            "tpot_model_ms": summarize_metric(tpot_values),
            "e2e_model_ms": summarize_metric(e2e_values),
        },
        "client_validation": {
            "mean_ttft_ms": client["mean_ttft_ms"],
            "mean_tpot_ms": client["mean_tpot_ms"],
            "mean_e2e_latency_ms": client["mean_e2e_latency_ms"],
            "input_lens": client["input_lens"],
            "output_lens": client["output_lens"],
            "dp_ranks": client_dp_ranks,
            "errors": client["errors"],
        },
        "request_window_validation": request_window_validation,
        "trace_validation": {
            "scheduler_range_counts_by_device": trace_validation,
            "prefill_cuda_graph_execution_by_device": (
                prefill_cuda_graph_execution_by_device
            ),
            "prefill_cuda_graph_execution_by_step": (
                prefill_cuda_graph_execution_by_step
            ),
            "topology": [
                {
                    "pid": item.pid,
                    "device": item.device,
                    "begin_ns": item.begin,
                    "end_ns": item.end,
                    "gpu_id": item.gpu_id,
                    "tp_rank": item.tp_rank,
                    "pp_rank": item.pp_rank,
                    "dp_rank": item.dp_rank,
                    "attn_dp_rank": item.attn_dp_rank,
                    "attn_tp_rank": item.attn_tp_rank,
                    "moe_ep_rank": item.moe_ep_rank,
                }
                for item in topology.values()
            ],
        },
        "request_details": request_details,
        "diagnostics": {
            "streams_per_selected_step": summarize(
                [float(step.rank.stream_count) for step in selected_critical_steps]
            ),
            "communication_kernels": [
                {"name": name, "count": count, "total_ms": duration * ns_to_ms}
                for name, (count, duration) in sorted(
                    communication_kernel_totals.items(),
                    key=lambda item: item[1][1],
                    reverse=True,
                )
            ],
            "selected_step_kernel_accounting_ms": {
                "communication_kernel_active": sum(
                    step.rank.communication_kernel_active
                    for step in selected_critical_steps
                )
                * ns_to_ms,
                "compute_communication_overlap": sum(
                    step.rank.communication_compute_overlap
                    for step in selected_critical_steps
                )
                * ns_to_ms,
                "uncovered": sum(
                    step.rank.uncovered for step in selected_critical_steps
                )
                * ns_to_ms,
            },
            "step_selection_audit": step_selection_audit,
        },
        "critical_steps": [
            {
                "index": step.index,
                "phase": "drain" if step.index in drain_step_indices else step.phase,
                "attn_dp_rank": topology[step.rank.device].attn_dp_rank,
                "device": step.rank.device,
                "all_rank_longest_device": step.all_rank_longest.device,
                "start_ms": step.rank.start * ns_to_ms,
                "end_ms": step.rank.end * ns_to_ms,
                "total_ms": sum(
                    (
                        step.rank.compute * ns_to_ms,
                        step.rank.communication * ns_to_ms,
                        step.rank.uncovered * ns_to_ms,
                    )
                ),
                "compute_ms": step.rank.compute * ns_to_ms,
                "communication_ms": step.rank.communication * ns_to_ms,
                "uncovered_ms": step.rank.uncovered * ns_to_ms,
                "communication_kernel_active_ms": (
                    step.rank.communication_kernel_active * ns_to_ms
                ),
                "compute_communication_overlap_ms": (
                    step.rank.communication_compute_overlap * ns_to_ms
                ),
                "kernel_coverage_ms": step.rank.kernel_coverage * ns_to_ms,
                "kernel_count": step.rank.kernel_count,
                "graph_kernel_count": step.rank.graph_kernel_count,
                "stream_count": step.rank.stream_count,
            }
            for step in selected_critical_steps
        ],
    }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if not args.quiet:
        print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
