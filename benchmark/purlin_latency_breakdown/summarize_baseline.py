#!/usr/bin/env python3
"""Summarize mean model-side and client latency for one stock SGLang trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_results import METRICS, MODEL_BUCKETS, load_json


def summarize_baseline(breakdown_path: Path) -> dict:
    trace = json.loads(breakdown_path.read_bytes())
    if trace["label"] != "baseline":
        raise ValueError(f"Expected a baseline breakdown, got {trace['label']!r}")

    clients = {"profiled": trace["client_validation"]}
    for phase in ("clean", "scheduler_markers"):
        path = breakdown_path.with_name(f"baseline_{phase}.jsonl")
        if path.is_file():
            clients[phase] = load_json(path)

    metrics = {}
    for metric, (client_key, trace_key) in METRICS.items():
        values = trace["metrics"][trace_key]
        buckets = (
            (*MODEL_BUCKETS, "launch_stall")
            if "launch_stall" in values
            else MODEL_BUCKETS
        )
        metrics[metric] = {
            "model_total_ms": values["total"]["mean"],
            "components": {
                bucket: {
                    "mean_ms": values[bucket]["mean"],
                    "percent_of_model_total": 100 * values["mean_fraction"][bucket],
                }
                for bucket in buckets
            },
            "client_mean_ms": {
                phase: row[client_key] for phase, row in clients.items()
            },
        }

    return {
        "label": trace["label"],
        "run_id": trace["run_id"],
        "concurrency": trace["concurrency"],
        "request_count": trace["request_count"],
        "rank_count": trace["rank_count"],
        "overlap_tolerance_ns": trace["definition"][
            "communication_compute_overlap_tolerance_ns"
        ],
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result", type=Path, help="Result directory or baseline breakdown JSON."
    )
    parser.add_argument("--output", type=Path, help="Write the summary as JSON.")
    args = parser.parse_args()
    path = (
        args.result / "baseline_full_breakdown.json"
        if args.result.is_dir()
        else args.result
    )
    if args.output and args.output.resolve() == path.resolve():
        parser.error("Choose a summary output path different from the input breakdown.")
    summary = summarize_baseline(path)
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"{'Metric':<7} {'Model total (ms)':>16} {'Communication (ms)':>18} {'% of model total':>16}"
    )
    for name, metric in summary["metrics"].items():
        comm = metric["components"]["communication"]
        print(
            f"{name.upper():<7} {metric['model_total_ms']:16.3f} "
            f"{comm['mean_ms']:18.3f} {comm['percent_of_model_total']:15.2f}%"
        )


if __name__ == "__main__":
    main()
