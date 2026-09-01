#!/usr/bin/env python3

"""Combine clean client metrics and full-trace breakdowns for two variants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

METRICS = {
    "ttft": ("mean_ttft_ms", "ttft_model_ms"),
    "tpot": ("mean_tpot_ms", "tpot_model_ms"),
    "e2e": ("mean_e2e_latency_ms", "e2e_model_ms"),
}
TRACE_BUCKETS = ("compute", "communication", "uncovered")
TRACE_COMPONENTS = ("total", *TRACE_BUCKETS)


def load_json(path: Path) -> dict:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".jsonl":
        return json.loads(lines[-1])
    return json.loads("\n".join(lines))


def reduction(baseline: float, purlin: float) -> float:
    return (baseline - purlin) / baseline


def overhead(reference: float, measured: float) -> float:
    return measured / reference - 1


def describe_change(label: str, baseline: float, purlin: float) -> str:
    delta = purlin - baseline
    if math.isclose(delta, 0.0, rel_tol=1e-6, abs_tol=1e-9):
        return f"{label} was unchanged at {baseline:.3f} ms."
    direction = "increased" if delta > 0 else "decreased"
    percent = abs(delta) / abs(baseline) * 100 if baseline else None
    percent_text = f", {percent:.2f}%" if percent is not None else ""
    return (
        f"{label} {direction} from {baseline:.3f} ms to {purlin:.3f} ms "
        f"({delta:+.3f} ms{percent_text})."
    )


def build_observations(metrics: dict) -> dict:
    observations = {}
    for metric, values in metrics.items():
        observations[metric] = {
            "clean_client": describe_change(
                f"Clean-client {metric.upper()}",
                values["clean_client"]["baseline_ms"],
                values["clean_client"]["purlin_ms"],
            ),
            "trace_model": {
                component: describe_change(
                    f"Trace-model {metric.upper()} {component}",
                    values["trace_model"][component]["baseline_ms"],
                    values["trace_model"][component]["purlin_ms"],
                )
                for component in TRACE_COMPONENTS
            },
        }
    return observations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--baseline-clean", type=Path)
    parser.add_argument("--purlin-clean", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = args.result_dir
    clean = {
        "baseline": load_json(args.baseline_clean or root / "baseline_clean.jsonl"),
        "purlin": load_json(args.purlin_clean or root / "purlin_clean.jsonl"),
    }
    markers = {
        variant: load_json(root / f"{variant}_scheduler_markers.jsonl")
        for variant in ("baseline", "purlin")
    }
    profiled = {
        variant: load_json(root / f"{variant}_full_profile.jsonl")
        for variant in ("baseline", "purlin")
    }
    breakdown = {
        variant: load_json(root / f"{variant}_full_breakdown.json")
        for variant in ("baseline", "purlin")
    }

    metrics = {}
    for metric, (client_key, trace_key) in METRICS.items():
        clean_baseline = clean["baseline"][client_key]
        clean_purlin = clean["purlin"][client_key]
        trace_baseline = breakdown["baseline"]["metrics"][trace_key]
        trace_purlin = breakdown["purlin"]["metrics"][trace_key]
        component_comparison = {}
        for component in TRACE_COMPONENTS:
            baseline_value = trace_baseline[component]["mean"]
            purlin_value = trace_purlin[component]["mean"]
            component_comparison[component] = {
                "baseline_ms": baseline_value,
                "purlin_ms": purlin_value,
                "saved_ms": baseline_value - purlin_value,
                "reduction": reduction(baseline_value, purlin_value),
            }
        saved_total = component_comparison["total"]["saved_ms"]
        component_comparison["raw_delta_fraction_of_total_delta"] = {
            bucket: (
                component_comparison[bucket]["saved_ms"] / saved_total
                if saved_total
                else None
            )
            for bucket in TRACE_BUCKETS
        }
        metrics[metric] = {
            "clean_client": {
                "baseline_ms": clean_baseline,
                "purlin_ms": clean_purlin,
                "reduction": reduction(clean_baseline, clean_purlin),
            },
            "scheduler_markers_client": {
                variant: markers[variant][client_key]
                for variant in ("baseline", "purlin")
            },
            "profiled_client": {
                variant: profiled[variant][client_key]
                for variant in ("baseline", "purlin")
            },
            "trace_model": component_comparison,
            "validation": {
                variant: {
                    "markers_vs_clean": overhead(
                        clean[variant][client_key], markers[variant][client_key]
                    ),
                    "profiled_client_vs_markers": overhead(
                        markers[variant][client_key], profiled[variant][client_key]
                    ),
                    "trace_model_vs_profiled_client": overhead(
                        profiled[variant][client_key],
                        breakdown[variant]["metrics"][trace_key]["total"]["mean"],
                    ),
                }
                for variant in ("baseline", "purlin")
            },
        }

    result = {
        "observations": build_observations(metrics),
        "metrics": metrics,
        "trace_attribution_mode": {
            variant: breakdown[variant].get("attribution_mode", "phase")
            for variant in ("baseline", "purlin")
        },
        "trace_request_count": {
            variant: breakdown[variant]["request_count"]
            for variant in ("baseline", "purlin")
        },
        "trace_step_counts": {
            variant: {
                "prefill": breakdown[variant]["prefill_step_count"],
                "decode": breakdown[variant]["timed_decode_step_count"],
                "drain_excluded": breakdown[variant]["drain_step_count"],
            }
            for variant in ("baseline", "purlin")
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if not args.quiet:
        print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
