#!/usr/bin/env python3

"""Combine clean client metrics and full-trace breakdowns for two variants.

The trace-model comparison excludes the analyzer's ``launch_stall`` bucket, so
``trace_model`` compares compute + communication + uncovered. The raw
four-bucket totals remain available as ``trace_model_raw`` together with the
per-variant launch stall that was separated out, and every trace number is
validated against the marker-only client run.
"""

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
VARIANTS = ("baseline", "purlin")
MODEL_BUCKETS = ("compute", "communication", "uncovered")
TRACE_COMPONENTS = ("total", *MODEL_BUCKETS)
# Deviation of the profiled client from the marker-only client above which the
# raw trace is reported as perturbed by the profiler.
CAPTURE_OVERHEAD_WARNING = 0.05


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


def has_launch_stall(metric_summary: dict) -> bool:
    return "launch_stall" in metric_summary


def corrected_components(metric_summary: dict) -> dict[str, float]:
    """Mean model components with launch stall excluded."""

    if has_launch_stall(metric_summary):
        total = metric_summary["total_excluding_launch_stall"]["mean"]
    else:
        total = metric_summary["total"]["mean"]
    return {
        "total": total,
        **{bucket: metric_summary[bucket]["mean"] for bucket in MODEL_BUCKETS},
    }


def raw_components(metric_summary: dict) -> dict[str, float]:
    """Mean components with launch stall folded back into its source buckets."""

    values = {bucket: metric_summary[bucket]["mean"] for bucket in MODEL_BUCKETS}
    if has_launch_stall(metric_summary):
        values["communication"] += metric_summary[
            "launch_stall_from_communication"
        ]["mean"]
        values["uncovered"] += metric_summary["launch_stall_from_uncovered"]["mean"]
    return {"total": metric_summary["total"]["mean"], **values}


def launch_stall_mean(metric_summary: dict) -> float:
    if has_launch_stall(metric_summary):
        return metric_summary["launch_stall"]["mean"]
    return 0.0


def compare_components(
    baseline: dict[str, float], purlin: dict[str, float]
) -> dict:
    comparison = {}
    for component in TRACE_COMPONENTS:
        baseline_value = baseline[component]
        purlin_value = purlin[component]
        comparison[component] = {
            "baseline_ms": baseline_value,
            "purlin_ms": purlin_value,
            "saved_ms": baseline_value - purlin_value,
            "reduction": reduction(baseline_value, purlin_value),
        }
    saved_total = comparison["total"]["saved_ms"]
    comparison["raw_delta_fraction_of_total_delta"] = {
        bucket: (comparison[bucket]["saved_ms"] / saved_total if saved_total else None)
        for bucket in MODEL_BUCKETS
    }
    return comparison


def build_observations(metrics: dict) -> dict:
    observations = {}
    for metric, values in metrics.items():
        label = metric.upper()
        observations[metric] = {
            "clean_client": describe_change(
                f"Clean-client {label}",
                values["clean_client"]["baseline_ms"],
                values["clean_client"]["purlin_ms"],
            ),
            "trace_model": {
                component: describe_change(
                    f"Trace-model {label} {component} (launch stall excluded)",
                    values["trace_model"][component]["baseline_ms"],
                    values["trace_model"][component]["purlin_ms"],
                )
                for component in TRACE_COMPONENTS
            },
            "trace_model_raw": describe_change(
                f"Raw trace {label} total (launch stall included)",
                values["trace_model_raw"]["total"]["baseline_ms"],
                values["trace_model_raw"]["total"]["purlin_ms"],
            ),
            "launch_stall": (
                f"Launch stall separated from {label}: baseline "
                f"{values['launch_stall']['baseline_ms']:.3f} ms "
                f"({values['launch_stall']['baseline_fraction_of_raw_total'] * 100:.2f}% "
                f"of the raw trace total), Purlin "
                f"{values['launch_stall']['purlin_ms']:.3f} ms "
                f"({values['launch_stall']['purlin_fraction_of_raw_total'] * 100:.2f}%)."
            ),
        }
    return observations


def describe_capture(variant: str, profiled_vs_markers: float) -> str:
    percent = profiled_vs_markers * 100
    if abs(profiled_vs_markers) > CAPTURE_OVERHEAD_WARNING:
        return (
            f"The {variant} profiled client deviates from the marker-only "
            f"client by {percent:+.2f}%; raw trace totals inherit that "
            "perturbation. Use trace_model, which excludes launch_stall, and "
            "check validation.trace_model_vs_markers."
        )
    return (
        f"The {variant} profiled client is within {percent:+.2f}% of the "
        "marker-only client."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--baseline-clean", type=Path)
    parser.add_argument("--purlin-clean", type=Path)
    parser.add_argument("--baseline-breakdown", type=Path)
    parser.add_argument("--purlin-breakdown", type=Path)
    parser.add_argument(
        "--breakdown-suffix",
        default="full_breakdown.json",
        help="Breakdown file name after '<variant>_' when no explicit path is given.",
    )
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
        for variant in VARIANTS
    }
    profiled = {
        variant: load_json(root / f"{variant}_full_profile.jsonl")
        for variant in VARIANTS
    }
    breakdown_paths = {
        "baseline": args.baseline_breakdown
        or root / f"baseline_{args.breakdown_suffix}",
        "purlin": args.purlin_breakdown or root / f"purlin_{args.breakdown_suffix}",
    }
    breakdown = {variant: load_json(path) for variant, path in breakdown_paths.items()}

    metrics = {}
    for metric, (client_key, trace_key) in METRICS.items():
        summaries = {
            variant: breakdown[variant]["metrics"][trace_key] for variant in VARIANTS
        }
        corrected = {
            variant: corrected_components(summary)
            for variant, summary in summaries.items()
        }
        raw = {variant: raw_components(summary) for variant, summary in summaries.items()}
        stall = {
            variant: launch_stall_mean(summary) for variant, summary in summaries.items()
        }
        clean_baseline = clean["baseline"][client_key]
        clean_purlin = clean["purlin"][client_key]
        metrics[metric] = {
            "clean_client": {
                "baseline_ms": clean_baseline,
                "purlin_ms": clean_purlin,
                "reduction": reduction(clean_baseline, clean_purlin),
            },
            "scheduler_markers_client": {
                variant: markers[variant][client_key] for variant in VARIANTS
            },
            "profiled_client": {
                variant: profiled[variant][client_key] for variant in VARIANTS
            },
            "trace_model": {
                **compare_components(corrected["baseline"], corrected["purlin"]),
                "excludes_launch_stall": True,
            },
            "trace_model_raw": {
                **compare_components(raw["baseline"], raw["purlin"]),
                "includes_launch_stall": True,
            },
            "launch_stall": {
                "baseline_ms": stall["baseline"],
                "purlin_ms": stall["purlin"],
                "baseline_fraction_of_raw_total": (
                    stall["baseline"] / raw["baseline"]["total"]
                ),
                "purlin_fraction_of_raw_total": stall["purlin"] / raw["purlin"]["total"],
            },
            "validation": {
                variant: {
                    "markers_vs_clean": overhead(
                        clean[variant][client_key], markers[variant][client_key]
                    ),
                    "profiled_client_vs_markers": overhead(
                        markers[variant][client_key], profiled[variant][client_key]
                    ),
                    "trace_model_raw_vs_profiled_client": overhead(
                        profiled[variant][client_key], raw[variant]["total"]
                    ),
                    "trace_model_raw_vs_markers": overhead(
                        markers[variant][client_key], raw[variant]["total"]
                    ),
                    "trace_model_vs_markers": overhead(
                        markers[variant][client_key], corrected[variant]["total"]
                    ),
                }
                for variant in VARIANTS
            },
        }

    capture_quality = {
        variant: {
            "profiled_client_vs_markers_tpot": metrics["tpot"]["validation"][variant][
                "profiled_client_vs_markers"
            ],
            "trace_model_vs_markers_tpot": metrics["tpot"]["validation"][variant][
                "trace_model_vs_markers"
            ],
            "assessment": describe_capture(
                variant,
                metrics["tpot"]["validation"][variant]["profiled_client_vs_markers"],
            ),
        }
        for variant in VARIANTS
    }

    result = {
        "observations": build_observations(metrics),
        "capture_quality": capture_quality,
        "metrics": metrics,
        "launch_stall_definition": {
            variant: breakdown[variant]["definition"].get("launch_stall")
            for variant in VARIANTS
        },
        "launch_stall_by_phase": {
            variant: breakdown[variant]["diagnostics"].get("launch_stall_by_phase")
            for variant in VARIANTS
        },
        "overlap_tolerance_ns": {
            variant: breakdown[variant]["definition"][
                "communication_compute_overlap_tolerance_ns"
            ]
            for variant in VARIANTS
        },
        "breakdown_files": {
            variant: str(path) for variant, path in breakdown_paths.items()
        },
        "trace_attribution_mode": {
            variant: breakdown[variant].get("attribution_mode", "phase")
            for variant in VARIANTS
        },
        "trace_request_count": {
            variant: breakdown[variant]["request_count"] for variant in VARIANTS
        },
        "trace_step_counts": {
            variant: {
                "prefill": breakdown[variant]["prefill_step_count"],
                "decode": breakdown[variant]["timed_decode_step_count"],
                "drain_excluded": breakdown[variant]["drain_step_count"],
            }
            for variant in VARIANTS
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if not args.quiet:
        print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
