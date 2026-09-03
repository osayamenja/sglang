#!/usr/bin/env python3

"""Combine clean one-batch timings and focused Nsight breakdowns."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ADDITIVE_COMPONENTS = ("total", "compute", "communication", "uncovered")
DIAGNOSTIC_COMPONENTS = (
    "communication_active",
    "compute_communication_overlap",
)
TRACE_KEYS = {
    "total": "span_ms",
    "compute": "compute_active_ms",
    "communication": "communication_exclusive_ms",
    "uncovered": "uncovered_ms",
    "communication_active": "communication_active_ms",
    "compute_communication_overlap": "compute_communication_overlap_ms",
}
CLEAN_KEYS = {
    "prefill_latency": "prefill_latency",
    "decode_latency": "median_decode_latency",
    "e2e_time": "total_latency",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def reduction(baseline: float, purlin: float) -> float | None:
    return (baseline - purlin) / baseline if baseline else None


def compare_values(baseline: float, purlin: float) -> dict:
    return {
        "baseline_ms": baseline,
        "purlin_ms": purlin,
        "saved_ms": baseline - purlin,
        "reduction": reduction(baseline, purlin),
    }


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


def stage_components(breakdown: dict) -> dict[str, float]:
    critical = breakdown["critical_path"]
    return {name: float(critical[key]) for name, key in TRACE_KEYS.items()}


def add_scaled_components(
    prefill: dict[str, float], decode: dict[str, float], decode_steps: int
) -> dict[str, float]:
    return {
        name: prefill[name] + decode_steps * decode[name]
        for name in (*ADDITIVE_COMPONENTS, *DIAGNOSTIC_COMPONENTS)
    }


def trace_comparison(baseline: dict[str, float], purlin: dict[str, float]) -> dict:
    result = {
        component: compare_values(baseline[component], purlin[component])
        for component in ADDITIVE_COMPONENTS
    }
    saved_total = result["total"]["saved_ms"]
    result["raw_delta_fraction_of_total_delta"] = {
        component: (
            result[component]["saved_ms"] / saved_total if saved_total else None
        )
        for component in ("compute", "communication", "uncovered")
    }
    result["diagnostics"] = {
        component: compare_values(baseline[component], purlin[component])
        for component in DIAGNOSTIC_COMPONENTS
    }
    return result


def build_observations(metrics: dict) -> dict:
    observations = {}
    for metric, values in metrics.items():
        observations[metric] = {
            "clean_static": describe_change(
                f"Clean static {metric.replace('_', ' ')}",
                values["clean_static"]["baseline_ms"],
                values["clean_static"]["purlin_ms"],
            ),
            "trace_model": {
                component: describe_change(
                    f"Trace-model {metric.replace('_', ' ')} {component}",
                    values["trace_model"][component]["baseline_ms"],
                    values["trace_model"][component]["purlin_ms"],
                )
                for component in ADDITIVE_COMPONENTS
            },
        }
    return observations


def build_comparison(
    clean_summary: dict,
    manifest: dict,
    breakdowns: dict[str, dict[str, dict]],
) -> dict:
    decode_steps = int(manifest["workload"]["output_len"]) - 1
    if decode_steps < 1:
        raise ValueError("An E2E one-batch breakdown requires output_len >= 2")

    components = {
        variant: {
            stage: stage_components(breakdowns[variant][stage])
            for stage in ("prefill", "decode")
        }
        for variant in ("baseline", "purlin")
    }
    for variant in ("baseline", "purlin"):
        components[variant]["e2e"] = add_scaled_components(
            components[variant]["prefill"],
            components[variant]["decode"],
            decode_steps,
        )

    trace_stage = {
        "prefill_latency": "prefill",
        "decode_latency": "decode",
        "e2e_time": "e2e",
    }
    metrics = {}
    for metric, clean_key in CLEAN_KEYS.items():
        clean_baseline = clean_summary["variants"]["baseline"]["metrics"][clean_key][
            "median"
        ]
        clean_purlin = clean_summary["variants"]["purlin"]["metrics"][clean_key][
            "median"
        ]
        stage = trace_stage[metric]
        metrics[metric] = {
            "clean_static": compare_values(clean_baseline, clean_purlin),
            "trace_model": trace_comparison(
                components["baseline"][stage], components["purlin"][stage]
            ),
        }

    return {
        "schema_version": 1,
        "scope": (
            "mechanistic static ModelRunner results from benchmark.one_batch; "
            "not realistic serving TTFT, TPOT, or request E2E latency"
        ),
        "metrics": metrics,
        "observations": build_observations(metrics),
        "trace_derivation": {
            "prefill_latency": "one captured prefill forward",
            "decode_latency": "one representative captured decode forward",
            "e2e_time": (
                "captured prefill + (output_len - 1) * representative captured "
                "decode forward"
            ),
            "decode_steps_in_e2e": decode_steps,
            "additive_identity": "total = compute + communication + uncovered",
            "communication": "exclusive communication-kernel-active time",
        },
        "trace_critical_devices": {
            variant: {
                stage: breakdowns[variant][stage]["critical_device"]
                for stage in ("prefill", "decode")
            }
            for variant in ("baseline", "purlin")
        },
        "clean_sample_counts": {
            variant: clean_summary["variants"][variant]["samples"]
            for variant in ("baseline", "purlin")
        },
    }


def comparison_inputs(root: Path) -> list[Path]:
    return [
        root / "summary.json",
        root / "manifest.json",
        *(
            root / f"{variant}_{stage}_breakdown.json"
            for variant in ("baseline", "purlin")
            for stage in ("prefill", "decode")
        ),
    ]


def build_comparison_from_directory(root: Path) -> dict:
    missing = [path for path in comparison_inputs(root) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot build static comparison; missing:\n"
            + "\n".join(f"  {path}" for path in missing)
        )
    breakdowns = {
        variant: {
            stage: load_json(root / f"{variant}_{stage}_breakdown.json")
            for stage in ("prefill", "decode")
        }
        for variant in ("baseline", "purlin")
    }
    return build_comparison(
        load_json(root / "summary.json"),
        load_json(root / "manifest.json"),
        breakdowns,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = build_comparison_from_directory(args.result_dir)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if not args.quiet:
        print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
