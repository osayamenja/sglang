#!/usr/bin/env python3

"""Run matched baseline/Purlin static-batch benchmarks and focused traces.

This is a companion to ``run_suite.py``.  It bypasses HTTP and the scheduler,
loads ``ModelRunner`` directly through ``sglang.benchmark.one_batch``, and is
therefore useful for repeatable model-forward measurements.  Its timings are
not serving TTFT or end-to-end request latency.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import statistics
import subprocess
from pathlib import Path

from summarize_one_batch_results import (
    build_comparison_from_directory,
    comparison_inputs,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_NSYS = (
    Path("/usr/local/bin/nsys")
    if Path("/usr/local/bin/nsys").exists()
    else Path("nsys")
)


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def run_streamed(command: list[str], log_path: Path) -> None:
    print(f"Running:\n  {format_command(command)}", flush=True)
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_logged(command: list[str], log_path: Path) -> None:
    print(f"Running:\n  {format_command(command)}", flush=True)
    with log_path.open("w") as log_file:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def ensure_absent(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        rendered = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing experiment artifacts:\n" + rendered
        )
    if overwrite:
        for path in existing:
            if not path.is_file():
                raise IsADirectoryError(f"Refusing to replace non-file artifact {path}")
            path.unlink()


def one_batch_command(
    args: argparse.Namespace,
    variant: str,
    result_path: Path,
    *,
    repeats: int,
    profile_stage: str | None = None,
) -> list[str]:
    command = [
        str(args.python),
        "-m",
        "sglang.benchmark.one_batch",
        "--model-path",
        args.model,
        "--tp-size",
        str(args.tp),
        "--dp-size",
        str(args.dp),
        "--ep-size",
        str(args.ep),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--batch-size",
        *([str(args.batch_size)] * repeats),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--run-name",
        f"{variant}-{'profile-' + profile_stage if profile_stage else 'clean'}",
        "--result-filename",
        str(result_path),
    ]
    if not args.use_server_default_attention_backend:
        command.extend(("--attention-backend", args.attention_backend))
    if args.revision:
        command.extend(("--revision", args.revision))
    if args.enable_dp_attention:
        command.append("--enable-dp-attention")
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.cuda_graph_backend_prefill:
        command.extend(
            ("--cuda-graph-backend-prefill", args.cuda_graph_backend_prefill)
        )
    if args.cuda_graph_backend_decode:
        command.extend(("--cuda-graph-backend-decode", args.cuda_graph_backend_decode))
    if args.cuda_graph_bs_prefill:
        command.append("--cuda-graph-bs-prefill")
        command.extend(str(value) for value in args.cuda_graph_bs_prefill)
    if variant == "purlin":
        command.append("--enable-purlin")
    if profile_stage:
        command.extend(
            (
                "--profile",
                "--profile-activities",
                "CUDA_PROFILER",
                "--profile-stage",
                profile_stage,
            )
        )
        if profile_stage == "decode":
            command.extend(
                (
                    "--profile-start-step",
                    str(args.profile_start_step),
                    "--profile-steps",
                    str(args.profile_steps),
                )
            )
    command.extend(args.server_arg)
    return command


def resolve_cuda_graph_trace_mode(args: argparse.Namespace) -> str | None:
    if args.nsys_cuda_graph_trace == "off":
        return None
    if args.nsys_cuda_graph_trace != "auto":
        return args.nsys_cuda_graph_trace
    result = subprocess.run(
        [str(args.nsys_command), "profile", "--help"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    if "--cuda-graph-trace=" not in result.stdout:
        raise RuntimeError(
            f"{args.nsys_command} lacks --cuda-graph-trace; node-level kernel "
            "records are required for this breakdown"
        )
    return "node"


def nsys_command(
    args: argparse.Namespace,
    command: list[str],
    report_base: Path,
    cuda_graph_trace_mode: str | None,
) -> list[str]:
    result = [
        str(args.nsys_command),
        "profile",
        "--force-overwrite=true",
        "--sample=none",
        "--cpuctxsw=none",
        "--trace=cuda",
        "--trace-fork-before-exec=true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        f"--output={report_base}",
    ]
    if cuda_graph_trace_mode:
        result.insert(7, f"--cuda-graph-trace={cuda_graph_trace_mode}")
    return [*result, *command]


def read_jsonlines(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON on {path}:{line_number}: {error}"
            ) from error
    return records


def summarize_samples(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Clean one-batch run produced no result records")
    metrics = (
        "prefill_latency",
        "median_decode_latency",
        "total_latency",
        "prefill_throughput",
        "median_decode_throughput",
        "overall_throughput",
    )
    result: dict[str, object] = {"samples": len(records), "metrics": {}}
    for metric in metrics:
        values = [float(record[metric]) for record in records if metric in record]
        if not values:
            continue
        scale = 1000 if metric.endswith("latency") else 1
        unit = "ms" if scale == 1000 else "tokens/s"
        scaled = [value * scale for value in values]
        result["metrics"][metric] = {
            "unit": unit,
            "median": statistics.median(scaled),
            "mean": statistics.fmean(scaled),
            "min": min(scaled),
            "max": max(scaled),
            "pstdev": statistics.pstdev(scaled),
        }
    return result


def compare_variants(variant_summaries: dict[str, dict]) -> dict:
    if not {"baseline", "purlin"}.issubset(variant_summaries):
        return {}
    comparison = {}
    baseline = variant_summaries["baseline"]["metrics"]
    purlin = variant_summaries["purlin"]["metrics"]
    for metric in baseline.keys() & purlin.keys():
        baseline_value = baseline[metric]["median"]
        purlin_value = purlin[metric]["median"]
        if not baseline_value or not purlin_value:
            continue
        lower_is_better = metric.endswith("latency")
        speedup = (
            baseline_value / purlin_value
            if lower_is_better
            else purlin_value / baseline_value
        )
        comparison[metric] = {
            "baseline_median": baseline_value,
            "purlin_median": purlin_value,
            "unit": baseline[metric]["unit"],
            "purlin_change_percent": 100
            * (purlin_value - baseline_value)
            / baseline_value,
            "purlin_speedup": speedup,
        }
    return comparison


def write_summary(args: argparse.Namespace) -> dict:
    variants = {}
    for variant in args.variants:
        clean_path = args.output_dir / f"{variant}_clean.jsonl"
        if clean_path.exists():
            variants[variant] = summarize_samples(read_jsonlines(clean_path))
    result = {
        "schema_version": 1,
        "scope": (
            "static ModelRunner microbenchmark; values exclude serving, scheduling, "
            "HTTP, and tokenization latency and are not client TTFT"
        ),
        "variants": variants,
        "comparison": compare_variants(variants),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    for variant, summary in variants.items():
        metrics = summary["metrics"]
        prefill = metrics["prefill_latency"]["median"]
        decode = metrics.get("median_decode_latency", {}).get("median")
        decode_text = f"{decode:.3f} ms" if decode is not None else "n/a"
        print(
            f"{variant}: median prefill={prefill:.3f} ms, "
            f"median decode={decode_text} ({summary['samples']} samples)",
            flush=True,
        )
    return result


def write_comparison(args: argparse.Namespace) -> dict | None:
    if not {"baseline", "purlin"}.issubset(args.variants):
        return None
    if any(not path.exists() for path in comparison_inputs(args.output_dir)):
        return None
    result = build_comparison_from_directory(args.output_dir)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote static comparison: {args.output_dir / 'comparison.json'}", flush=True)
    return result


def build_manifest(args: argparse.Namespace, cuda_graph_trace_mode: str | None) -> dict:
    return {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": (
            "static ModelRunner microbenchmark; not serving TTFT or end-to-end latency"
        ),
        "model": args.model,
        "revision": args.revision,
        "parallelism": {"tp": args.tp, "dp": args.dp, "ep": args.ep},
        "workload": {
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "repeats": args.repeats,
        },
        "runtime": {
            "mem_fraction_static": args.mem_fraction_static,
            "attention_backend": (
                None
                if args.use_server_default_attention_backend
                else args.attention_backend
            ),
            "enable_dp_attention": args.enable_dp_attention,
            "trust_remote_code": args.trust_remote_code,
            "cuda_graph_backend_prefill": args.cuda_graph_backend_prefill,
            "cuda_graph_backend_decode": args.cuda_graph_backend_decode,
            "cuda_graph_bs_prefill": args.cuda_graph_bs_prefill,
            "extra_args": args.server_arg,
        },
        "profiling": {
            "enabled": not args.skip_profile,
            "stages": list(args.profile_stages),
            "profile_start_step": args.profile_start_step,
            "profile_steps": args.profile_steps,
            "nsys_command": str(args.nsys_command),
            "nsys_cuda_graph_trace": cuda_graph_trace_mode,
            "additional_communication_kernel_names": args.communication_pattern,
        },
        "variants": list(args.variants),
    }


def write_manifest(args: argparse.Namespace, cuda_graph_trace_mode: str | None) -> None:
    manifest = build_manifest(args, cuda_graph_trace_mode)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def validate_resume_manifest(
    args: argparse.Namespace, cuda_graph_trace_mode: str | None, manifest_path: Path
) -> None:
    actual = json.loads(manifest_path.read_text())
    expected = build_manifest(args, cuda_graph_trace_mode)
    expected["created_utc"] = actual.get("created_utc")
    if actual != expected:
        raise ValueError(
            f"Cannot resume {manifest_path}: requested configuration differs "
            "from the recorded experiment"
        )


def run_clean(args: argparse.Namespace, variant: str) -> None:
    result_path = (args.output_dir / f"{variant}_clean.jsonl").resolve()
    log_path = args.output_dir / f"{variant}_clean.log"
    if args.resume and result_path.exists():
        records = read_jsonlines(result_path)
        if len(records) == args.repeats:
            print(f"Reusing completed clean run: {result_path}", flush=True)
            return
        print(
            f"Clean run has {len(records)} of {args.repeats} samples; rerunning it.",
            flush=True,
        )
    ensure_absent([result_path, log_path], args.overwrite or args.resume)
    run_streamed(
        one_batch_command(
            args, variant, result_path, repeats=args.repeats, profile_stage=None
        ),
        log_path,
    )


def run_profile(
    args: argparse.Namespace,
    variant: str,
    stage: str,
    cuda_graph_trace_mode: str | None,
) -> None:
    prefix = f"{variant}_{stage}"
    report_base = (args.output_dir / prefix).resolve()
    report = report_base.with_suffix(".nsys-rep")
    sqlite_path = report_base.with_suffix(".sqlite")
    result_path = args.output_dir / f"{prefix}_one_batch.jsonl"
    profile_log = args.output_dir / f"{prefix}_profile.log"
    export_log = args.output_dir / f"{prefix}_export.log"
    breakdown_path = args.output_dir / f"{prefix}_breakdown.json"
    analysis_log = args.output_dir / f"{prefix}_analysis.log"
    artifacts = [
        report,
        sqlite_path,
        result_path,
        profile_log,
        export_log,
        breakdown_path,
        analysis_log,
    ]
    if args.resume and all(path.exists() for path in artifacts):
        print(f"Reusing completed {variant} {stage} trace.", flush=True)
        return
    ensure_absent(
        artifacts,
        args.overwrite or args.resume,
    )

    command = one_batch_command(
        args,
        variant,
        result_path.resolve(),
        repeats=1,
        profile_stage=stage,
    )
    run_streamed(
        nsys_command(args, command, report_base, cuda_graph_trace_mode),
        profile_log,
    )
    if not report.exists():
        raise FileNotFoundError(f"Nsight did not create {report}")

    run_logged(
        [
            str(args.nsys_command),
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            f"--output={sqlite_path.resolve()}",
            str(report),
        ],
        export_log,
    )
    analysis_command = [
        str(args.python),
        str(SCRIPT_DIR / "analyze_one_batch_trace.py"),
        str(sqlite_path),
        "--label",
        variant,
        "--stage",
        stage,
        "--output",
        str(breakdown_path),
    ]
    for pattern in args.communication_pattern:
        analysis_command.extend(("--communication-pattern", pattern))
    run_streamed(analysis_command, analysis_log)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("baseline", "purlin"),
        default=("baseline", "purlin"),
    )
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument(
        "--enable-dp-attention", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.88)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--use-server-default-attention-backend", action="store_true")
    parser.add_argument(
        "--cuda-graph-backend-prefill",
        choices=("full", "breakable", "tc_piecewise", "disabled"),
        default="breakable",
    )
    parser.add_argument(
        "--cuda-graph-backend-decode",
        choices=("full", "breakable", "tc_piecewise", "disabled"),
        default="full",
    )
    parser.add_argument("--cuda-graph-bs-prefill", type=int, nargs="+")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument(
        "--profile-stages",
        nargs="+",
        choices=("prefill", "decode"),
        default=("prefill", "decode"),
    )
    parser.add_argument("--profile-start-step", type=int)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete artifacts and rerun only missing or partial work.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra one_batch/server argument; use --server-arg=--flag for flags.",
    )
    parser.add_argument(
        "--communication-pattern",
        action="append",
        default=[],
        help="Additional reviewed exact communication-kernel name.",
    )
    parser.add_argument(
        "--python", type=Path, default=REPO_ROOT / ".venv" / "bin" / "python"
    )
    parser.add_argument("--nsys-command", type=Path, default=DEFAULT_NSYS)
    parser.add_argument(
        "--nsys-cuda-graph-trace",
        choices=("auto", "off", "node", "graph"),
        default="auto",
        help="Use node for kernel-level CUDA-graph breakdowns; auto probes Nsight.",
    )
    args = parser.parse_args()

    for name in ("tp", "dp", "ep", "batch_size", "input_len", "output_len", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.profile_steps < 1:
        parser.error("--profile-steps must be positive")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.cuda_graph_bs_prefill and any(
        value < 1 for value in args.cuda_graph_bs_prefill
    ):
        parser.error("--cuda-graph-bs-prefill values must be positive")
    if (
        args.cuda_graph_bs_prefill is None
        and args.cuda_graph_backend_prefill != "disabled"
    ):
        args.cuda_graph_bs_prefill = [args.batch_size * args.input_len]
    if args.profile_start_step is None:
        args.profile_start_step = min(5, max(0, args.output_len - 2))
    if (
        not args.skip_profile
        and "decode" in args.profile_stages
        and args.profile_start_step + args.profile_steps > args.output_len - 1
    ):
        parser.error("decode profile range must fit in the output_len - 1 decode steps")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cuda_graph_trace_mode = (
        None if args.skip_profile else resolve_cuda_graph_trace_mode(args)
    )
    manifest_path = args.output_dir / "manifest.json"
    summary_path = args.output_dir / "summary.json"
    comparison_path = args.output_dir / "comparison.json"
    metadata_paths = [summary_path, comparison_path]
    if not args.resume:
        metadata_paths.append(manifest_path)
    ensure_absent(
        metadata_paths,
        args.overwrite or args.resume,
    )
    if not (args.resume and manifest_path.exists()):
        write_manifest(args, cuda_graph_trace_mode)
    else:
        validate_resume_manifest(args, cuda_graph_trace_mode, manifest_path)
        print(f"Reusing manifest: {manifest_path}", flush=True)

    for variant in args.variants:
        if not args.skip_clean:
            run_clean(args, variant)
        if not args.skip_profile:
            for stage in args.profile_stages:
                run_profile(args, variant, stage, cuda_graph_trace_mode)
    write_summary(args)
    write_comparison(args)
    print(f"Completed static one-batch suite: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
