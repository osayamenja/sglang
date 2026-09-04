#!/usr/bin/env python3

"""Run portable baseline/Purlin latency-breakdown experiments.

The suite records:

1. a clean production run for client TTFT, TPOT, and E2E latency;
2. a scheduler-NVTX-only run while Nsight capture is inactive;
3. a full-request CUDA/NVTX capture, stopped only after every response arrives;
4. an Nsight SQLite export and additive TTFT/TPOT/E2E model-time analysis.

At concurrency one, the analyzer assigns prefill/decode phase groups directly
to requests. At higher concurrency, it aligns client request timestamps with
the Nsight session clock. In both modes it selects critical-path components
only from the attention-DP group that served each request.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OVERLAP_TOLERANCE_NS = 25_000


def format_command(command: list[str]) -> str:
    return shlex.join(command)


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


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def tail(path: Path, line_count: int = 30) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[-line_count:])


def wait_for_server(
    process: subprocess.Popen,
    host: str,
    port: int,
    timeout_seconds: int,
    log_path: Path,
) -> None:
    url = f"http://{host}:{port}/health_generate"
    deadline = time.monotonic() + timeout_seconds
    next_update = time.monotonic()
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Server exited with status {return_code} before readiness.\n"
                f"Last server log lines:\n{tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print(f"Server ready: {url}", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        now = time.monotonic()
        if now >= next_update:
            elapsed = timeout_seconds - max(0, int(deadline - now))
            print(
                f"Waiting for server readiness ({elapsed}s); log: {log_path}",
                flush=True,
            )
            next_update = now + 30
        time.sleep(2)
    raise TimeoutError(
        f"Server did not become ready within {timeout_seconds}s.\n"
        f"Last server log lines:\n{tail(log_path)}"
    )


def stop_process_group(process: subprocess.Popen, timeout_seconds: int = 900) -> None:
    if process.poll() is not None:
        return
    print(f"Stopping server process group {process.pid}...", flush=True)
    os.killpg(process.pid, signal.SIGINT)
    deadline = time.monotonic() + timeout_seconds
    next_update = time.monotonic() + 30
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            print(f"Server stopped with status {return_code}.", flush=True)
            return
        if time.monotonic() >= next_update:
            print("Waiting for server/Nsight trace finalization...", flush=True)
            next_update += 30
        time.sleep(1)

    print("Server did not stop after SIGINT; sending SIGTERM.", flush=True)
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        print("Server did not stop after SIGTERM; sending SIGKILL.", flush=True)
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


@contextlib.contextmanager
def server_session(
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    ready_host: str,
    port: int,
    ready_timeout: int,
) -> Iterator[subprocess.Popen]:
    print(f"Starting server:\n  {format_command(command)}", flush=True)
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            wait_for_server(process, ready_host, port, ready_timeout, log_path)
            yield process
        finally:
            stop_process_group(process)


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


def common_server_command(args: argparse.Namespace, variant: str) -> list[str]:
    command = [
        str(args.sglang_command),
        "serve",
        "--model-path",
        args.model,
        "--tp",
        str(args.tp),
        "--dp",
        str(args.dp),
        "--ep",
        str(args.ep),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
    ]
    if not args.use_server_default_attention_backend:
        command.extend(("--attention-backend", args.attention_backend))
    command.extend(
        [
            "--chunked-prefill-size",
            str(args.chunked_prefill_size),
            "--cuda-graph-max-bs-decode",
            str(args.cuda_graph_max_bs_decode),
            "--watchdog-timeout",
            str(args.watchdog_timeout),
            "--host",
            args.server_host,
            "--port",
            str(args.port),
        ]
    )
    if args.revision:
        command.extend(("--revision", args.revision))
    if args.enable_dp_attention:
        command.append("--enable-dp-attention")
    if args.enable_prefill_delayer:
        command.append("--enable-prefill-delayer")
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.cuda_graph_backend_prefill is not None:
        command.extend(
            ("--cuda-graph-backend-prefill", args.cuda_graph_backend_prefill)
        )
    if args.cuda_graph_bs_prefill:
        command.append("--cuda-graph-bs-prefill")
        command.extend(str(value) for value in args.cuda_graph_bs_prefill)
    if variant == "purlin":
        command.append("--enable-purlin")
    command.extend(args.server_arg)
    return command


def client_command(
    args: argparse.Namespace,
    output_file: Path,
    num_prompts: int,
    profile: bool,
    run_id: str,
) -> list[str]:
    command = [
        str(args.python),
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--host",
        args.client_host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--dataset-name",
        "random",
        "--random-range-ratio",
        str(args.random_range_ratio),
        "--seed",
        str(args.seed),
        "--random-input-len",
        str(args.input_len),
        "--random-output-len",
        str(args.output_len),
        "--max-concurrency",
        str(args.concurrency),
        "--num-prompts",
        str(num_prompts),
        "--warmup-requests",
        str(args.warmup_requests),
        "--flush-cache",
        "--cache-report",
        "--require-zero-cached-tokens",
        "--require-dp-rank",
        "--measurement-run-id",
        run_id,
        "--measurement-dp-size",
        str(args.dp),
        "--output-details",
        "--output-file",
        str(output_file),
    ]
    if args.tokenize_prompt:
        command.append("--tokenize-prompt")
    if profile:
        # With no profile-num-steps, the serving benchmark calls /stop_profile
        # only after every response has completed. cudaProfilerStop therefore
        # cannot contaminate any request TTFT or E2E measurement.
        command.extend(("--profile", "--profile-activities", "CUDA_PROFILER"))
    command.extend(args.client_arg)
    return command


def clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("SGLANG_ENABLE_NVTX", None)
    environment.pop("SGLANG_ENABLE_NVTX_SCHEDULER", None)
    environment.pop("SGLANG_ENABLE_NVTX_OPERATIONS", None)
    return environment


def profile_environment() -> dict[str, str]:
    environment = clean_environment()
    environment["SGLANG_ENABLE_NVTX_SCHEDULER"] = "1"
    return environment


def nsys_server_command(
    args: argparse.Namespace, server_command: list[str], report_base: Path
) -> list[str]:
    return [
        str(args.nsys_command),
        "profile",
        "--force-overwrite=true",
        "--sample=none",
        "--cpuctxsw=none",
        "--trace=cuda,nvtx",
        "--trace-fork-before-exec=true",
        "--cuda-graph-trace=node:nvtx-precapture",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        f"--output={report_base}",
        *server_command,
    ]


def report_capture_quality(summary_path: Path) -> None:
    """Print the profiler-perturbation checks recorded in comparison.json."""

    summary = json.loads(summary_path.read_text())
    print("Capture quality (TPOT):", flush=True)
    for variant, quality in summary["capture_quality"].items():
        print(
            f"  {variant}: profiled client vs markers "
            f"{quality['profiled_client_vs_markers_tpot'] * 100:+.2f}%, "
            "trace model (launch stall excluded) vs markers "
            f"{quality['trace_model_vs_markers_tpot'] * 100:+.2f}%",
            flush=True,
        )
        print(f"    {quality['assessment']}", flush=True)
    for metric, values in summary["metrics"].items():
        stall = values["launch_stall"]
        print(
            f"  {metric.upper()} launch stall separated: baseline "
            f"{stall['baseline_ms']:.3f} ms "
            f"({stall['baseline_fraction_of_raw_total'] * 100:.2f}%), purlin "
            f"{stall['purlin_ms']:.3f} ms "
            f"({stall['purlin_fraction_of_raw_total'] * 100:.2f}%)",
            flush=True,
        )


def write_manifest(args: argparse.Namespace, output_dir: Path) -> None:
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": args.model,
        "revision": args.revision,
        "parallelism": {"tp": args.tp, "dp": args.dp, "ep": args.ep},
        "workload": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "concurrency": args.concurrency,
            "num_prompts": args.num_prompts,
            "trace_prompts": args.trace_prompts,
            "warmup_requests": args.warmup_requests,
            "seed": args.seed,
            "random_range_ratio": args.random_range_ratio,
            "tokenize_prompt": args.tokenize_prompt,
        },
        "server": {
            "mem_fraction_static": args.mem_fraction_static,
            "attention_backend": (
                None
                if args.use_server_default_attention_backend
                else args.attention_backend
            ),
            "use_server_default_attention_backend": (
                args.use_server_default_attention_backend
            ),
            "chunked_prefill_size": args.chunked_prefill_size,
            "cuda_graph_max_bs_decode": args.cuda_graph_max_bs_decode,
            "cuda_graph_backend_prefill": args.cuda_graph_backend_prefill,
            "cuda_graph_bs_prefill": args.cuda_graph_bs_prefill,
            "enable_dp_attention": args.enable_dp_attention,
            "enable_prefill_delayer": args.enable_prefill_delayer,
            "trust_remote_code": args.trust_remote_code,
            "extra_args": args.server_arg,
        },
        "analysis": {
            "overlap_tolerance_ns": args.overlap_tolerance_ns,
            "additional_communication_kernel_names": args.communication_pattern,
            "require_prefill_cuda_graph": args.require_prefill_cuda_graph,
            # comparison.json compares model time with the analyzer's
            # launch_stall bucket excluded; see README "Launch-stall correction".
            "launch_stall_correction": True,
        },
        "variants": args.variants,
        "skip_clean": args.skip_clean,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def run_variant(args: argparse.Namespace, output_dir: Path, variant: str) -> None:
    base_server = common_server_command(args, variant)
    clean_jsonl = output_dir / f"{variant}_clean.jsonl"
    clean_client_log = output_dir / f"{variant}_clean.client.log"
    clean_server_log = output_dir / f"{variant}_clean.server.log"

    if not args.skip_clean:
        ensure_absent([clean_jsonl, clean_client_log, clean_server_log], args.overwrite)
        with server_session(
            base_server,
            clean_environment(),
            clean_server_log,
            args.client_host,
            args.port,
            args.ready_timeout,
        ):
            clean_run_id = f"{variant}-clean-{uuid.uuid4().hex}"
            run_streamed(
                client_command(
                    args,
                    clean_jsonl,
                    args.num_prompts,
                    profile=False,
                    run_id=clean_run_id,
                ),
                clean_client_log,
            )

    markers_jsonl = output_dir / f"{variant}_scheduler_markers.jsonl"
    markers_client_log = output_dir / f"{variant}_scheduler_markers.client.log"
    markers_server_log = output_dir / f"{variant}_scheduler_markers.server.log"
    full_jsonl = output_dir / f"{variant}_full_profile.jsonl"
    full_client_log = output_dir / f"{variant}_full_profile.client.log"
    profile_server_log = output_dir / f"{variant}_full_profile.server.log"
    report_base = output_dir / f"{variant}_full"
    report = report_base.with_suffix(".nsys-rep")
    sqlite_path = report_base.with_suffix(".sqlite")
    breakdown = output_dir / f"{variant}_full_breakdown.json"
    export_log = output_dir / f"{variant}_nsys_export.log"
    analysis_log = output_dir / f"{variant}_analysis.log"
    ensure_absent(
        [
            markers_jsonl,
            markers_client_log,
            markers_server_log,
            full_jsonl,
            full_client_log,
            profile_server_log,
            report,
            sqlite_path,
            breakdown,
            export_log,
            analysis_log,
        ],
        args.overwrite,
    )

    # Calibration B: a fresh server with scheduler markers active and no
    # profiler. Keeping calibration separate from the capture server avoids
    # carrying scheduler/radix state from a complete high-concurrency run into
    # the profiled warmup.
    with server_session(
        base_server,
        profile_environment(),
        markers_server_log,
        args.client_host,
        args.port,
        args.ready_timeout,
    ):
        marker_run_id = f"{variant}-markers-{uuid.uuid4().hex}"
        run_streamed(
            client_command(
                args,
                markers_jsonl,
                args.num_prompts,
                profile=False,
                run_id=marker_run_id,
            ),
            markers_client_log,
        )

    # Calibration C and the full trace use another fresh server. Nsight capture
    # remains inactive during warmup and starts only through /start_profile.
    with server_session(
        nsys_server_command(args, base_server, report_base),
        profile_environment(),
        profile_server_log,
        args.client_host,
        args.port,
        args.ready_timeout,
    ):
        profile_run_id = f"{variant}-profile-{uuid.uuid4().hex}"
        run_streamed(
            client_command(
                args,
                full_jsonl,
                args.trace_prompts,
                profile=True,
                run_id=profile_run_id,
            ),
            full_client_log,
        )

    if not report.exists():
        raise FileNotFoundError(f"Nsight did not create {report}")
    export_command = [
        str(args.nsys_command),
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        f"--output={sqlite_path}",
        str(report),
    ]
    run_logged(export_command, export_log)

    analysis_command = [
        str(args.python),
        str(SCRIPT_DIR / "analyze_full_trace.py"),
        str(sqlite_path),
        str(full_jsonl),
        "--label",
        variant,
        "--output",
        str(breakdown),
        "--quiet",
        "--overlap-tolerance-ns",
        str(args.overlap_tolerance_ns),
    ]
    if args.require_prefill_cuda_graph:
        analysis_command.append("--require-prefill-cuda-graph")
    for pattern in args.communication_pattern:
        analysis_command.extend(("--communication-pattern", pattern))
    run_streamed(analysis_command, analysis_log)
    print(f"Completed {variant}: {breakdown}", flush=True)


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
    parser.add_argument(
        "--enable-dp-attention", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--enable-prefill-delayer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the server prefill delayer so attention-DP ranks admit "
            "prefills together for clean TTFT breakdowns."
        ),
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.88)
    parser.add_argument(
        "--attention-backend",
        default="triton",
        help=(
            "Explicit attention backend. Ignored when "
            "--use-server-default-attention-backend is set."
        ),
    )
    parser.add_argument(
        "--use-server-default-attention-backend",
        action="store_true",
        help=(
            "Omit --attention-backend from the server command and let SGLang "
            "select its model/GPU-specific default."
        ),
    )
    parser.add_argument("--chunked-prefill-size", type=int, default=4096)
    parser.add_argument("--cuda-graph-max-bs-decode", type=int, default=256)
    parser.add_argument(
        "--cuda-graph-backend-prefill",
        choices=("breakable", "tc_piecewise", "disabled"),
        default="breakable",
        help="Prefill CUDA-graph backend passed to the server.",
    )
    parser.add_argument(
        "--cuda-graph-bs-prefill",
        type=int,
        nargs="+",
        help="Exact prefill token sizes captured by the selected graph backend.",
    )
    parser.add_argument("--watchdog-timeout", type=float, default=3600)
    parser.add_argument("--input-len", type=int, default=4096)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument(
        "--trace-prompts",
        type=int,
        help="Number of complete requests in the node trace; defaults to --num-prompts.",
    )
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--random-range-ratio", type=float, default=1.0)
    parser.add_argument(
        "--tokenize-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Send random prompts as token IDs so --input-len exactly matches "
            "the server-side prefill shape."
        ),
    )
    parser.add_argument(
        "--require-prefill-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Require every measured active prefill to execute CUDA graph nodes. "
            "Defaults on for an explicitly selected non-disabled prefill backend."
        ),
    )
    parser.add_argument("--server-host", default="0.0.0.0")
    parser.add_argument("--client-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--ready-timeout", type=int, default=3600)
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing manifest and run only missing requested variant artifacts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra server argument; use --server-arg=--flag for values beginning with '-'.",
    )
    parser.add_argument(
        "--client-arg",
        action="append",
        default=[],
        help="Extra client argument; use --client-arg=--flag for values beginning with '-'.",
    )
    parser.add_argument(
        "--communication-pattern",
        action="append",
        default=[],
        help="Additional exact communication-kernel name passed to the analyzer.",
    )
    parser.add_argument(
        "--overlap-tolerance-ns",
        type=int,
        default=DEFAULT_OVERLAP_TOLERANCE_NS,
        help=(
            "Maximum compute/communication overlap allowed per scheduler step "
            "or request window during trace analysis."
        ),
    )
    parser.add_argument(
        "--python", type=Path, default=REPO_ROOT / ".venv" / "bin" / "python"
    )
    parser.add_argument(
        "--sglang-command",
        type=Path,
        default=REPO_ROOT / ".venv" / "bin" / "sglang",
    )
    parser.add_argument("--nsys-command", type=Path, default=Path("nsys"))
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.overlap_tolerance_ns < 0:
        parser.error("--overlap-tolerance-ns must be non-negative")
    if args.cuda_graph_bs_prefill and any(
        value < 1 for value in args.cuda_graph_bs_prefill
    ):
        parser.error("--cuda-graph-bs-prefill values must be positive")
    if args.require_prefill_cuda_graph is None:
        args.require_prefill_cuda_graph = args.cuda_graph_backend_prefill not in (
            None,
            "disabled",
        )
    if args.output_len < 2:
        parser.error("full E2E decomposition requires --output-len >= 2")
    if args.trace_prompts is None:
        args.trace_prompts = args.num_prompts
    if args.trace_prompts < 1 or args.num_prompts < 1:
        parser.error("prompt counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if port_is_open(args.client_host, args.port):
        raise RuntimeError(
            f"Port {args.client_host}:{args.port} is already open; refusing to reuse it"
        )
    manifest_path = args.output_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        print(f"Reusing existing manifest: {manifest_path}", flush=True)
    else:
        ensure_absent([manifest_path], args.overwrite)
        write_manifest(args, args.output_dir)
    for variant in args.variants:
        run_variant(args, args.output_dir, variant)
    summary_inputs = [
        args.output_dir / f"{variant}_{suffix}"
        for variant in ("baseline", "purlin")
        for suffix in (
            "clean.jsonl",
            "scheduler_markers.jsonl",
            "full_profile.jsonl",
            "full_breakdown.json",
        )
    ]
    if all(path.exists() for path in summary_inputs):
        summary_path = args.output_dir / "comparison.json"
        ensure_absent([summary_path], args.overwrite)
        subprocess.run(
            [
                str(args.python),
                str(SCRIPT_DIR / "summarize_results.py"),
                str(args.output_dir),
                "--output",
                str(summary_path),
                "--quiet",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        print(f"Wrote comparison: {summary_path}", flush=True)
        report_capture_quality(summary_path)
    print(f"Suite complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
