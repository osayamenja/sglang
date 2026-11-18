"""Launch one node's SGLang server and finish its Nsight report before exiting."""

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path

from run_suite import nsys_server_command, shutdown_ack_files, stop_process_group


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-file", required=True, type=Path)
    parser.add_argument("--launcher-nodes", type=int, required=True)
    parser.add_argument("--report-base", type=Path)
    parser.add_argument("--nsys-command", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    node = int(os.environ["SLURM_NODEID"])
    command = [*args.command[1:], "--node-rank", str(node)]
    if args.report_base is not None:
        base = args.report_base.with_name(f"{args.report_base.name}_node{node}")
        command = nsys_server_command(args.nsys_command, command, base)

    with tempfile.TemporaryDirectory(
        dir=os.environ.get("SLURM_TMPDIR", "/tmp"), prefix="sglang-triton-"
    ) as cache:
        process = subprocess.Popen(
            command,
            env={**os.environ, "TRITON_CACHE_DIR": cache},
            start_new_session=True,
        )
        try:
            while not args.stop_file.exists():
                try:
                    status = process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    continue
                raise RuntimeError(
                    f"Server on node {node} exited before shutdown (status {status})"
                )
            if (status := process.poll()) is not None:
                raise RuntimeError(
                    f"Server on node {node} exited before shutdown (status {status})"
                )

            # Each launcher must enter teardown before any peer can lose its
            # distributed connection as a result of an intentional SIGINT.
            acknowledgements = shutdown_ack_files(args.stop_file, args.launcher_nodes)
            acknowledgements[node].touch()
            deadline = time.monotonic() + 120
            while missing := [ack for ack in acknowledgements if not ack.exists()]:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Missing shutdown acknowledgements: {missing}")
                time.sleep(0.25)
        finally:
            stop_process_group(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
