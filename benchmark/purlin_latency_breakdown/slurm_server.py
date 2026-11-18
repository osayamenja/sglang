"""Launch one node's SGLang server and finish its Nsight report before exiting."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from run_suite import nsys_server_command, stop_process_group


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-file", required=True, type=Path)
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
                if not args.stop_file.exists():
                    raise RuntimeError(
                        f"Server on node {node} exited before shutdown (status {status})"
                    )
        finally:
            stop_process_group(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
