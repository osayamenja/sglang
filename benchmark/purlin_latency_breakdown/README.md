# Purlin latency-breakdown suite

This suite compares an unmodified SGLang server with `--enable-purlin` and
produces three additive model-execution breakdowns:

- TTFT: model critical-path time experienced before the first token;
- TPOT: model critical-path time experienced after the first token, divided by
  the request's actual number of inter-token intervals;
- E2E: model critical-path time experienced over the complete request.

Each measured response records the DP rank that served it. For every traced
scheduler step, the analyzer considers only devices whose marker topology has
the matching attention-DP rank, then selects the longest-running device within
that group. An idle DP rank can therefore remain visible in audit data without
being used for the request's reported metrics. Compute is the union of
non-communication kernel intervals on the selected device. Communication is
the remaining step wall time, so collective execution, dependency waits, and
communication-induced synchronization stay in one bucket and
`total = compute + communication`.

More formally, for scheduler step `s` and request DP rank `d`, the analyzer
selects `r*` from the marked topology group `attn_dp_rank == d` and computes:

```text
total(s)         = last_kernel_end(s, r*) - first_kernel_start(s, r*)
compute(s)       = duration(union(non_communication_kernel_intervals(s, r*)))
communication(s) = total(s) - compute(s)
```

At concurrency one, phase groups map unambiguously to requests. Metric values
are assembled without overlap:

```text
TTFT = sum(all chunked-prefill steps, including first-token sampling)
TPOT = sum(the output_len - 1 client-timed decode steps) / (output_len - 1)
E2E  = TTFT + sum(the output_len - 1 client-timed decode steps)
```

At higher concurrency, scheduler steps contain several requests, so phase
groups cannot be assigned to one request. The analyzer instead uses the
benchmark's per-request `send_times`, `ttfts`, and `itls`, together with
Nsight's `TARGET_INFO_SESSION_START_TIME.systemClockNs`, to place every client
window in the trace clock:

```text
trace_ns(client_time) = round(client_time * 1e9) - systemClockNs
TTFT window            = [send, send + ttft]
decode window          = [send + ttft, send + ttft + sum(itls)]
E2E window             = TTFT window + decode window
```

The analyzer constructs a separate critical-step timeline for every
attention-DP rank. Each request window is intersected only with the timeline
matching its returned DP rank. This measures experienced latency: model work
for co-batched or preceding requests is included when it occupies that model
path while the request is outstanding. Time outside `scheduler.run_batch`
remains an explicit `outside_model` validation residual and is never mislabeled
as communication.

Communication is therefore an operational communication/synchronization
bucket, not merely the sum of NCCL or Purlin kernel residency. Kernel-name
classification determines what is excluded from compute; the complement also
captures collective dependency waits, inter-stream synchronization, and launch
gaps on the communication path. This is the requested binary decomposition and
is appropriate here because communication and compute do not meaningfully
overlap. By default, an overlap above 25 microseconds in any step or request
window is a hard validation failure; smaller timestamp-scale slivers are
tolerated. Use `--overlap-tolerance-ns` to supply a hardware-calibrated limit.
A failure reports the observed overlap, configured threshold, excess, and the
offending device step or request window in both nanoseconds and microseconds.

The breakdown is deliberately model-side. Client latency also contains HTTP,
tokenization, scheduling outside model forwards, and detokenization; clean
client TTFT/TPOT/E2E values are retained as production references rather than
misclassifying that residual as communication.

## Interpretation

`summarize_results.py` describes baseline/Purlin changes separately for TTFT,
TPOT, and E2E. Its wording is derived from each signed delta: increased,
decreased, or unchanged. It does not infer a mechanism or claim a consistent
direction when the metrics are mixed. Stronger statistical or causal claims
require repeated experiments and separate evidence.

## Prerequisites

- Install the checkout and Purlin first (`scripts/install_purlin.sh`).
- Install Nsight Systems with CUDA graph node/NVTX precapture support. This
  suite was validated with Nsight Systems 2026.4.1.
- Ensure the selected port is unused and the model is already accessible.

## Example

```bash
.venv/bin/python benchmark/purlin_latency_breakdown/run_suite.py \
  --model Qwen/Qwen3.5-122B-A10B \
  --revision dc4d348443bc740c68e2d77492492c11606384d5 \
  --tp 8 --dp 4 --ep 8 \
  --input-len 4096 --output-len 256 \
  --concurrency 1 --num-prompts 8 --trace-prompts 8 \
  --output-dir benchmark_results/my_machine_qwen
```

For a quick mechanistic trace on another GPU type, use `--trace-prompts 1`
while retaining eight clean requests. For a request-level breakdown matching
the clean sample count, use `--trace-prompts 8`. Trace size grows approximately
linearly with output length and trace-prompt count.

At concurrency `c > 1`, use at least `--trace-prompts c` to reach the requested
load. The production protocol uses `--num-prompts 8*c`; use the same value for
`--trace-prompts` when a fully request-matched trace is desired. For example:

```bash
.venv/bin/python benchmark/purlin_latency_breakdown/run_suite.py \
  --model Qwen/Qwen3.5-122B-A10B \
  --revision dc4d348443bc740c68e2d77492492c11606384d5 \
  --tp 8 --dp 4 --ep 8 \
  --input-len 4096 --output-len 256 \
  --concurrency 16 --num-prompts 128 --trace-prompts 128 \
  --warmup-requests 16 \
  --output-dir benchmark_results/a100_qwen_c16
```

Hardware and model changes are normal command-line parameters:

```bash
.venv/bin/python benchmark/purlin_latency_breakdown/run_suite.py \
  --model <model> --revision <optional-revision> \
  --tp <tp> --dp <dp> --ep <ep> \
  --mem-fraction-static 0.88 \
  --attention-backend triton \
  --chunked-prefill-size 4096 \
  --cuda-graph-max-bs-decode 256 \
  --input-len <input-tokens> --output-len <output-tokens> \
  --num-prompts 8 --trace-prompts 1 \
  --output-dir <result-directory>
```

Use repeated `--server-arg=--flag` or `--client-arg=--flag` options for
model-specific switches. A flag and value can normally be passed as one token,
for example `--server-arg=--some-option=some-value`; a boolean flag uses
`--server-arg=--some-flag`. These arguments are appended unchanged to both the
baseline and Purlin commands. The only automatic variant difference is
`--enable-purlin`.

### B300 example

The following translates the B300 server configuration for
`nvidia/DeepSeek-V4-Pro-NVFP4` into a complete suite invocation:

```bash
.venv/bin/python benchmark/purlin_latency_breakdown/run_suite.py \
  --model nvidia/DeepSeek-V4-Pro-NVFP4 \
  --tp 8 --dp 4 --ep 8 \
  --mem-fraction-static 0.90 \
  --use-server-default-attention-backend \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs-decode 256 \
  --watchdog-timeout 3600 \
  --server-host 0.0.0.0 \
  --port 30000 \
  --server-arg=--moe-runner-backend=flashinfer_trtllm_routed \
  --server-arg=--disable-flashinfer-autotune \
  --server-arg=--swa-full-tokens-ratio=0.1 \
  --input-len <input-tokens> --output-len <output-tokens> \
  --concurrency 1 --num-prompts 8 --trace-prompts 1 \
  --output-dir benchmark_results/b300_deepseek_v4_pro_nvfp4
```

`--enable-dp-attention` and `--trust-remote-code` are enabled by default, so
they need not be repeated. The runner also defaults to host `0.0.0.0`, port
`30000`, CUDA-graph max decode batch size `256`, and watchdog timeout `3600`;
they are written explicitly above to make the translation auditable.

The attention-backend distinction is intentional:

- For the validated A100 setup, omit
  `--use-server-default-attention-backend`; the runner explicitly supplies its
  default `--attention-backend triton`, which is required for that server to
  start correctly.
- For the supplied B300 setup, use
  `--use-server-default-attention-backend`; this omits the server argument and
  preserves SGLang's model/GPU-specific backend selection.

Communication classification is deliberately explicit. Names beginning with
`ncclDevKernel_`, the known SGLang custom-collective names, and the known Purlin
collective names are communication; everything else is compute. A future
collective-looking but unrecognized name is a hard failure. If another backend
needs an addition, pass its reviewed exact name with
`--communication-pattern <kernel-name>` and verify the emitted diagnostics.

## What the runner does

For each of `baseline` and `purlin`, the runner:

1. Starts a production server without NVTX or Nsight and runs the clean client.
2. Starts a fresh server with only `SGLANG_ENABLE_NVTX_SCHEDULER=1` and runs the
   marker-only calibration.
3. Starts another fresh server under Nsight with the same scheduler markers.
4. In every client mode, runs concurrency-bounded warmups and waits for an
   all-rank cache flush with `empty_cache=false`.
5. Starts CUDA profiling for the full-profile mode.
6. Sends the same two-token primer sequentially to every DP rank, routing only
   these primer requests explicitly.
7. Waits for a second all-rank flush, leaving measured requests naturally
   routed and requiring each to report zero cached tokens and one stable DP
   rank.
8. For the full profile, broadcasts a `measurement.begin` NVTX marker carrying
   the unique run ID and scheduler topology.
9. Runs the measured requests and records the benchmark end timestamp as soon
   as all measured responses finish.
10. Uses a post-run flush as an idle barrier, emits `measurement.end`, and then
    stops profiling. None of these cleanup operations contributes to benchmark
    duration.
11. Stops the server, exports the `.nsys-rep` to SQLite, creates lookup indexes,
    and runs `analyze_full_trace.py`.

Fresh marker and capture servers prevent scheduler/radix state from one full
high-concurrency run from entering the next warmup.

The portable suite derives TPOT from this same complete-request trace, so TTFT,
TPOT, and E2E use one capture and one critical-path definition. The A100 pilot
report also retains an earlier bounded 59-replay TPOT experiment as its headline
TPOT result because that was the originally requested and separately validated
measurement. Its full-request TPOT cross-check is reported alongside it. New
machine runs from this suite should compare against the suite's full-request
TPOT output rather than mixing the two capture designs.

Layerwise NVTX markers are intentionally disabled because they can materially
perturb prefill latency. Scheduler NVTX supplies request-step boundaries while
CUDA runtime correlation IDs associate each boundary with its GPU work. The
begin/end markers delimit measured work independently on every scheduler PID.
Ranges wholly outside those per-device boundaries are reported as discarded
prefix/suffix validation counts. Missing, duplicate, crossing, or imbalanced
measured ranges remain hard failures.

With SGLang's default overlap scheduler, concurrency-one traces contain one
final graph replay per request that drains speculative overlapped work after
the last client-timed decode result. Phase attribution identifies and excludes
that replay. Concurrent attribution is bounded by client token windows, so
work entirely after the last token is excluded automatically.

To reuse clean files from an earlier run, pass `--skip-clean`. To run only one
system, pass `--variants baseline` or `--variants purlin`. If a completed
variant should be preserved after an interrupted suite, rerun the missing
variant with `--resume`.

## Script layout

- `run_suite.py` is the entry point. It builds the baseline and Purlin server
  commands, waits for readiness, runs the clean/marker/profile clients, manages
  server shutdown, exports Nsight to SQLite, invokes the analyzer, and writes a
  manifest and final comparison.
- `analyze_full_trace.py` is the trace engine. It maps each
  `scheduler.run_batch` NVTX range to CUDA kernels through runtime correlation
  IDs, validates marker boundaries and topology, and chooses the critical
  device within the request's attention-DP group. At concurrency one it uses
  prefill/CUDA-graph phase groups and removes the overlap scheduler's drain
  replay. At higher concurrency it clock-aligns each request with its own DP
  timeline.
- `summarize_results.py` joins clean client metrics, marker-only calibration,
  profiled client metrics, and traced model components into `comparison.json`.

All three programs use only the Python standard library. The external work is
performed by the checkout's `sglang`/Python executables and the `nsys` command,
whose paths can be overridden on the command line.

## Key artifacts

For each variant, the output directory contains:

- `<variant>_clean.jsonl`: uninstrumented production metrics;
- `<variant>_scheduler_markers.jsonl`: scheduler-marker calibration;
- `<variant>_full_profile.jsonl`: client metrics during the complete trace;
- `<variant>_full.nsys-rep` and `.sqlite`: raw and exported traces;
- `<variant>_full_breakdown.json`: TTFT, TPOT, and E2E decompositions;
- server, client, export, and analyzer logs;
- `manifest.json`: model, parallelism, and workload parameters;
- `comparison.json`: raw component differences, signed data-dependent
  observations, and overhead checks when both variants and clean runs exist.

Concurrent request-window attribution requires the client and profiled server
to share the same monotonic system clock. `run_suite.py` satisfies this by
launching both locally. The output records per-metric model coverage and
`outside_model` residuals so clock alignment and model/client agreement can be
audited directly.
