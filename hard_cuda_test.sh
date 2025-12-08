#!/usr/bin/env bash
# hard_gpu_test.sh
#
# Benchmark script that stresses the CUDA histogram path with a multi-phrase
# workload and parses the "took N ms" timings from --perf-test output.
#
# Usage:
#   ./hard_gpu_test.sh [binary] [logfile] [threads]
#
# Defaults:
#   binary  = ./build/bin/prir
#   logfile = ./access.log
#   threads = 16

set -euo pipefail

BINARY=${1:-./build/bin/prir}
LOG=${2:-./access.log}
THREADS=${3:-16}

if [[ ! -x "$BINARY" ]]; then
  echo "ERROR: Binary not found or not executable: $BINARY" >&2
  exit 1
fi

if [[ ! -f "$LOG" ]]; then
  echo "ERROR: Log file not found: $LOG" >&2
  exit 1
fi

# Multi-phrase set chosen from your existing scenarios to create more work
# for the histogram (more buckets, more hits).
PHRASES=(
  "GET"
  "POST"
  " 200 "
  " 404 "
  " 500 "
  "bingbot"
  "googlebot"
  "AhrefsBot"
)

CMD=("$BINARY" "--file" "$LOG" "--no-stats" "--threads" "$THREADS" "--perf-test")
for p in "${PHRASES[@]}"; do
  CMD+=("--phrase" "$p")
done

echo "=== hard_gpu_test.sh ==="
echo "Binary : $BINARY"
echo "Log    : $LOG"
echo "Threads: $THREADS"
echo "Phrases: ${PHRASES[*]}"
echo
echo "Running perf-test (CPU baseline + CUDA pass)..."
echo "Command: ${CMD[*]}"
echo

# Run and capture all output (stdout+stderr)
OUT="$("${CMD[@]}" 2>&1 || true)"

echo "----- [perf-test output] -----"
echo "$OUT" | grep '^\[perf-test\]' || echo "No [perf-test] lines found."
echo "------------------------------"
echo

# Parse ms timings
CPU_MS=$(
  echo "$OUT" |
    sed -n 's/.*\[perf-test\] CPU baseline took \([0-9]\+\) ms.*/\1/p' |
    head -n1
)

GPU_MS=$(
  echo "$OUT" |
    sed -n 's/.*\[perf-test\] CUDA pass took \([0-9]\+\) ms.*/\1/p' |
    head -n1
)

if [[ -z "${CPU_MS:-}" || -z "${GPU_MS:-}" ]]; then
  echo "ERROR: Failed to parse ms timings from perf-test output." >&2
  echo "CPU_MS='$CPU_MS', GPU_MS='$GPU_MS'" >&2
  exit 1
fi

echo "Timing summary:"
echo "  CPU baseline: ${CPU_MS} ms"
echo "  CUDA pass   : ${GPU_MS} ms"

awk -v c="$CPU_MS" -v g="$GPU_MS" '
BEGIN {
  if (c <= 0 || g <= 0) {
    printf("  (cannot compute speedup: non-positive timings)\n");
    exit 0;
  }
  speedup = c / g;
  delta   = c - g;
  printf("  Speedup (CPU/GPU): %.2fx\n", speedup);
  printf("  Absolute difference: %d ms\n", int(delta));
  if (g < c)
    printf("  Verdict: GPU faster in this scenario.\n");
  else
    printf("  Verdict: GPU not faster (likely CPU-bound workload).\n");
}
'

exit 0