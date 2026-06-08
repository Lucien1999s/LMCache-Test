#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
CHUNK_COUNTS="${CHUNK_COUNTS:-2 4 8 16 24 30}"
MAX_NEW_TOKENS_LIST="${MAX_NEW_TOKENS_LIST:-1 16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32648}"
GPU_UTIL="${GPU_UTIL:-0.7}"
GPU_INDEX="${GPU_INDEX:-0}"
REPEATS="${REPEATS:-3}"
RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-0.15}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
MAX_LOCAL_CPU_SIZE_GB="${MAX_LOCAL_CPU_SIZE_GB:-5}"
SLEEP_BETWEEN_REQUESTS="${SLEEP_BETWEEN_REQUESTS:-1.0}"
MEMORY_SAMPLE_INTERVAL_SEC="${MEMORY_SAMPLE_INTERVAL_SEC:-0.05}"
METHODS="${METHODS:-full_recompute lmcache_prefix_chunked_reuse lmcache_naive_segment_reuse lmcache_blend}"
ENABLE_SPARSE="${ENABLE_SPARSE:-0}"
RESET_OUT="${RESET_OUT:-1}"

RESULT_DIR="${RESULT_DIR:-examples/contextflow/results}"
OUT="${OUT:-$RESULT_DIR/contextflow_clean_latency_runs.jsonl}"
SUMMARY_OUT="${SUMMARY_OUT:-$RESULT_DIR/contextflow_clean_latency_summary.csv}"

mkdir -p "$RESULT_DIR" /tmp/cf

if [[ "$RESET_OUT" == "1" ]]; then
  rm -f "$OUT" "$SUMMARY_OUT"
fi

echo "Model: $MODEL"
echo "Chunk size: $CHUNK_SIZE"
echo "Chunk counts: $CHUNK_COUNTS"
echo "Max new tokens: $MAX_NEW_TOKENS_LIST"
echo "Methods: $METHODS"
echo "Repeats: $REPEATS"
echo "GPU index for NVML sampling: $GPU_INDEX"
echo "Run JSONL: $OUT"
echo "Summary CSV: $SUMMARY_OUT"

run_one () {
  local method="$1"
  local chunk_count="$2"
  local max_new_tokens="$3"
  local repeat_id="$4"

  echo
  echo "===== method=$method chunks=$chunk_count max_new_tokens=$max_new_tokens repeat=$repeat_id ====="

  rm -rf /tmp/cf/*
  mkdir -p /tmp/cf

  local sparse_args=()
  if [[ "$ENABLE_SPARSE" == "1" ]]; then
    sparse_args+=(--enable-sparse)
  fi

  TMPDIR=/tmp/cf TMP=/tmp/cf TEMP=/tmp/cf PYTHONHASHSEED="${PYTHONHASHSEED:-0}" \
  PYTHONPATH="$PWD/examples/contextflow/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" examples/contextflow/src/clean_latency.py \
    --method "$method" \
    --repeat-id "$repeat_id" \
    --model "$MODEL" \
    --chunk-count "$chunk_count" \
    --chunk-size-tokens "$CHUNK_SIZE" \
    --max-new-tokens "$max_new_tokens" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --gpu-index "$GPU_INDEX" \
    --recompute-ratio "$RECOMPUTE_RATIO" \
    --lmcache-chunk-size "$LMCACHE_CHUNK_SIZE" \
    --max-local-cpu-size-gb "$MAX_LOCAL_CPU_SIZE_GB" \
    --sleep-between-requests "$SLEEP_BETWEEN_REQUESTS" \
    --memory-sample-interval-sec "$MEMORY_SAMPLE_INTERVAL_SEC" \
    "${sparse_args[@]}" \
    --output "$OUT"
}

for method in $METHODS; do
  for c in $CHUNK_COUNTS; do
    for max_new_tokens in $MAX_NEW_TOKENS_LIST; do
      for repeat_id in $(seq 1 "$REPEATS"); do
        run_one "$method" "$c" "$max_new_tokens" "$repeat_id"
      done
    done
  done
done

"$PYTHON_BIN" examples/contextflow/src/summarize/clean_latency.py \
  "$OUT" \
  --output "$SUMMARY_OUT"

echo
echo "Done."
echo "Run-level JSONL: $OUT"
echo "Summary CSV:    $SUMMARY_OUT"
