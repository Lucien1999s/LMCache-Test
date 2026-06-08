#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32648}"
GPU_UTIL="${GPU_UTIL:-0.7}"
RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-0.15}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
ENABLE_SPARSE="${ENABLE_SPARSE:-0}"
CUDA_SYNC="${CUDA_SYNC:-0}"
PROFILE_MEMORY="${PROFILE_MEMORY:-0}"
RESET_OUT="${RESET_OUT:-1}"

# Format: chunk_count:max_new_tokens:repeats
CONDITIONS="${CONDITIONS:-4:16:5 16:16:5 30:16:5}"

RESULT_DIR="${RESULT_DIR:-examples/contextflow/results}"
OUT="${OUT:-$RESULT_DIR/cacheblend_repair_internal_runs.jsonl}"
PROFILE_OUT="${PROFILE_OUT:-$RESULT_DIR/cacheblend_repair_internal_events.jsonl}"
EXCLUSIVE_EVENTS_OUT="${EXCLUSIVE_EVENTS_OUT:-$RESULT_DIR/cacheblend_repair_internal_exclusive_events.jsonl}"
EXCLUSIVE_SUMMARY_OUT="${EXCLUSIVE_SUMMARY_OUT:-$RESULT_DIR/cacheblend_repair_internal_exclusive_summary.csv}"
STATS_OUT="${STATS_OUT:-$RESULT_DIR/cacheblend_repair_internal_stats.csv}"

mkdir -p "$RESULT_DIR" /tmp/cf

if [[ "$RESET_OUT" == "1" ]]; then
  rm -f "$OUT" "$PROFILE_OUT" "$EXCLUSIVE_EVENTS_OUT" "$EXCLUSIVE_SUMMARY_OUT" "$STATS_OUT"
fi

echo "Model: $MODEL"
echo "Conditions: $CONDITIONS"
echo "Chunk size: $CHUNK_SIZE"
echo "LMCache chunk size: $LMCACHE_CHUNK_SIZE"
echo "Recompute ratio: $RECOMPUTE_RATIO"
echo "CUDA sync: $CUDA_SYNC"
echo "Per-event memory: $PROFILE_MEMORY"
echo "Run JSONL: $OUT"
echo "Fine-event JSONL: $PROFILE_OUT"
echo "Repair exclusive events: $EXCLUSIVE_EVENTS_OUT"
echo "Repair exclusive summary: $EXCLUSIVE_SUMMARY_OUT"
echo "Repair stats: $STATS_OUT"

run_one () {
  local chunk_count="$1"
  local max_new_tokens="$2"
  local repeat_id="$3"
  local run_id="lmcache_blend-repair-internal-c${chunk_count}-tok${max_new_tokens}-r${repeat_id}-$(date +%s%N)"

  echo
  echo "===== method=lmcache_blend chunks=$chunk_count max_new_tokens=$max_new_tokens repeat=$repeat_id ====="

  rm -rf /tmp/cf/*
  mkdir -p /tmp/cf

  local sparse_args=()
  if [[ "$ENABLE_SPARSE" == "1" ]]; then
    sparse_args+=(--enable-sparse)
  fi

  local profile_args=(
    --stage-profile-output "$PROFILE_OUT"
    --stage-profile-run-id "$run_id"
  )

  local profile_mode_args=()
  if [[ "$CUDA_SYNC" == "1" ]]; then
    profile_mode_args+=(--stage-profile-cuda-sync)
  fi
  if [[ "$PROFILE_MEMORY" != "1" ]]; then
    profile_mode_args+=(--stage-profile-no-memory)
  fi

  TMPDIR=/tmp/cf TMP=/tmp/cf TEMP=/tmp/cf \
  CONTEXTFLOW_FINE_BREAKDOWN=1 \
  LMCACHE_CONTEXTFLOW_FINE_BREAKDOWN=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  PYTHONPATH="$PWD/examples/contextflow/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" examples/contextflow/src/synthetic_runner.py \
    --method lmcache_blend \
    --model "$MODEL" \
    --chunk-count "$chunk_count" \
    --chunk-size-tokens "$CHUNK_SIZE" \
    --max-new-tokens "$max_new_tokens" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --recompute-ratio "$RECOMPUTE_RATIO" \
    --lmcache-chunk-size "$LMCACHE_CHUNK_SIZE" \
    "${profile_args[@]}" \
    "${profile_mode_args[@]}" \
    "${sparse_args[@]}" \
    --output "$OUT"
}

for condition in $CONDITIONS; do
  IFS=: read -r chunk_count max_new_tokens repeats <<< "$condition"
  if [[ -z "${chunk_count:-}" || -z "${max_new_tokens:-}" || -z "${repeats:-}" ]]; then
    echo "Invalid condition '$condition'. Expected chunk_count:max_new_tokens:repeats" >&2
    exit 1
  fi
  for repeat_id in $(seq 1 "$repeats"); do
    run_one "$chunk_count" "$max_new_tokens" "$repeat_id"
  done
done

"$PYTHON_BIN" examples/contextflow/src/summarize/cacheblend_repair_internal_exclusive.py \
  "$PROFILE_OUT" \
  --summary-output "$EXCLUSIVE_SUMMARY_OUT" \
  --events-output "$EXCLUSIVE_EVENTS_OUT"

"$PYTHON_BIN" examples/contextflow/src/summarize/cacheblend_repair_internal_stats.py \
  "$EXCLUSIVE_SUMMARY_OUT" \
  --output "$STATS_OUT"

echo
echo "Done."
echo "Run-level JSONL:              $OUT"
echo "Fine-event JSONL:             $PROFILE_OUT"
echo "Repair exclusive events JSONL:$EXCLUSIVE_EVENTS_OUT"
echo "Repair exclusive summary CSV: $EXCLUSIVE_SUMMARY_OUT"
echo "Repair stats CSV:             $STATS_OUT"
