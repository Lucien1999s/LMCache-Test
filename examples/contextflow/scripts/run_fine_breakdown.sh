#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
CHUNK_COUNTS="${CHUNK_COUNTS:-2 4 8 16 24 30}"
MAX_NEW_TOKENS_LIST="${MAX_NEW_TOKENS_LIST:-1 16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32648}"
GPU_UTIL="${GPU_UTIL:-0.7}"
RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-0.15}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
REPEATS="${REPEATS:-1}"
ENABLE_SPARSE="${ENABLE_SPARSE:-0}"
CUDA_SYNC="${CUDA_SYNC:-0}"
PROFILE="${PROFILE:-1}"
PROFILE_MEMORY="${PROFILE_MEMORY:-0}"
RESET_OUT="${RESET_OUT:-1}"

RESULT_DIR="${RESULT_DIR:-examples/contextflow/results}"
OUT="${OUT:-$RESULT_DIR/contextflow_finer_breakdown_runs.jsonl}"
PROFILE_OUT="${PROFILE_OUT:-$RESULT_DIR/contextflow_finer_breakdown_events.jsonl}"
SUMMARY_OUT="${SUMMARY_OUT:-$RESULT_DIR/contextflow_finer_breakdown_summary.csv}"

mkdir -p "$RESULT_DIR" /tmp/cf

if [[ "$RESET_OUT" == "1" ]]; then
  rm -f "$OUT" "$PROFILE_OUT" "$SUMMARY_OUT"
fi

echo "Model: $MODEL"
echo "Chunk size: $CHUNK_SIZE"
echo "Chunk counts: $CHUNK_COUNTS"
echo "Max new tokens: $MAX_NEW_TOKENS_LIST"
echo "Repeats: $REPEATS"
echo "Profile enabled: $PROFILE"
echo "Per-event memory: $PROFILE_MEMORY"
echo "Fine breakdown: 1"
echo "Profile JSONL: $PROFILE_OUT"
echo "Run JSONL: $OUT"
echo "Summary CSV: $SUMMARY_OUT"

run_one () {
  local chunk_count="$1"
  local max_new_tokens="$2"
  local repeat_id="$3"
  local run_id="lmcache_blend-fine-c${chunk_count}-tok${max_new_tokens}-r${repeat_id}-$(date +%s%N)"

  echo
  echo "===== method=lmcache_blend chunks=$chunk_count max_new_tokens=$max_new_tokens repeat=$repeat_id ====="

  rm -rf /tmp/cf/*
  mkdir -p /tmp/cf

  local sparse_args=()
  if [[ "$ENABLE_SPARSE" == "1" ]]; then
    sparse_args+=(--enable-sparse)
  fi

  local cuda_sync_args=()
  if [[ "$CUDA_SYNC" == "1" ]]; then
    cuda_sync_args+=(--stage-profile-cuda-sync)
  fi
  if [[ "$PROFILE_MEMORY" != "1" ]]; then
    cuda_sync_args+=(--stage-profile-no-memory)
  fi

  local profile_args=()
  if [[ "$PROFILE" == "1" ]]; then
    profile_args+=(
      --stage-profile-output "$PROFILE_OUT"
      --stage-profile-run-id "$run_id"
    )
  fi

  TMPDIR=/tmp/cf TMP=/tmp/cf TEMP=/tmp/cf \
  CONTEXTFLOW_FINE_BREAKDOWN=1 \
  LMCACHE_CONTEXTFLOW_FINE_BREAKDOWN=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  PYTHONPATH="$PWD/examples/contextflow/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python examples/contextflow/src/synthetic_runner.py \
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
    "${cuda_sync_args[@]}" \
    "${sparse_args[@]}" \
    --output "$OUT"
}

for c in $CHUNK_COUNTS; do
  for max_new_tokens in $MAX_NEW_TOKENS_LIST; do
    for repeat_id in $(seq 1 "$REPEATS"); do
      run_one "$c" "$max_new_tokens" "$repeat_id"
    done
  done
done

if [[ "$PROFILE" == "1" ]]; then
  python examples/contextflow/src/summarize/fine_breakdown.py \
    "$PROFILE_OUT" \
    --runs-jsonl "$OUT" \
    --output "$SUMMARY_OUT"
fi

echo
echo "Done."
echo "Run-level JSONL:      $OUT"
if [[ "$PROFILE" == "1" ]]; then
  echo "Fine-event JSONL:     $PROFILE_OUT"
  echo "Fine summary CSV:     $SUMMARY_OUT"
else
  echo "Fine profiling disabled; only run-level JSONL was written."
fi
