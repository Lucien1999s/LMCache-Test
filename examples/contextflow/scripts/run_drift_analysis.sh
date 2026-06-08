#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
CHUNK_COUNTS="${CHUNK_COUNTS:-2 4 8 16}"
MAX_NEW_TOKENS_LIST="${MAX_NEW_TOKENS_LIST:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32648}"
GPU_UTIL="${GPU_UTIL:-0.7}"
RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-0.15}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
BLEND_CHECK_LAYERS="${BLEND_CHECK_LAYERS:-1}"
ENABLE_SPARSE="${ENABLE_SPARSE:-0}"
CUDA_SYNC="${CUDA_SYNC:-0}"
PROFILE_MEMORY="${PROFILE_MEMORY:-0}"
NO_WARMUP="${NO_WARMUP:-0}"
RESET_OUT="${RESET_OUT:-1}"

RESULT_DIR="${RESULT_DIR:-examples/contextflow/results}"
OUT="${OUT:-$RESULT_DIR/contextflow_drift_analysis_runs.jsonl}"
PROFILE_OUT="${PROFILE_OUT:-$RESULT_DIR/contextflow_drift_analysis_events.jsonl}"
SUMMARY_OUT="${SUMMARY_OUT:-$RESULT_DIR/contextflow_drift_analysis_summary.csv}"

mkdir -p "$RESULT_DIR" /tmp/cf

if [[ "$RESET_OUT" == "1" ]]; then
  rm -f "$OUT" "$PROFILE_OUT" "$SUMMARY_OUT"
fi

echo "Model: $MODEL"
echo "Chunk size: $CHUNK_SIZE"
echo "Chunk counts: $CHUNK_COUNTS"
echo "Max new tokens: $MAX_NEW_TOKENS_LIST"
echo "Recompute ratio: $RECOMPUTE_RATIO"
echo "Blend check layers: $BLEND_CHECK_LAYERS"
echo "Run JSONL: $OUT"
echo "Drift events JSONL: $PROFILE_OUT"
echo "Drift summary CSV: $SUMMARY_OUT"

run_one () {
  local chunk_count="$1"
  local max_new_tokens="$2"
  local run_id="lmcache_blend-drift-c${chunk_count}-tok${max_new_tokens}-$(date +%s%N)"

  echo
  echo "===== drift chunks=$chunk_count max_new_tokens=$max_new_tokens ====="

  rm -rf /tmp/cf/*
  mkdir -p /tmp/cf

  local sparse_args=()
  if [[ "$ENABLE_SPARSE" == "1" ]]; then
    sparse_args+=(--enable-sparse)
  fi

  local cuda_sync_args=()
  if [[ "$CUDA_SYNC" == "1" ]]; then
    cuda_sync_args+=(--cuda-sync)
  fi

  local memory_args=()
  if [[ "$PROFILE_MEMORY" != "1" ]]; then
    memory_args+=(--no-profile-memory)
  fi

  local warmup_args=()
  if [[ "$NO_WARMUP" == "1" ]]; then
    warmup_args+=(--no-warmup)
  fi

  TMPDIR=/tmp/cf TMP=/tmp/cf TEMP=/tmp/cf \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  PYTHONPATH="$PWD/examples/contextflow/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python examples/contextflow/src/diagnostics/drift_runner.py \
    --model "$MODEL" \
    --chunk-count "$chunk_count" \
    --chunk-size-tokens "$CHUNK_SIZE" \
    --max-new-tokens "$max_new_tokens" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --recompute-ratio "$RECOMPUTE_RATIO" \
    --lmcache-chunk-size "$LMCACHE_CHUNK_SIZE" \
    --blend-check-layers "$BLEND_CHECK_LAYERS" \
    --profile-output "$PROFILE_OUT" \
    --profile-run-id "$run_id" \
    "${cuda_sync_args[@]}" \
    "${memory_args[@]}" \
    "${warmup_args[@]}" \
    "${sparse_args[@]}" \
    --output "$OUT"
}

for chunk_count in $CHUNK_COUNTS; do
  for max_new_tokens in $MAX_NEW_TOKENS_LIST; do
    run_one "$chunk_count" "$max_new_tokens"
  done
done

python examples/contextflow/src/summarize/drift_analysis.py \
  "$PROFILE_OUT" \
  --output "$SUMMARY_OUT"

echo
echo "Done."
echo "Run-level JSONL:  $OUT"
echo "Drift-event JSONL: $PROFILE_OUT"
echo "Drift summary CSV: $SUMMARY_OUT"
