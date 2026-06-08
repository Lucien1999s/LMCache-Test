#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
CHUNK_COUNTS="${CHUNK_COUNTS:-2 4 8 16 24 30}"
MAX_NEW_TOKENS_LIST="${MAX_NEW_TOKENS_LIST:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32648}"
GPU_UTIL="${GPU_UTIL:-0.7}"
RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-0.15}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
METHODS="${METHODS:-full_recompute lmcache_blend}"
ENABLE_SPARSE="${ENABLE_SPARSE:-0}"
CUDA_SYNC="${CUDA_SYNC:-0}"
PROFILE_MEMORY="${PROFILE_MEMORY:-0}"
LAYOUT_SCATTER="${LAYOUT_SCATTER:-0}"
NO_WARMUP="${NO_WARMUP:-0}"
RESET_OUT="${RESET_OUT:-1}"

RESULT_DIR="${RESULT_DIR:-examples/contextflow/results}"
OUT="${OUT:-$RESULT_DIR/contextflow_kv_layout_runs.jsonl}"
PROFILE_OUT="${PROFILE_OUT:-$RESULT_DIR/contextflow_kv_layout_events.jsonl}"
SUMMARY_OUT="${SUMMARY_OUT:-$RESULT_DIR/contextflow_kv_layout_summary.csv}"
PHYSICAL_LAYOUT_OUT="${PHYSICAL_LAYOUT_OUT:-$RESULT_DIR/contextflow_kv_physical_layout.jsonl}"

mkdir -p "$RESULT_DIR" /tmp/cf

if [[ "$RESET_OUT" == "1" ]]; then
  rm -f "$OUT" "$PROFILE_OUT" "$SUMMARY_OUT" "$PHYSICAL_LAYOUT_OUT"
fi

echo "Model: $MODEL"
echo "Chunk size: $CHUNK_SIZE"
echo "Chunk counts: $CHUNK_COUNTS"
echo "Max new tokens: $MAX_NEW_TOKENS_LIST"
echo "Methods: $METHODS"
echo "KV layout events JSONL: $PROFILE_OUT"
echo "Run JSONL: $OUT"
echo "Summary CSV: $SUMMARY_OUT"
if [[ "$LAYOUT_SCATTER" == "1" ]]; then
  echo "Physical layout JSONL: $PHYSICAL_LAYOUT_OUT"
fi

run_one () {
  local method="$1"
  local chunk_count="$2"
  local max_new_tokens="$3"
  local run_id="$method-kvlayout-c${chunk_count}-tok${max_new_tokens}-$(date +%s%N)"

  echo
  echo "===== method=$method chunks=$chunk_count max_new_tokens=$max_new_tokens ====="

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

  local scatter_args=()
  if [[ "$LAYOUT_SCATTER" == "1" ]]; then
    scatter_args+=(--layout-scatter)
  fi

  local warmup_args=()
  if [[ "$NO_WARMUP" == "1" ]]; then
    warmup_args+=(--no-warmup)
  fi

  TMPDIR=/tmp/cf TMP=/tmp/cf TEMP=/tmp/cf \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  PYTHONPATH="$PWD/examples/contextflow/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python examples/contextflow/src/diagnostics/kv_layout_runner.py \
    --method "$method" \
    --model "$MODEL" \
    --chunk-count "$chunk_count" \
    --chunk-size-tokens "$CHUNK_SIZE" \
    --max-new-tokens "$max_new_tokens" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --recompute-ratio "$RECOMPUTE_RATIO" \
    --lmcache-chunk-size "$LMCACHE_CHUNK_SIZE" \
    --profile-output "$PROFILE_OUT" \
    --profile-run-id "$run_id" \
    "${cuda_sync_args[@]}" \
    "${memory_args[@]}" \
    "${scatter_args[@]}" \
    "${warmup_args[@]}" \
    "${sparse_args[@]}" \
    --output "$OUT"
}

for method in $METHODS; do
  for chunk_count in $CHUNK_COUNTS; do
    for max_new_tokens in $MAX_NEW_TOKENS_LIST; do
      run_one "$method" "$chunk_count" "$max_new_tokens"
    done
  done
done

python examples/contextflow/src/summarize/kv_layout.py \
  "$PROFILE_OUT" \
  --output "$SUMMARY_OUT"

if [[ "$LAYOUT_SCATTER" == "1" ]]; then
  python examples/contextflow/src/diagnostics/extract_kv_physical_layout.py \
    "$PROFILE_OUT" \
    --output "$PHYSICAL_LAYOUT_OUT"
fi

echo
echo "Done."
echo "Run-level JSONL:     $OUT"
echo "KV layout JSONL:     $PROFILE_OUT"
echo "KV layout summary:   $SUMMARY_OUT"
if [[ "$LAYOUT_SCATTER" == "1" ]]; then
  echo "Physical layout:     $PHYSICAL_LAYOUT_OUT"
fi
