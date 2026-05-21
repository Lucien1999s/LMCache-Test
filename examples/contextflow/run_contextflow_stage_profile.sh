#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
CHUNK_COUNTS="${CHUNK_COUNTS:-2 4 8 16}"
MAX_NEW_TOKENS_LIST="${MAX_NEW_TOKENS_LIST:-1 16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32648}"
GPU_UTIL="${GPU_UTIL:-0.7}"
RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-0.15}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
METHODS="${METHODS:-lmcache_blend}"
ENABLE_SPARSE="${ENABLE_SPARSE:-0}"
CUDA_SYNC="${CUDA_SYNC:-1}"
PROFILE="${PROFILE:-1}"
RESET_OUT="${RESET_OUT:-1}"

RESULT_DIR="${RESULT_DIR:-examples/contextflow/results}"
OUT="${OUT:-$RESULT_DIR/contextflow_stage_profile_runs.jsonl}"
PROFILE_OUT="${PROFILE_OUT:-$RESULT_DIR/contextflow_stage_profile_events.jsonl}"
SUMMARY_OUT="${SUMMARY_OUT:-$RESULT_DIR/contextflow_stage_profile_events.summary.csv}"

mkdir -p "$RESULT_DIR" /tmp/cf

if [[ "$RESET_OUT" == "1" ]]; then
  rm -f "$OUT" "$PROFILE_OUT" "$SUMMARY_OUT"
fi

echo "Model: $MODEL"
echo "Chunk size: $CHUNK_SIZE"
echo "Chunk counts: $CHUNK_COUNTS"
echo "Max new tokens: $MAX_NEW_TOKENS_LIST"
echo "Methods: $METHODS"
echo "Profile enabled: $PROFILE"
echo "Profile JSONL: $PROFILE_OUT"
echo "Run JSONL: $OUT"
echo "Summary CSV: $SUMMARY_OUT"

run_one () {
  local method="$1"
  local chunk_count="$2"
  local max_new_tokens="$3"
  local run_id="$method-c${chunk_count}-tok${max_new_tokens}-$(date +%s%N)"

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
    cuda_sync_args+=(--stage-profile-cuda-sync)
  fi

  local profile_args=()
  if [[ "$PROFILE" == "1" ]]; then
    profile_args+=(
      --stage-profile-output "$PROFILE_OUT"
      --stage-profile-run-id "$run_id"
    )
  fi

  TMPDIR=/tmp/cf TMP=/tmp/cf TEMP=/tmp/cf \
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python examples/contextflow/contextflow_lmcache_single_run.py \
    --method "$method" \
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

for method in $METHODS; do
  for c in $CHUNK_COUNTS; do
    for max_new_tokens in $MAX_NEW_TOKENS_LIST; do
      run_one "$method" "$c" "$max_new_tokens"
    done
  done
done

if [[ "$PROFILE" == "1" ]]; then
  python examples/contextflow/summarize_contextflow_stage_profile.py \
    "$PROFILE_OUT" \
    --output "$SUMMARY_OUT"
fi

echo
echo "Done."
echo "Run-level JSONL:   $OUT"
if [[ "$PROFILE" == "1" ]]; then
  echo "Stage-event JSONL: $PROFILE_OUT"
  echo "Stage summary CSV: $SUMMARY_OUT"
else
  echo "Stage profiling disabled; only run-level JSONL was written."
fi
