#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
CHUNK_COUNTS="${CHUNK_COUNTS:-2 4 8 16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32648}"
GPU_UTIL="${GPU_UTIL:-0.7}"
OUT="${OUT:-examples/contextflow/results/qwen2_7b_official_lmcache_sweep_512.jsonl}"
RESET_OUT="${RESET_OUT:-1}"

mkdir -p "$(dirname "$OUT")"

if [[ "$RESET_OUT" == "1" ]]; then
  rm -f "$OUT"
fi

echo "Output: $OUT"
echo "Model: $MODEL"
echo "Chunk size: $CHUNK_SIZE"
echo "Chunk counts: $CHUNK_COUNTS"
echo "Max model len: $MAX_MODEL_LEN"
echo "GPU util: $GPU_UTIL"

run_one () {
  local label="$1"
  local method="$2"
  local recompute_ratio="$3"
  local chunk_count="$4"
  local max_new_tokens="$5"

  echo
  echo "===== $label | chunks=$chunk_count | max_new_tokens=$max_new_tokens | ratio=$recompute_ratio ====="

  rm -rf /tmp/cf/*
  mkdir -p /tmp/cf

  TMPDIR=/tmp/cf TMP=/tmp/cf TEMP=/tmp/cf \
  python examples/contextflow/contextflow_lmcache_single_run.py \
    --method "$method" \
    --model "$MODEL" \
    --chunk-count "$chunk_count" \
    --chunk-size-tokens "$CHUNK_SIZE" \
    --max-new-tokens "$max_new_tokens" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --recompute-ratio "$recompute_ratio" \
    --output "$OUT"
}

for c in $CHUNK_COUNTS; do
  run_one "full_recompute_ttft_proxy" "full_recompute" "0.0" "$c" "1"
  run_one "lmcache_selective_ttft_proxy" "lmcache_blend" "0.15" "$c" "1"

  run_one "full_recompute_e2e_16tok" "full_recompute" "0.0" "$c" "16"
  run_one "lmcache_selective_e2e_16tok" "lmcache_blend" "0.15" "$c" "16"
done

echo
echo "Sweep done: $OUT"
