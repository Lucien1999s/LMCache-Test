#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATASETS="${DATASETS:-musique:data/longbench/musique/test.parquet 2wikimqa:data/longbench/2wikimqa/test.parquet}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
CHUNK_COUNTS="${CHUNK_COUNTS:-6}"
SELECTION_MODES="${SELECTION_MODES:-oracle_answer_topk}"
METHODS="${METHODS:-full_recompute lmcache_prefix_chunked_reuse lmcache_naive_segment_reuse lmcache_blend}"
LIMIT="${LIMIT:-50}"
START_INDEX="${START_INDEX:-0}"
EXAMPLES_PER_PROCESS="${EXAMPLES_PER_PROCESS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32648}"
GPU_UTIL="${GPU_UTIL:-0.7}"
RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-0.15}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
MAX_LOCAL_CPU_SIZE_GB="${MAX_LOCAL_CPU_SIZE_GB:-5}"
SLEEP_BETWEEN_REQUESTS="${SLEEP_BETWEEN_REQUESTS:-0.2}"
ENABLE_SPARSE="${ENABLE_SPARSE:-0}"
WARMUP="${WARMUP:-1}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-mpnet-base-v2}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
EMBEDDING_LOCAL_FILES_ONLY="${EMBEDDING_LOCAL_FILES_ONLY:-0}"

RESULT_DIR="${RESULT_DIR:-examples/contextflow/results}"
SELECTION_OUT="${SELECTION_OUT:-$RESULT_DIR/contextflow_accuracy_selection.jsonl}"
OUT="${OUT:-$RESULT_DIR/contextflow_accuracy_runs.jsonl}"
SUMMARY_OUT="${SUMMARY_OUT:-$RESULT_DIR/contextflow_accuracy_summary.csv}"
RESET_OUT="${RESET_OUT:-1}"
RESET_SELECTION="${RESET_SELECTION:-1}"

mkdir -p "$RESULT_DIR" /tmp/cf

if [[ "$RESET_OUT" == "1" ]]; then
  rm -f "$OUT" "$SUMMARY_OUT"
fi

if [[ "$RESET_SELECTION" == "1" ]]; then
  rm -f "$SELECTION_OUT" "${SELECTION_OUT%.jsonl}.summary.csv"
fi

echo "Model: $MODEL"
echo "Datasets: $DATASETS"
echo "Chunk size: $CHUNK_SIZE"
echo "Chunk counts: $CHUNK_COUNTS"
echo "Selection modes: $SELECTION_MODES"
echo "Methods: $METHODS"
echo "Limit/start: $LIMIT / $START_INDEX"
echo "Examples per process: $EXAMPLES_PER_PROCESS"
echo "Selection JSONL: $SELECTION_OUT"
echo "Run JSONL: $OUT"
echo "Summary CSV: $SUMMARY_OUT"

selection_args=()
if [[ "$EMBEDDING_LOCAL_FILES_ONLY" == "1" ]]; then
  selection_args+=(--embedding-local-files-only)
fi

if [[ "$RESET_SELECTION" == "1" || ! -f "$SELECTION_OUT" ]]; then
  TMPDIR=/tmp/cf TMP=/tmp/cf TEMP=/tmp/cf PYTHONHASHSEED="${PYTHONHASHSEED:-0}" \
  PYTHONPATH="$PWD/examples/contextflow/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" examples/contextflow/src/accuracy/selection.py \
    --model "$MODEL" \
    --chunk-size-tokens "$CHUNK_SIZE" \
    --chunk-counts $CHUNK_COUNTS \
    --selection-modes $SELECTION_MODES \
    --limit "$LIMIT" \
    --start-index "$START_INDEX" \
    --datasets $DATASETS \
    --embedding-model "$EMBEDDING_MODEL" \
    --embedding-device "$EMBEDDING_DEVICE" \
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
    "${selection_args[@]}" \
    --output "$SELECTION_OUT"
else
  echo "Reusing existing selection file: $SELECTION_OUT"
fi

run_one () {
  local method="$1"
  local dataset_spec="$2"
  local chunk_count="$3"
  local selection_mode="$4"
  local process_start_index="$5"
  local process_limit="$6"
  local dataset_name="${dataset_spec%%:*}"
  local dataset_path="${dataset_spec#*:}"

  echo
  echo "===== method=$method dataset=$dataset_name chunks=$chunk_count selection=$selection_mode start=$process_start_index limit=$process_limit ====="

  rm -rf /tmp/cf/*
  mkdir -p /tmp/cf

  local sparse_args=()
  if [[ "$ENABLE_SPARSE" == "1" ]]; then
    sparse_args+=(--enable-sparse)
  fi

  local warmup_args=()
  if [[ "$WARMUP" == "1" ]]; then
    warmup_args+=(--warmup)
  fi

  TMPDIR=/tmp/cf TMP=/tmp/cf TEMP=/tmp/cf PYTHONHASHSEED="${PYTHONHASHSEED:-0}" \
  PYTHONPATH="$PWD/examples/contextflow/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" examples/contextflow/src/accuracy/runner.py \
    --method "$method" \
    --dataset-name "$dataset_name" \
    --dataset-path "$dataset_path" \
    --selection-file "$SELECTION_OUT" \
    --selection-mode "$selection_mode" \
    --model "$MODEL" \
    --chunk-size-tokens "$CHUNK_SIZE" \
    --chunk-count "$chunk_count" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --limit "$process_limit" \
    --start-index "$process_start_index" \
    --recompute-ratio "$RECOMPUTE_RATIO" \
    --lmcache-chunk-size "$LMCACHE_CHUNK_SIZE" \
    --max-local-cpu-size-gb "$MAX_LOCAL_CPU_SIZE_GB" \
    --sleep-between-requests "$SLEEP_BETWEEN_REQUESTS" \
    "${sparse_args[@]}" \
    "${warmup_args[@]}" \
    --output "$OUT"
}

for method in $METHODS; do
  for dataset_spec in $DATASETS; do
    for chunk_count in $CHUNK_COUNTS; do
      for selection_mode in $SELECTION_MODES; do
        if [[ "$EXAMPLES_PER_PROCESS" -le 0 ]]; then
          run_one \
            "$method" \
            "$dataset_spec" \
            "$chunk_count" \
            "$selection_mode" \
            "$START_INDEX" \
            "$LIMIT"
        else
          end_index=$((START_INDEX + LIMIT))
          for ((batch_start = START_INDEX; batch_start < end_index; batch_start += EXAMPLES_PER_PROCESS)); do
            batch_limit="$EXAMPLES_PER_PROCESS"
            if ((batch_start + batch_limit > end_index)); then
              batch_limit=$((end_index - batch_start))
            fi
            run_one \
              "$method" \
              "$dataset_spec" \
              "$chunk_count" \
              "$selection_mode" \
              "$batch_start" \
              "$batch_limit"
          done
        fi
      done
    done
  done
done

"$PYTHON_BIN" examples/contextflow/src/summarize/accuracy.py \
  "$OUT" \
  --output "$SUMMARY_OUT"

echo
echo "Done."
echo "Selection JSONL: $SELECTION_OUT"
echo "Selection CSV:   ${SELECTION_OUT%.jsonl}.summary.csv"
echo "Run JSONL:       $OUT"
echo "Summary CSV:     $SUMMARY_OUT"
