#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATASETS="${DATASETS:-musique:data/longbench/musique/test.parquet 2wikimqa:data/longbench/2wikimqa/test.parquet}"
LIMIT="${LIMIT:-15}"
START_INDEX="${START_INDEX:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
RETRIEVAL_K_COUNTS="${RETRIEVAL_K_COUNTS:-2 4 8 16}"
K_COUNTS="${K_COUNTS:-2 4 8 16}"
METHODS="${METHODS:-full_recompute}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
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
NORMALIZE_EMBEDDINGS="${NORMALIZE_EMBEDDINGS:-0}"

BLEND_SPECIAL_STR="${BLEND_SPECIAL_STR:-# #}"
RESULT_DIR="${RESULT_DIR:-examples/contextflow/results}"
RETRIEVAL_OUT="${RETRIEVAL_OUT:-$RESULT_DIR/contextflow_cacheblend_accuracy_retrieval_debug.jsonl}"
PREDICTIONS_OUT="${PREDICTIONS_OUT:-$RESULT_DIR/contextflow_cacheblend_accuracy_predictions.jsonl}"
RUNS_OUT="${RUNS_OUT:-$RESULT_DIR/contextflow_cacheblend_accuracy_runs.jsonl}"
SUMMARY_OUT="${SUMMARY_OUT:-$RESULT_DIR/contextflow_cacheblend_accuracy_metrics_summary.csv}"
JUDGE_INPUTS_OUT="${JUDGE_INPUTS_OUT:-$RESULT_DIR/contextflow_cacheblend_accuracy_llm_judge_inputs.jsonl}"
RESET_OUT="${RESET_OUT:-1}"
RESET_RETRIEVAL="${RESET_RETRIEVAL:-1}"

mkdir -p "$RESULT_DIR" /tmp/cf

if [[ "$RESET_OUT" == "1" ]]; then
  rm -f "$PREDICTIONS_OUT" "$RUNS_OUT" "$SUMMARY_OUT" "$JUDGE_INPUTS_OUT"
fi

if [[ "$RESET_RETRIEVAL" == "1" ]]; then
  rm -f "$RETRIEVAL_OUT"
fi

echo "Model: $MODEL"
echo "Datasets: $DATASETS"
echo "Limit/start per dataset: $LIMIT / $START_INDEX"
echo "Chunk size: $CHUNK_SIZE"
echo "Retrieval k trend: $RETRIEVAL_K_COUNTS"
echo "Generation k counts: $K_COUNTS"
echo "Methods: $METHODS"
echo "Embedding model: $EMBEDDING_MODEL"
echo "Embedding device: $EMBEDDING_DEVICE"
echo "Retrieval JSONL: $RETRIEVAL_OUT"
echo "Predictions JSONL: $PREDICTIONS_OUT"
echo "Runs JSONL: $RUNS_OUT"
echo "Summary CSV: $SUMMARY_OUT"
echo "Judge inputs JSONL: $JUDGE_INPUTS_OUT"

retrieval_args=()
if [[ "$EMBEDDING_LOCAL_FILES_ONLY" == "1" ]]; then
  retrieval_args+=(--embedding-local-files-only)
fi
if [[ "$NORMALIZE_EMBEDDINGS" == "1" ]]; then
  retrieval_args+=(--normalize-embeddings)
fi

if [[ "$RESET_RETRIEVAL" == "1" || ! -f "$RETRIEVAL_OUT" ]]; then
  TMPDIR=/tmp/cf TMP=/tmp/cf TEMP=/tmp/cf PYTHONHASHSEED="${PYTHONHASHSEED:-0}" \
  PYTHONPATH="$PWD/examples/contextflow/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" examples/contextflow/src/accuracy/cacheblend_retrieval.py \
    --model "$MODEL" \
    --datasets $DATASETS \
    --limit "$LIMIT" \
    --start-index "$START_INDEX" \
    --chunk-size-tokens "$CHUNK_SIZE" \
    --chunk-counts $RETRIEVAL_K_COUNTS \
    --embedding-model "$EMBEDDING_MODEL" \
    --embedding-device "$EMBEDDING_DEVICE" \
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
    --blend-special-str "$BLEND_SPECIAL_STR" \
    "${retrieval_args[@]}" \
    --output "$RETRIEVAL_OUT"
else
  echo "Reusing retrieval file: $RETRIEVAL_OUT"
fi

run_one () {
  local method="$1"
  local dataset_spec="$2"
  local chunk_count="$3"
  local dataset_name="${dataset_spec%%:*}"

  echo
  echo "===== method=$method dataset=$dataset_name k=$chunk_count ====="

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
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  PYTHONPATH="$PWD/examples/contextflow/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" examples/contextflow/src/accuracy/cacheblend_runner.py \
    --method "$method" \
    --dataset-name "$dataset_name" \
    --retrieval-file "$RETRIEVAL_OUT" \
    --model "$MODEL" \
    --chunk-size-tokens "$CHUNK_SIZE" \
    --chunk-count "$chunk_count" \
    --limit "$LIMIT" \
    --start-index "$START_INDEX" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --blend-special-str "$BLEND_SPECIAL_STR" \
    --recompute-ratio "$RECOMPUTE_RATIO" \
    --lmcache-chunk-size "$LMCACHE_CHUNK_SIZE" \
    --max-local-cpu-size-gb "$MAX_LOCAL_CPU_SIZE_GB" \
    --sleep-between-requests "$SLEEP_BETWEEN_REQUESTS" \
    "${sparse_args[@]}" \
    "${warmup_args[@]}" \
    --predictions-output "$PREDICTIONS_OUT" \
    --runs-output "$RUNS_OUT" \
    --llm-judge-inputs-output "$JUDGE_INPUTS_OUT"
}

for method in $METHODS; do
  for dataset_spec in $DATASETS; do
    for chunk_count in $K_COUNTS; do
      run_one "$method" "$dataset_spec" "$chunk_count"
    done
  done
done

"$PYTHON_BIN" examples/contextflow/src/summarize/cacheblend_accuracy.py \
  "$PREDICTIONS_OUT" \
  --output "$SUMMARY_OUT"

echo
echo "Done."
echo "Retrieval debug JSONL: $RETRIEVAL_OUT"
echo "Predictions JSONL:     $PREDICTIONS_OUT"
echo "Run-level JSONL:       $RUNS_OUT"
echo "Summary CSV:           $SUMMARY_OUT"
echo "LLM judge inputs:      $JUDGE_INPUTS_OUT"
