# ContextFlow Experiments

This directory contains local ContextFlow experiment runners built on top of
LMCache and vLLM. The scripts are research utilities for latency, profiling,
diagnostics, and LongBench QA accuracy; generated outputs are written to
`examples/contextflow/results/` and are ignored by git.

## Environment

The current experiment target is Qwen2.5 on an A100-class GPU with vLLM and
LMCache CacheBlend enabled. The optional bootstrap script preserves the
previously validated CUDA 13.0 environment:

```bash
bash examples/contextflow/scripts/setup_cacheblend_env.sh
```

For normal development in this repository, prefer using the active editable
checkout and install the project dependencies from the repository root.

## Main Runners

- `scripts/run_clean_latency.sh`: clean end-to-end latency comparison across
  `full_recompute`, prefix chunk reuse, naive segment reuse, and CacheBlend.
- `scripts/run_fine_breakdown.sh`: single-pass fine profiling sweep for
  CacheBlend.
- `scripts/run_fine_breakdown_repeats.sh`: repeat-run fine profiling for
  CacheBlend with exclusive timing buckets and tail-latency statistics.
- `scripts/run_cacheblend_repair_internal.sh`: focused breakdown of CacheBlend repair
  internals.
- `scripts/run_drift_analysis.sh`: layer-wise KV drift diagnostic.
- `scripts/run_kv_layout.sh`: KV physical/logical layout diagnostic.
- `scripts/run_accuracy.sh`: LongBench QA accuracy pipeline with a reusable
  chunk-selection file.
- `scripts/run_cacheblend_accuracy.sh`: CacheBlend-shaped LongBench QA
  pipeline with retrieval preprocessing and judge-input export.

## Worker And Summary Files

The shell scripts invoke single-condition Python workers so each GPU-heavy
condition starts in a fresh process. Summary scripts consume JSONL outputs and
write CSVs.

The Python code lives under a shallow `src` layout:

- `src/`: shared latency/profiling workers and common helpers.
- `src/accuracy/`: LongBench selection, retrieval, generation, and judge-export
  utilities.
- `src/diagnostics/`: drift and KV-layout diagnostic workers.
- `src/summarize/`: post-processing tools for generated JSONL experiment
  outputs.

The scripts set the required `LMCACHE_CONTEXTFLOW_*` gates for each diagnostic.
By default, the added LMCache instrumentation remains passive unless a
ContextFlow runner explicitly enables profiling, fine breakdown, drift, or KV
layout recording.

Synthetic latency and profiling:

- `src/common.py`
- `src/synthetic_runner.py`
- `src/clean_latency.py`
- `src/summarize/clean_latency.py`
- `src/summarize/fine_breakdown.py`
- `src/summarize/fine_exclusive.py`
- `src/summarize/fine_repeat_stats.py`
- `src/summarize/cacheblend_repair_internal_exclusive.py`
- `src/summarize/cacheblend_repair_internal_stats.py`

Diagnostics:

- `src/diagnostics/drift_runner.py`
- `src/summarize/drift_analysis.py`
- `src/diagnostics/kv_layout_runner.py`
- `src/summarize/kv_layout.py`
- `src/diagnostics/extract_kv_physical_layout.py`

Accuracy:

- `src/accuracy/selection.py`
- `src/accuracy/runner.py`
- `src/summarize/accuracy.py`
- `src/accuracy/cacheblend_retrieval.py`
- `src/accuracy/cacheblend_runner.py`
- `src/summarize/cacheblend_accuracy.py`
- `src/accuracy/judge_export.py`

## Data

The accuracy runners expect local LongBench parquet files under `data/`, for
example:

```bash
DATASETS="musique:data/longbench/musique/test.parquet 2wikimqa:data/longbench/2wikimqa/test.parquet"
```

Dataset recreation details are tracked in the root `ContextFlow.md`.

## Cleanup Notes

Removed historical files include early smoke sweeps, old QA runners, stage-only
profiling, one-off recall diagnostics, frozen environment snapshots, and
vendored patch copies. Their roles are covered by the main runners above or by
the current LMCache source changes in this branch.
