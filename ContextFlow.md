# ContextFlow on LMCache

This branch contains local ContextFlow research changes on top of LMCache. The
goal is to keep the LMCache-facing code reviewable while preserving the
experiment, profiling, and dataset setup needed to reproduce the current runs.

## Data Policy

Experiment datasets live under `data/` and are intentionally not committed to
git. Keep scripts pointed at stable local paths, and document every required
dataset here so the environment can be rebuilt after cloning or moving the
repo.

Current required LongBench subsets:

- `musique`: `data/longbench/musique/test.parquet`
- `2wikimqa`: `data/longbench/2wikimqa/test.parquet`

The ContextFlow accuracy scripts use these defaults:

```bash
DATASETS="musique:data/longbench/musique/test.parquet 2wikimqa:data/longbench/2wikimqa/test.parquet"
```

## Recreate Local Datasets

From the repository root, install the Hugging Face Hub CLI if needed:

```bash
python -m pip install "huggingface_hub[cli]"
```

Download the two LongBench parquet shards:

```bash
mkdir -p data/longbench
huggingface-cli download THUDM/LongBench \
  --repo-type dataset \
  --revision 8cbd111d8ac61f5cae05535389385177fd63a63a \
  --include "musique/test-00000-of-00001.parquet" \
  --include "2wikimqa/test-00000-of-00001.parquet" \
  --local-dir data/longbench
```

Create the canonical filenames expected by the current scripts:

```bash
cp data/longbench/musique/test-00000-of-00001.parquet \
  data/longbench/musique/test.parquet
cp data/longbench/2wikimqa/test-00000-of-00001.parquet \
  data/longbench/2wikimqa/test.parquet
```

After this, the default ContextFlow runners can read the datasets without
additional path overrides.

The local Hugging Face cache under `data/longbench/.cache/` is disposable. Keep
the canonical `test.parquet` files above, or regenerate them with the commands
in this section.

## Notes To Keep Updated

- Record any new dataset name and canonical path before adding it to scripts.
- Keep generated outputs in `examples/contextflow/results/`; this path is
  ignored by git.
- Keep any core LMCache instrumentation notes here until they are moved into a
  more specific design or experiment document.

## LMCache ContextFlow Gates

The LMCache package changes in this branch are intended to be passive by
default. The original LMCache behavior should remain unchanged unless one of
the following ContextFlow experiment gates is explicitly enabled by a runner:

- `LMCACHE_CONTEXTFLOW_PROFILE=1`: enables coarse JSONL stage profiling.
- `LMCACHE_CONTEXTFLOW_FINE_BREAKDOWN=1` or `CONTEXTFLOW_FINE_BREAKDOWN=1`:
  enables fine-grained profiling hooks on top of coarse profiling.
- `LMCACHE_CONTEXTFLOW_DRIFT_ANALYSIS=1`: enables layer-wise KV drift event
  recording on top of coarse profiling.
- `LMCACHE_CONTEXTFLOW_KV_LAYOUT=1`: enables KV layout accounting events on top
  of coarse profiling.
- `LMCACHE_CONTEXTFLOW_KV_LAYOUT_SCATTER=1`: adds raw physical/logical layout
  scatter metadata to KV layout events.
- `lmcache.contextflow_naive_segment_reuse=true`: vLLM connector extra config
  used only by ContextFlow experiments to compare segment reuse without
  CacheBlend repair.

The vLLM attention import fallback in `lmcache/v1/compute/attention/flash_attn.py`
is a version-compatibility shim for the local experiment environment, not an
experiment mode.

## ContextFlow Example Layout

The maintained experiment entrypoints live under `examples/contextflow/scripts/`.

- Latency/profiling: `run_clean_latency.sh`, `run_fine_breakdown.sh`,
  `run_fine_breakdown_repeats.sh`, and `run_cacheblend_repair_internal.sh`.
- Diagnostics: `run_drift_analysis.sh` and `run_kv_layout.sh`.
- Accuracy: `run_accuracy.sh` and `run_cacheblend_accuracy.sh`.

The Python implementation uses a shallow `examples/contextflow/src/` layout.
Shared latency/profiling workers live directly in `src/`, with accuracy
workflows in `src/accuracy/`, diagnostic workers in `src/diagnostics/`, and
post-processing tools in `src/summarize/`.

Generated result files are not kept in the repository. Historical smoke tests,
environment freeze files, vendored patch snapshots, and superseded QA/recall
diagnostics were removed from `examples/contextflow/` so the directory reflects
the current runnable workflows.
