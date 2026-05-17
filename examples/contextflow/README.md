# ContextFlow LMCache CacheBlend Diagnostic

This directory contains the ContextFlow adaptation of the official LMCache `blend_kv_v1` example.

Verified environment:
- GPU: NVIDIA A100-SXM4-40GB
- vLLM: 0.18.0+cu130
- LMCache: 0.4.6.dev6
- Torch: 2.10.0+cu130
- Model sanity check:
  - Qwen/Qwen2.5-0.5B-Instruct: success
  - Qwen/Qwen2.5-7B-Instruct: success

Required compatibility patches:
1. Register the loaded vLLM model in `gpu_worker.py` before LMCache KV transfer initialization.
2. Patch LMCache `flash_attn.py` to fallback from `vllm.attention` to `vllm.model_executor.layers.attention`.

Goal:
Run a controlled chunk-scaling diagnostic for LMCache CacheBlend:
- chunk counts: 2, 4, 8, 16
- model: Qwen2.5
- metrics: latency, TTFT-style timing, output tokens, prompt tokens, GPU memory, success/failure
