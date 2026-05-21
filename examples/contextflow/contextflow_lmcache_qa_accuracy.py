# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

# Must be set before vLLM/LMCache imports.
os.environ["TMPDIR"] = "/tmp/cf"
os.environ["TMP"] = "/tmp/cf"
os.environ["TEMP"] = "/tmp/cf"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.pop("VLLM_ATTENTION_BACKEND", None)

import argparse
import contextlib
import json
import re
import string
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import pynvml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


LMCACHE_ENV_KEYS = [
    "LMCACHE_CHUNK_SIZE",
    "LMCACHE_ENABLE_BLENDING",
    "LMCACHE_BLEND_SPECIAL_STR",
    "LMCACHE_USE_LAYERWISE",
    "LMCACHE_BLEND_CHECK_LAYERS",
    "LMCACHE_BLEND_RECOMPUTE_RATIOS",
    "LMCACHE_LOCAL_CPU",
    "LMCACHE_MAX_LOCAL_CPU_SIZE",
    "LMCACHE_LOCAL_DISK",
    "LMCACHE_MAX_LOCAL_DISK_SIZE",
    "LMCACHE_EXTRA_CONFIG",
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "at", "is", "are", "was", "were", "be", "been", "being",
    "what", "who", "when", "where", "which", "why", "how", "did", "does",
    "do", "this", "that", "these", "those", "as", "it", "its", "into",
}


def clear_lmcache_env() -> None:
    for key in LMCACHE_ENV_KEYS:
        os.environ.pop(key, None)


def setup_common_env() -> None:
    Path("/tmp/cf").mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = "/tmp/cf"
    os.environ["TMP"] = "/tmp/cf"
    os.environ["TEMP"] = "/tmp/cf"
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf-cache"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/hf-cache/hub"))


def setup_lmcache_env(blend_special_str: str, recompute_ratio: float, lmcache_chunk_size: int) -> None:
    clear_lmcache_env()
    os.environ["LMCACHE_CHUNK_SIZE"] = str(lmcache_chunk_size)
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = str(recompute_ratio)
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


@contextlib.contextmanager
def build_plain_llm(model: str, max_model_len: int, gpu_memory_utilization: float):
    clear_lmcache_env()
    llm_args = EngineArgs(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    llm = LLM(**asdict(llm_args))
    yield llm


@contextlib.contextmanager
def build_lmcache_llm(model: str, max_model_len: int, gpu_memory_utilization: float):
    ktc = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def gpu_used_mb(gpu_index: int = 0) -> float:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = float(info.used) / (1024 * 1024)
    pynvml.nvmlShutdown()
    return used


def encode_no_special(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def normalize_text(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    counts = {}
    for t in pred_tokens:
        counts[t] = counts.get(t, 0) + 1
    overlap = 0
    for t in gold_tokens:
        if counts.get(t, 0) > 0:
            overlap += 1
            counts[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def contains_match(prediction: str, ground_truth: str) -> float:
    p = normalize_text(prediction)
    g = normalize_text(ground_truth)
    if not p or not g:
        return 0.0
    return float(g in p or p in g)


def max_metric(prediction: str, answers: list[str], metric_fn) -> float:
    if not answers:
        return 0.0
    return max(metric_fn(prediction, str(a)) for a in answers)


def to_answer_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    if hasattr(x, "tolist"):
        y = x.tolist()
        if isinstance(y, list):
            return [str(v) for v in y]
        return [str(y)]
    return [str(x)]


def word_set(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {t for t in toks if len(t) > 2 and t not in STOPWORDS}


def chunk_context(tokenizer, context: str, chunk_size_tokens: int) -> list[dict[str, Any]]:
    ids = encode_no_special(tokenizer, context)
    chunks = []
    for start in range(0, len(ids), chunk_size_tokens):
        chunk_ids = ids[start:start + chunk_size_tokens]
        if not chunk_ids:
            continue
        text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append({"index": len(chunks), "ids": chunk_ids, "text": text})
    return chunks


def select_chunks(chunks: list[dict[str, Any]], question: str, k: int) -> list[dict[str, Any]]:
    qwords = word_set(question)
    scored = []
    for ch in chunks:
        cwords = word_set(ch["text"])
        overlap = len(qwords & cwords)
        scored.append((overlap, -ch["index"], ch))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = [x[2] for x in scored[:k]]
    selected.sort(key=lambda ch: ch["index"])
    return selected


def build_prompt_ids(tokenizer, selected_chunks, question: str, blend_special_str: str, measured_order: bool) -> list[int]:
    sys_text = (
        "You are a helpful assistant. "
        "Use the provided context chunks to answer the question. "
        "Return only the final short answer span. Do not explain. Answer within 5 words.\n"
    )
    sys_ids = encode_no_special(tokenizer, sys_text)
    sep_ids = encode_no_special(tokenizer, blend_special_str)

    chunks = list(selected_chunks)
    if measured_order and len(chunks) >= 2:
        chunks = [chunks[1], chunks[0]] + chunks[2:]

    query_ids = encode_no_special(tokenizer, f"\nQuestion: {question}\nAnswer:")
    prompt = list(sys_ids)
    prompt += sep_ids
    for ch in chunks:
        prompt += ch["ids"]
        prompt += sep_ids
    prompt += query_ids
    return prompt


def generate_once(llm: LLM, prompt_ids: list[int], max_new_tokens: int) -> tuple[str, float, int]:
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
    start = time.time()
    outputs = llm.generate(prompts={"prompt_token_ids": prompt_ids}, sampling_params=params)
    sec = time.time() - start
    text = outputs[0].outputs[0].text.strip()
    output_tokens = len(outputs[0].outputs[0].token_ids)
    return text, sec, output_tokens


def run(args: argparse.Namespace) -> None:
    setup_common_env()
    df = pd.read_parquet(args.dataset_path)
    df = df.iloc[args.start_index: args.start_index + args.limit] if args.limit else df.iloc[args.start_index:]

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.method == "lmcache_blend":
        setup_lmcache_env(args.blend_special_str, args.recompute_ratio, args.lmcache_chunk_size)
        context = build_lmcache_llm(args.model, args.max_model_len, args.gpu_memory_utilization)
    else:
        clear_lmcache_env()
        context = build_plain_llm(args.model, args.max_model_len, args.gpu_memory_utilization)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with context as llm, out_path.open("a") as f:
        for local_i, (_, row) in enumerate(df.iterrows()):
            ex = row.to_dict()
            q = str(ex.get("input", ""))
            context_text = str(ex.get("context", ""))
            answers = to_answer_list(ex.get("answers"))

            result = {
                "dataset": args.dataset_name,
                "method": args.method,
                "model": args.model,
                "example_index": int(args.start_index + local_i),
                "example_id": str(ex.get("_id", args.start_index + local_i)),
                "question": q,
                "answers": answers,
                "chunk_size_tokens": args.chunk_size_tokens,
                "chunk_count": args.chunk_count,
                "max_new_tokens": args.max_new_tokens,
                "success": False,
                "error": None,
            }

            try:
                all_chunks = chunk_context(tokenizer, context_text, args.chunk_size_tokens)
                selected = select_chunks(all_chunks, q, args.chunk_count)
                if len(selected) < args.chunk_count:
                    raise RuntimeError(f"Not enough chunks: got {len(selected)}")

                answer_prompt = build_prompt_ids(tokenizer, selected, q, args.blend_special_str, measured_order=True)
                gpu_before = gpu_used_mb(args.gpu_index)

                first_sec = None
                second_sec = None
                if args.method == "lmcache_blend":
                    first_prompt = build_prompt_ids(tokenizer, selected, "Read the context and answer OK.", args.blend_special_str, measured_order=False)
                    second_prompt = build_prompt_ids(tokenizer, selected, "Read the context and answer OK.", args.blend_special_str, measured_order=True)
                    _, first_sec, _ = generate_once(llm, first_prompt, 1)
                    time.sleep(args.sleep_between_requests)
                    _, second_sec, _ = generate_once(llm, second_prompt, 1)
                    time.sleep(args.sleep_between_requests)

                pred, generation_sec, output_tokens = generate_once(llm, answer_prompt, args.max_new_tokens)
                gpu_after = gpu_used_mb(args.gpu_index)

                result.update({
                    "success": True,
                    "num_context_chunks": len(all_chunks),
                    "selected_chunk_indices": [int(ch["index"]) for ch in selected],
                    "prompt_tokens": len(answer_prompt),
                    "output_text": pred,
                    "output_tokens": output_tokens,
                    "first_sec": first_sec,
                    "second_sec": second_sec,
                    "generation_sec": generation_sec,
                    "gpu_before_mb": gpu_before,
                    "gpu_after_mb": gpu_after,
                    "exact_match": max_metric(pred, answers, exact_match),
                    "f1": max_metric(pred, answers, f1_score),
                    "contains_match": max_metric(pred, answers, contains_match),
                })
            except Exception as e:
                result["error"] = repr(e)

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps(result, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["full_recompute", "lmcache_blend"], required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--chunk-size-tokens", type=int, default=512)
    p.add_argument("--chunk-count", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=32648)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--blend-special-str", default="# #")
    p.add_argument("--recompute-ratio", type=float, default=0.15)
    p.add_argument("--lmcache-chunk-size", type=int, default=256)
    p.add_argument("--sleep-between-requests", type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
