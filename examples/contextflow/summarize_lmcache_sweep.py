from __future__ import annotations

import argparse
import json
from pathlib import Path


def method_label(r: dict) -> str:
    if r["method"] == "full_recompute":
        return "full_recompute"
    ratio = r.get("lmcache_recompute_ratio")
    if ratio == 0.0:
        return "lmcache_reuse_proxy"
    return f"lmcache_selective_r{ratio}"


def metric_label(r: dict) -> str:
    if r["max_new_tokens"] == 1:
        return "ttft_proxy_1tok"
    return f"e2e_{r['max_new_tokens']}tok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    args = parser.parse_args()

    path = Path(args.jsonl)
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))

    print(
        "chunk_count,method,metric,success,prompt_tokens,"
        "generation_sec,warmup_sec,first_sec,second_sec,"
        "output_tokens,gpu_start_mb,gpu_peak_mb,gpu_end_mb,gpu_peak_delta_mb,error"
    )

    rows.sort(
        key=lambda r: (
            r["chunk_count"],
            metric_label(r),
            method_label(r),
        )
    )

    for r in rows:
        print(
            f"{r.get('chunk_count')},"
            f"{method_label(r)},"
            f"{metric_label(r)},"
            f"{r.get('success')},"
            f"{r.get('prompt_tokens')},"
            f"{r.get('generation_sec')},"
            f"{r.get('warmup_sec')},"
            f"{r.get('first_sec')},"
            f"{r.get('second_sec')},"
            f"{r.get('output_tokens')},"
            f"{r.get('gpu_start_mb')},"
            f"{r.get('gpu_peak_mb')},"
            f"{r.get('gpu_end_mb')},"
            f"{r.get('gpu_peak_delta_mb')},"
            f"{r.get('error')}"
        )


if __name__ == "__main__":
    main()
