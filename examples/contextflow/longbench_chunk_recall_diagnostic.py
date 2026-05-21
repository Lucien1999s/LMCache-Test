from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "at", "is", "are", "was", "were", "be", "been", "being",
    "what", "who", "when", "where", "which", "why", "how", "did", "does",
    "do", "this", "that", "these", "those", "as", "it", "its", "into",
}


def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answer_in_text(text: str, answers: list[str]) -> bool:
    nt = normalize_text(text)
    for ans in answers:
        na = normalize_text(ans)
        if na and na in nt:
            return True
    return False


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
    toks = re.findall(r"[A-Za-z0-9]+", str(text).lower())
    return {t for t in toks if len(t) > 2 and t not in STOPWORDS}


def chunk_context(tokenizer, context: str, chunk_size_tokens: int) -> list[dict[str, Any]]:
    ids = tokenizer.encode(context, add_special_tokens=False)
    chunks = []
    for start in range(0, len(ids), chunk_size_tokens):
        chunk_ids = ids[start:start + chunk_size_tokens]
        if not chunk_ids:
            continue
        text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append({
            "index": len(chunks),
            "start_token": start,
            "end_token": start + len(chunk_ids),
            "num_tokens": len(chunk_ids),
            "text": text,
        })
    return chunks


def lexical_rank_chunks(chunks: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    qwords = word_set(question)
    scored = []
    for ch in chunks:
        cwords = word_set(ch["text"])
        overlap = len(qwords & cwords)
        scored.append((overlap, -ch["index"], ch))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    ranked = []
    for rank, (score, _, ch) in enumerate(scored, start=1):
        item = dict(ch)
        item["lexical_score"] = score
        item["lexical_rank"] = rank
        ranked.append(item)
    return ranked


def run(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for dataset_name, dataset_path in args.datasets:
        df = pd.read_parquet(dataset_path)
        if args.limit:
            df = df.iloc[:args.limit]

        with out_path.open("a") as f:
            for i, (_, row) in enumerate(df.iterrows()):
                ex = row.to_dict()
                q = str(ex.get("input", ""))
                context = str(ex.get("context", ""))
                answers = to_answer_list(ex.get("answers"))

                chunks = chunk_context(tokenizer, context, args.chunk_size_tokens)
                ranked = lexical_rank_chunks(chunks, q)

                answer_chunk_indices = [
                    ch["index"] for ch in chunks
                    if answer_in_text(ch["text"], answers)
                ]

                lexical_rank_of_first_answer_chunk = None
                lexical_score_of_first_answer_chunk = None
                for ch in ranked:
                    if ch["index"] in answer_chunk_indices:
                        lexical_rank_of_first_answer_chunk = ch["lexical_rank"]
                        lexical_score_of_first_answer_chunk = ch["lexical_score"]
                        break

                result = {
                    "dataset": dataset_name,
                    "example_index": i,
                    "example_id": str(ex.get("_id", i)),
                    "question": q,
                    "answers": answers,
                    "chunk_size_tokens": args.chunk_size_tokens,
                    "num_chunks": len(chunks),
                    "full_context_contains_answer": answer_in_text(context, answers),
                    "num_answer_chunks": len(answer_chunk_indices),
                    "answer_chunk_indices": answer_chunk_indices[:20],
                    "lexical_rank_of_first_answer_chunk": lexical_rank_of_first_answer_chunk,
                    "lexical_score_of_first_answer_chunk": lexical_score_of_first_answer_chunk,
                }

                for k in args.k_values:
                    first_indices = [ch["index"] for ch in chunks[:k]]
                    lexical_indices = [ch["index"] for ch in ranked[:k]]

                    result[f"first_{k}_contains_answer"] = any(
                        idx in answer_chunk_indices for idx in first_indices
                    )
                    result[f"lexical_top_{k}_contains_answer"] = any(
                        idx in answer_chunk_indices for idx in lexical_indices
                    )
                    result[f"lexical_top_{k}_indices"] = lexical_indices

                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                all_rows.append(result)

    summary = pd.DataFrame(all_rows)
    csv_path = out_path.with_suffix(".summary.csv")
    rows = []

    for dataset_name, g in summary.groupby("dataset"):
        row = {
            "dataset": dataset_name,
            "num_examples": len(g),
            "avg_num_chunks": g["num_chunks"].mean(),
            "full_context_answer_recall": g["full_context_contains_answer"].mean(),
            "avg_num_answer_chunks": g["num_answer_chunks"].mean(),
        }
        for k in args.k_values:
            row[f"first_{k}_answer_recall"] = g[f"first_{k}_contains_answer"].mean()
            row[f"lexical_top_{k}_answer_recall"] = g[f"lexical_top_{k}_contains_answer"].mean()
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(csv_path, index=False)

    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nWrote JSONL: {out_path}")
    print(f"Wrote CSV:   {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--chunk-size-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--k-values", type=int, nargs="+", default=[6, 8, 16, 32])
    p.add_argument("--output", required=True)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "musique:data/longbench/musique/test.parquet",
            "2wikimqa:data/longbench/2wikimqa/test.parquet",
        ],
    )
    args = p.parse_args()

    parsed = []
    for item in args.datasets:
        name, path = item.split(":", 1)
        parsed.append((name, path))
    args.datasets = parsed
    return args


if __name__ == "__main__":
    run(parse_args())
