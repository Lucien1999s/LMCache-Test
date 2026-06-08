# SPDX-License-Identifier: Apache-2.0
"""Select LongBench chunks for ContextFlow accuracy experiments.

This script creates a fixed selection file that downstream QA runs consume.
Keeping chunk selection separate from generation makes quality comparisons
between KV reuse methods fair and reproducible.
"""

from __future__ import annotations

# Standard
import argparse
import csv
import json
import re
import string
from pathlib import Path
from typing import Any

# Third Party
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "from",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "what",
    "who",
    "when",
    "where",
    "which",
    "why",
    "how",
    "did",
    "does",
    "do",
    "this",
    "that",
    "these",
    "those",
    "as",
    "it",
    "its",
    "into",
}


def normalize_text(text: str) -> str:
    """Normalize answer text for answer-containment diagnostics."""
    text = str(text).lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_in_text(text: str, answers: list[str]) -> bool:
    """Return whether any gold answer string appears in ``text``."""
    normalized_text = normalize_text(text)
    for answer in answers:
        normalized_answer = normalize_text(answer)
        if normalized_answer and normalized_answer in normalized_text:
            return True
    return False


def to_answer_list(value: Any) -> list[str]:
    """Convert a parquet answer cell into a plain list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted]
        return [str(converted)]
    return [str(value)]


def encode_no_special(tokenizer: Any, text: str) -> list[int]:
    """Encode text without adding model-specific special tokens."""
    return tokenizer.encode(text, add_special_tokens=False)


def chunk_context(
    tokenizer: Any,
    context: str,
    chunk_size_tokens: int,
) -> list[dict[str, Any]]:
    """Split a context string into fixed-size token chunks."""
    token_ids = encode_no_special(tokenizer, context)
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(token_ids), chunk_size_tokens):
        chunk_ids = token_ids[start : start + chunk_size_tokens]
        if not chunk_ids:
            continue
        text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(
            {
                "index": len(chunks),
                "start_token": start,
                "end_token": start + len(chunk_ids),
                "num_tokens": len(chunk_ids),
                "ids": chunk_ids,
                "text": text,
            }
        )
    return chunks


def _word_set(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", str(text).lower())
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def lexical_rank_chunks(
    chunks: list[dict[str, Any]],
    question: str,
) -> list[dict[str, Any]]:
    """Rank chunks by simple question-word overlap."""
    question_words = _word_set(question)
    scored = []
    for chunk in chunks:
        chunk_words = _word_set(chunk["text"])
        score = len(question_words & chunk_words)
        scored.append((score, -int(chunk["index"]), chunk))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    ranked: list[dict[str, Any]] = []
    for rank, (score, _, chunk) in enumerate(scored, start=1):
        item = dict(chunk)
        item["selection_score"] = float(score)
        item["selection_rank"] = rank
        ranked.append(item)
    return ranked


class EmbeddingRanker:
    """Small transformers-only mean-pooling embedding ranker."""

    def __init__(
        self,
        model_name: str,
        device: str,
        local_files_only: bool,
        batch_size: int,
    ) -> None:
        """Load an embedding model.

        Args:
            model_name: Hugging Face model name or local path.
            device: Torch device string for encoding.
            local_files_only: Whether to forbid model downloads.
            batch_size: Number of texts encoded per batch.
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts into normalized mean-pooled embeddings."""
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            output = self.model(**encoded)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            vectors.append(F.normalize(summed / counts, p=2, dim=1).cpu())
        return torch.cat(vectors, dim=0)

    def rank(
        self,
        chunks: list[dict[str, Any]],
        question: str,
    ) -> list[dict[str, Any]]:
        """Rank chunks by cosine similarity to the question."""
        if not chunks:
            return []
        query_vec = self.encode([question])
        chunk_vecs = self.encode([str(chunk["text"]) for chunk in chunks])
        scores = torch.mv(chunk_vecs, query_vec[0])
        order = torch.argsort(scores, descending=True).tolist()

        ranked: list[dict[str, Any]] = []
        for rank, chunk_idx in enumerate(order, start=1):
            item = dict(chunks[chunk_idx])
            item["selection_score"] = float(scores[chunk_idx].item())
            item["selection_rank"] = rank
            ranked.append(item)
        return ranked


def select_chunks(
    *,
    chunks: list[dict[str, Any]],
    question: str,
    answers: list[str],
    chunk_count: int,
    selection_mode: str,
    embedding_ranker: EmbeddingRanker | None,
) -> list[dict[str, Any]]:
    """Select chunks according to the requested mode."""
    lexical_ranked = lexical_rank_chunks(chunks, question)

    if selection_mode == "lexical_topk":
        selected = lexical_ranked[:chunk_count]
    elif selection_mode == "embedding_topk":
        if embedding_ranker is None:
            raise ValueError("embedding_topk requires an embedding ranker")
        selected = embedding_ranker.rank(chunks, question)[:chunk_count]
    elif selection_mode == "oracle_answer_topk":
        answer_chunks = [
            dict(chunk)
            for chunk in lexical_ranked
            if answer_in_text(chunk["text"], answers)
        ]
        selected_by_index: dict[int, dict[str, Any]] = {}
        for chunk in answer_chunks:
            selected_by_index[int(chunk["index"])] = chunk
            if len(selected_by_index) >= chunk_count:
                break
        for chunk in lexical_ranked:
            if len(selected_by_index) >= chunk_count:
                break
            selected_by_index.setdefault(int(chunk["index"]), chunk)
            if len(selected_by_index) >= chunk_count:
                break
        selected = list(selected_by_index.values())
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    selected.sort(key=lambda chunk: int(chunk["index"]))
    return selected


def _public_chunk_fields(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_chunks = []
    for chunk in chunks:
        public_chunks.append(
            {
                "index": int(chunk["index"]),
                "start_token": int(chunk["start_token"]),
                "end_token": int(chunk["end_token"]),
                "num_tokens": int(chunk["num_tokens"]),
                "selection_rank": chunk.get("selection_rank"),
                "selection_score": chunk.get("selection_score"),
            }
        )
    return public_chunks


def _write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    summary_path = output_path.with_suffix(".summary.csv")
    fieldnames = [
        "dataset",
        "selection_mode",
        "chunk_count",
        "num_examples",
        "avg_num_context_chunks",
        "avg_num_answer_chunks",
        "full_context_answer_recall",
        "selected_answer_recall",
        "avg_selected_chunks",
    ]

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["selection_mode"]),
            int(row["chunk_count"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (dataset, mode, chunk_count), group in sorted(grouped.items()):
        count = len(group)
        summary_rows.append(
            {
                "dataset": dataset,
                "selection_mode": mode,
                "chunk_count": chunk_count,
                "num_examples": count,
                "avg_num_context_chunks": sum(
                    row["num_context_chunks"] for row in group
                )
                / count,
                "avg_num_answer_chunks": sum(
                    row["num_answer_chunks"] for row in group
                )
                / count,
                "full_context_answer_recall": sum(
                    bool(row["full_context_contains_answer"]) for row in group
                )
                / count,
                "selected_answer_recall": sum(
                    bool(row["selected_contains_answer"]) for row in group
                )
                / count,
                "avg_selected_chunks": sum(
                    len(row["selected_chunk_indices"]) for row in group
                )
                / count,
            }
        )

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote selection summary CSV: {summary_path}")


def run(args: argparse.Namespace) -> None:
    """Create fixed chunk selections for all requested datasets/settings."""
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    embedding_ranker = None
    if "embedding_topk" in args.selection_modes:
        embedding_ranker = EmbeddingRanker(
            model_name=args.embedding_model,
            device=args.embedding_device,
            local_files_only=args.embedding_local_files_only,
            batch_size=args.embedding_batch_size,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    with output_path.open("w") as f:
        for dataset_name, dataset_path in args.datasets:
            df = pd.read_parquet(dataset_path)
            if args.limit is not None:
                df = df.iloc[args.start_index : args.start_index + args.limit]
            else:
                df = df.iloc[args.start_index :]

            for local_index, (_, row) in enumerate(df.iterrows()):
                example_index = int(args.start_index + local_index)
                example = row.to_dict()
                question = str(example.get("input", ""))
                context = str(example.get("context", ""))
                answers = to_answer_list(example.get("answers"))
                chunks = chunk_context(tokenizer, context, args.chunk_size_tokens)
                answer_chunk_indices = [
                    int(chunk["index"])
                    for chunk in chunks
                    if answer_in_text(chunk["text"], answers)
                ]

                for chunk_count in args.chunk_counts:
                    for selection_mode in args.selection_modes:
                        selected = select_chunks(
                            chunks=chunks,
                            question=question,
                            answers=answers,
                            chunk_count=chunk_count,
                            selection_mode=selection_mode,
                            embedding_ranker=embedding_ranker,
                        )
                        selected_indices = [int(chunk["index"]) for chunk in selected]

                        result = {
                            "dataset": dataset_name,
                            "dataset_path": dataset_path,
                            "example_index": example_index,
                            "example_id": str(example.get("_id", example_index)),
                            "question": question,
                            "answers": answers,
                            "chunk_size_tokens": args.chunk_size_tokens,
                            "chunk_count": chunk_count,
                            "selection_mode": selection_mode,
                            "num_context_chunks": len(chunks),
                            "num_answer_chunks": len(answer_chunk_indices),
                            "answer_chunk_indices": answer_chunk_indices[:50],
                            "selected_chunk_indices": selected_indices,
                            "selected_chunks": _public_chunk_fields(selected),
                            "full_context_contains_answer": answer_in_text(
                                context,
                                answers,
                            ),
                            "selected_contains_answer": any(
                                index in answer_chunk_indices
                                for index in selected_indices
                            ),
                            "embedding_model": (
                                args.embedding_model
                                if selection_mode == "embedding_topk"
                                else None
                            ),
                        }
                        rows.append(result)
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    _write_summary(rows, output_path)
    print(f"Wrote selection JSONL: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--chunk-size-tokens", type=int, default=512)
    parser.add_argument("--chunk-counts", type=int, nargs="+", default=[6])
    parser.add_argument(
        "--selection-modes",
        nargs="+",
        choices=["oracle_answer_topk", "lexical_topk", "embedding_topk"],
        default=["oracle_answer_topk"],
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "musique:data/longbench/musique/test.parquet",
            "2wikimqa:data/longbench/2wikimqa/test.parquet",
        ],
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-local-files-only", action="store_true")

    args = parser.parse_args()
    parsed_datasets = []
    for item in args.datasets:
        name, path = item.split(":", 1)
        parsed_datasets.append((name, path))
    args.datasets = parsed_datasets
    return args


if __name__ == "__main__":
    run(parse_args())
