# SPDX-License-Identifier: Apache-2.0
"""CacheBlend-aligned RAG retrieval preprocessing for accuracy checks."""

from __future__ import annotations

# Standard
import argparse
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


def normalize_text(text: str) -> str:
    """Normalize text for answer-string containment diagnostics."""
    text = str(text).lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_in_text(text: str, answers: list[str]) -> bool:
    """Return whether any normalized answer string appears in text."""
    normalized = normalize_text(text)
    return any(
        normalize_text(answer) and normalize_text(answer) in normalized
        for answer in answers
    )


def to_answer_list(value: Any) -> list[str]:
    """Convert a parquet answer cell into a list of strings."""
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


def parse_passages(context: str) -> list[dict[str, Any]]:
    """Parse LongBench-style Passage N blocks into source documents."""
    matches = list(re.finditer(r"(?m)^Passage\s+(\d+):\s*$", context))
    if not matches:
        return [{"source_id": 0, "source_label": "context", "text": context.strip()}]

    passages: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(context)
        source_id = int(match.group(1))
        text = context[start:end].strip()
        if text:
            passages.append(
                {
                    "source_id": source_id,
                    "source_label": f"Passage {source_id}",
                    "text": text,
                }
            )
    return passages


def encode_no_special(tokenizer: Any, text: str) -> list[int]:
    """Encode text without model-specific special tokens."""
    return tokenizer.encode(text, add_special_tokens=False)


def qa_system_prompt() -> str:
    """Return the fixed QA system prompt used for accuracy evaluation."""
    return (
        "You are a careful extractive QA assistant. Use only the provided "
        "context chunks. Return only the final short answer span. Do not "
        "explain."
    )


def qa_query_text(question: str) -> str:
    """Return the fixed QA query suffix."""
    return f"Question: {question}\nAnswer within 5 words.\nAnswer:"


def build_prompt_text(
    *,
    selected_chunks: list[dict[str, Any]],
    question: str,
    blend_special_str: str,
) -> str:
    """Build a human-readable prompt preview matching the token prompt shape."""
    pieces = [qa_system_prompt(), blend_special_str]
    for chunk in selected_chunks:
        pieces.append(
            f"[Context rank={chunk['retrieval_rank']} source={chunk['source_label']}]\n"
            f"{chunk['text']}"
        )
        pieces.append(blend_special_str)
    pieces.append(qa_query_text(question))
    return "\n".join(pieces)


def build_prompt_ids(
    *,
    tokenizer: Any,
    selected_chunks: list[dict[str, Any]],
    question: str,
    blend_special_str: str,
) -> list[int]:
    """Build prompt token IDs while preserving exact separator tokens."""
    sep_ids = encode_no_special(tokenizer, blend_special_str)
    newline_ids = encode_no_special(tokenizer, "\n")
    prompt = encode_no_special(tokenizer, qa_system_prompt())
    prompt += newline_ids + sep_ids + newline_ids
    for chunk in selected_chunks:
        header = (
            f"[Context rank={chunk['retrieval_rank']} "
            f"source={chunk['source_label']}]\n"
        )
        prompt += encode_no_special(tokenizer, header)
        prompt += encode_no_special(tokenizer, str(chunk["text"]))
        prompt += newline_ids + sep_ids + newline_ids
    prompt += encode_no_special(tokenizer, qa_query_text(question))
    return prompt


def chunk_passages(
    tokenizer: Any,
    context: str,
    chunk_size_tokens: int,
) -> list[dict[str, Any]]:
    """Split context passages into fixed token-size chunks.

    This is a dependency-free token splitter. The CacheBlend paper states that
    LangChain chunking is used but does not expose the exact splitter class; the
    output records this fallback explicitly.
    """
    chunks: list[dict[str, Any]] = []
    for passage in parse_passages(context):
        ids = encode_no_special(tokenizer, passage["text"])
        for local_chunk_id, start in enumerate(range(0, len(ids), chunk_size_tokens)):
            chunk_ids = ids[start : start + chunk_size_tokens]
            if not chunk_ids:
                continue
            text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "source_id": passage["source_id"],
                    "source_label": passage["source_label"],
                    "source_chunk_id": local_chunk_id,
                    "start_token": start,
                    "end_token": start + len(chunk_ids),
                    "token_len": len(chunk_ids),
                    "text": text,
                }
            )
    return chunks


class SentenceTransformersCompatibleRanker:
    """Mean-pooling ranker compatible with common SentenceTransformers models."""

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        local_files_only: bool,
        normalize_embeddings: bool,
    ) -> None:
        """Load tokenizer and encoder model."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
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
        """Encode text with attention-mask mean pooling."""
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
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            if self.normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.detach().cpu())
        return torch.cat(vectors, dim=0)

    def rank(self, question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rank chunks by least L2 distance to the query embedding."""
        if not chunks:
            return []
        query_vec = self.encode([question])[0]
        chunk_vecs = self.encode([chunk["text"] for chunk in chunks])
        distances = torch.linalg.vector_norm(chunk_vecs - query_vec, ord=2, dim=1)
        order = torch.argsort(distances, descending=False).tolist()
        ranked: list[dict[str, Any]] = []
        for rank, chunk_idx in enumerate(order, start=1):
            item = dict(chunks[chunk_idx])
            item["retrieval_rank"] = rank
            item["l2_distance"] = float(distances[chunk_idx].item())
            item["answer_string_hit"] = False
            ranked.append(item)
        return ranked


def build_prompt_preview(
    *,
    selected_chunks: list[dict[str, Any]],
    question: str,
    blend_special_str: str,
    max_chars: int,
) -> str:
    """Build a text prompt preview matching the generation prompt shape."""
    prompt = build_prompt_text(
        selected_chunks=selected_chunks,
        question=question,
        blend_special_str=blend_special_str,
    )
    return prompt[:max_chars]


def make_retrieval_row(
    *,
    tokenizer: Any,
    dataset_name: str,
    dataset_path: str,
    example_index: int,
    example: dict[str, Any],
    chunks: list[dict[str, Any]],
    ranked_chunks: list[dict[str, Any]],
    chunk_count: int,
    chunk_size_tokens: int,
    embedding_model: str,
    embedding_backend: str,
    normalize_embeddings: bool,
    blend_special_str: str,
    prompt_preview_chars: int,
) -> dict[str, Any]:
    """Create one retrieval-debug row."""
    question = str(example.get("input", ""))
    answers = to_answer_list(example.get("answers"))
    selected = [dict(chunk) for chunk in ranked_chunks[:chunk_count]]
    for chunk in selected:
        chunk["answer_string_hit"] = answer_in_text(chunk["text"], answers)

    return {
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "example_index": example_index,
        "example_id": str(example.get("_id", example_index)),
        "question": question,
        "answers": answers,
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_count": chunk_count,
        "retrieval_top_k": chunk_count,
        "retrieval_metric": "l2_distance",
        "retrieval_order": "ascending_l2",
        "embedding_model": embedding_model,
        "embedding_backend": embedding_backend,
        "embedding_normalized": normalize_embeddings,
        "vector_index_scope": "per_sample_context_chunks",
        "paper_exact_embedding_model": "unknown",
        "chunker": "passage_token_fixed_fallback_langchain_unavailable",
        "paper_chunker": "LangChain text chunking mechanism; exact splitter unknown",
        "support_evidence_available": False,
        "retrieved_support_hit": None,
        "num_context_chunks": len(chunks),
        "full_context_answer_hit": answer_in_text(
            str(example.get("context", "")), answers
        ),
        "retrieved_answer_hit": any(chunk["answer_string_hit"] for chunk in selected),
        "retrieved_answer_hit_count": sum(
            int(chunk["answer_string_hit"]) for chunk in selected
        ),
        "selected_chunk_ids": [chunk["chunk_id"] for chunk in selected],
        "selected_source_ids": [chunk["source_id"] for chunk in selected],
        "selected_chunks": selected,
        "retrieved_chunk_previews": [
            {
                "chunk_id": chunk["chunk_id"],
                "retrieval_rank": chunk["retrieval_rank"],
                "l2_distance": chunk["l2_distance"],
                "source_label": chunk["source_label"],
                "answer_string_hit": chunk["answer_string_hit"],
                "text_preview": chunk["text"][:360],
            }
            for chunk in selected
        ],
        "prompt_tokens": len(
            build_prompt_ids(
                tokenizer=tokenizer,
                selected_chunks=selected,
                question=question,
                blend_special_str=blend_special_str,
            )
        ),
        "prompt_preview": build_prompt_preview(
            selected_chunks=selected,
            question=question,
            blend_special_str=blend_special_str,
            max_chars=prompt_preview_chars,
        ),
    }


def parse_dataset_spec(spec: str) -> tuple[str, str]:
    """Parse ``name:path`` dataset spec."""
    if ":" not in spec:
        raise ValueError(f"Dataset spec must be name:path, got {spec}")
    name, path = spec.split(":", 1)
    return name, path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "musique:data/longbench/musique/test.parquet",
            "2wikimqa:data/longbench/2wikimqa/test.parquet",
        ],
    )
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--chunk-size-tokens", type=int, default=512)
    parser.add_argument("--chunk-counts", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-local-files-only", action="store_true")
    parser.add_argument("--normalize-embeddings", action="store_true")
    parser.add_argument("--blend-special-str", default="# #")
    parser.add_argument("--prompt-preview-chars", type=int, default=1800)
    parser.add_argument("--preview-count", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ranker = SentenceTransformersCompatibleRanker(
        model_name=args.embedding_model,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
        local_files_only=args.embedding_local_files_only,
        normalize_embeddings=args.normalize_embeddings,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_printed = 0

    with output_path.open("w") as out:
        for dataset_spec in args.datasets:
            dataset_name, dataset_path = parse_dataset_spec(dataset_spec)
            df = pd.read_parquet(dataset_path)
            end_index = min(len(df), args.start_index + args.limit)
            for example_index in range(args.start_index, end_index):
                example = df.iloc[example_index].to_dict()
                question = str(example.get("input", ""))
                answers = to_answer_list(example.get("answers"))
                chunks = chunk_passages(
                    tokenizer=tokenizer,
                    context=str(example.get("context", "")),
                    chunk_size_tokens=args.chunk_size_tokens,
                )
                ranked = ranker.rank(question, chunks)
                for chunk_count in args.chunk_counts:
                    row = make_retrieval_row(
                        tokenizer=tokenizer,
                        dataset_name=dataset_name,
                        dataset_path=dataset_path,
                        example_index=example_index,
                        example=example,
                        chunks=chunks,
                        ranked_chunks=ranked,
                        chunk_count=chunk_count,
                        chunk_size_tokens=args.chunk_size_tokens,
                        embedding_model=args.embedding_model,
                        embedding_backend="transformers_mean_pooling",
                        normalize_embeddings=args.normalize_embeddings,
                        blend_special_str=args.blend_special_str,
                        prompt_preview_chars=args.prompt_preview_chars,
                    )
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")

                    if (
                        preview_printed < args.preview_count
                        and chunk_count == args.chunk_counts[0]
                    ):
                        preview_printed += 1
                        print("\n===== Retrieval Preview =====")
                        print(f"dataset={dataset_name} index={example_index}")
                        print(f"question={question}")
                        print(f"answers={answers}")
                        for chunk in row["selected_chunks"][:3]:
                            print(
                                "rank={rank} dist={dist:.4f} answer_hit={hit} "
                                "source={source} text={text}".format(
                                    rank=chunk["retrieval_rank"],
                                    dist=chunk["l2_distance"],
                                    hit=chunk["answer_string_hit"],
                                    source=chunk["source_label"],
                                    text=chunk["text"][:240].replace("\n", " "),
                                )
                            )
                        print("prompt_preview=")
                        print(row["prompt_preview"])

    print(f"Wrote retrieval debug JSONL: {output_path}")


if __name__ == "__main__":
    main()
