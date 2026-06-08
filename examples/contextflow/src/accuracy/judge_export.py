# SPDX-License-Identifier: Apache-2.0
"""Export ContextFlow accuracy runs into LLM-judge-ready records."""

from __future__ import annotations

# Standard
import argparse
import csv
import json
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "judge_id",
    "dataset",
    "example_index",
    "example_id",
    "method",
    "model",
    "selection_mode",
    "chunk_count",
    "chunk_size_tokens",
    "max_new_tokens",
    "question",
    "reference_answers",
    "prediction",
    "raw_output",
    "f1",
    "exact_match",
    "contains_match",
    "num_cached_tokens",
    "generation_sec",
]


def build_judge_id(row: dict[str, Any]) -> str:
    """Build a stable identifier for one judged answer."""
    parts = [
        str(row.get("dataset", "")),
        str(row.get("selection_mode", "")),
        str(row.get("chunk_count", "")),
        str(row.get("example_id", row.get("example_index", ""))),
        str(row.get("method", "")),
    ]
    return "::".join(parts)


def build_judge_prompt(
    question: str,
    reference_answers: list[str],
    prediction: str,
) -> str:
    """Create a method-blind prompt for answer-equivalence judging."""
    references = json.dumps(reference_answers, ensure_ascii=False)
    return (
        "You are judging a short-answer QA prediction.\n"
        "Decide whether the predicted answer is semantically equivalent to "
        "any reference answer.\n\n"
        "Rules:\n"
        "- Ignore harmless casing, punctuation, and leading phrases.\n"
        "- Mark correct if the prediction clearly contains the reference answer "
        "without contradiction.\n"
        "- Mark incorrect if it gives a different entity, date, place, or yes/no "
        "answer.\n"
        "- Use partial only when the prediction is incomplete but captures a "
        "material part of the answer.\n\n"
        f"Question: {question}\n"
        f"Reference answers: {references}\n"
        f"Predicted answer: {prediction}\n\n"
        "Return JSON only with this schema:\n"
        '{"score": 0.0 | 0.5 | 1.0, '
        '"label": "incorrect" | "partial" | "correct", '
        '"reason": "brief reason"}'
    )


def normalize_answers(value: Any) -> list[str]:
    """Normalize an answers field loaded from JSONL."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def convert_row(row: dict[str, Any], include_prompt: bool) -> dict[str, Any]:
    """Convert one accuracy run row into a judge-ready record."""
    answers = normalize_answers(row.get("answers"))
    question = str(row.get("question", ""))
    prediction = str(row.get("prediction_text", row.get("output_text", "")))

    out = {
        "judge_id": build_judge_id(row),
        "dataset": row.get("dataset"),
        "example_index": row.get("example_index"),
        "example_id": row.get("example_id"),
        "method": row.get("method"),
        "model": row.get("model"),
        "selection_mode": row.get("selection_mode"),
        "chunk_count": row.get("chunk_count"),
        "chunk_size_tokens": row.get("chunk_size_tokens"),
        "max_new_tokens": row.get("max_new_tokens"),
        "question": question,
        "reference_answers": answers,
        "prediction": prediction,
        "raw_output": row.get("output_text"),
        "selected_chunk_indices": row.get("selected_chunk_indices", []),
        "selected_contains_answer": row.get("selected_contains_answer"),
        "full_context_contains_answer": row.get("full_context_contains_answer"),
        "f1": row.get("f1"),
        "exact_match": row.get("exact_match"),
        "contains_match": row.get("contains_match"),
        "num_cached_tokens": row.get("num_cached_tokens"),
        "generation_sec": row.get("generation_sec"),
        "engine_ttft_sec": row.get("engine_ttft_sec"),
        "engine_prefill_sec": row.get("engine_prefill_sec"),
        "engine_decode_sec": row.get("engine_decode_sec"),
        "engine_tpot_sec": row.get("engine_tpot_sec"),
        "wall_tpot_sec": row.get("wall_tpot_sec"),
    }
    if include_prompt:
        out["judge_prompt"] = build_judge_prompt(question, answers, prediction)
    return out


def load_rows(input_path: Path, include_failures: bool) -> list[dict[str, Any]]:
    """Load successful accuracy rows from a JSONL file."""
    rows = []
    for line in input_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if include_failures or row.get("success"):
            rows.append(row)
    return rows


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write judge records to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a compact CSV view of judge records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = {field: row.get(field) for field in CSV_FIELDS}
            csv_row["reference_answers"] = json.dumps(
                csv_row["reference_answers"],
                ensure_ascii=False,
            )
            writer.writerow(csv_row)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--include-prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Export judge-ready JSONL and optional CSV artifacts."""
    args = parse_args()
    source_rows = load_rows(args.input, args.include_failures)
    judge_rows = [
        convert_row(row, include_prompt=args.include_prompt) for row in source_rows
    ]

    write_jsonl(judge_rows, args.output_jsonl)
    print(f"Wrote judge JSONL: {args.output_jsonl}")

    if args.output_csv is not None:
        write_csv(judge_rows, args.output_csv)
        print(f"Wrote judge CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
