"""Shared evaluation utilities for hypothesis experiments.

Provides recall/precision/NDCG metrics, LLM-as-judge scoring, and
benchmark loading helpers used across all experiment_*.py scripts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from src.knowledge_graph.openrouter_client import OpenRouterClient


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[int], gold_ids: list[int], k: int) -> float:
    if not gold_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & set(gold_ids)) / len(gold_ids)


def precision_at_k(retrieved_ids: list[int], gold_ids: list[int], k: int) -> float:
    if k == 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    gold_set = set(gold_ids)
    return sum(1 for r in top_k if r in gold_set) / k


def ndcg_at_k(retrieved_ids: list[int], gold_ids: list[int], k: int) -> float:
    gold_set = set(gold_ids)
    top_k = retrieved_ids[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(top_k) if r in gold_set)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / ideal if ideal > 0 else 0.0


def scores_to_ranked_ids(scores: dict[int, float], k: int | None = None) -> list[int]:
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    return ranked[:k] if k is not None else ranked


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

def grade_passages(
    client: OpenRouterClient,
    model: str,
    query: str,
    retrieved: list[tuple[int, str]],
) -> list[dict]:
    """Grade each retrieved passage for relevance (0–2) using GRADE_PROMPT.

    Returns a list of dicts with keys: chunk_id, score (int 0–2), reason.
    """
    from src.knowledge_graph.prompts import GRADE_PROMPT

    if not retrieved:
        return []

    passages = "\n\n".join(
        f"[{i + 1}] {text[:600].strip()}" for i, (_, text) in enumerate(retrieved)
    )
    prompt = GRADE_PROMPT.format(query=query, passages=passages)
    raw = client.chat(
        model,
        [{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    grades = json.loads(raw).get("grades", [])

    results = []
    for i, (chunk_id, _) in enumerate(retrieved):
        grade = next((g for g in grades if g.get("id") == i + 1), {})
        results.append({
            "chunk_id": chunk_id,
            "score": int(grade["score"]) if "score" in grade else -1,
            "reason": grade.get("reason", ""),
        })
    return results


def llm_judge(
    client: OpenRouterClient,
    model: str,
    query: str,
    retrieved: list[tuple[int, str]],
) -> float:
    """Return mean relevance score in [0, 1] for retrieved (chunk_id, text) pairs.

    Normalises the 0–2 GRADE_PROMPT scale to [0, 1].
    Returns 0.0 when no passages are retrieved or grading fails.
    """
    grades = grade_passages(client, model, query, retrieved)
    scored = [g["score"] for g in grades if g["score"] >= 0]
    if not scored:
        return 0.0
    return sum(scored) / (len(scored) * 2.0)


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------

def load_benchmarks(path: str = "tests/benchmarks.yaml") -> list[dict]:
    """Load benchmark entries from a YAML file.

    Resolves relative paths from the repo root (4 levels up from this file).
    Returns only entries that have ``ideal_retrieved_chunks`` defined.
    """
    p = Path(path)
    if not p.is_absolute():
        root = Path(__file__).parent.parent.parent.parent
        p = root / path
    with open(p) as f:
        data = yaml.safe_load(f)
    return data.get("benchmarks", [])


def load_labeled_benchmarks(path: str = "tests/benchmarks.yaml") -> list[dict]:
    """Load only benchmarks with ground-truth ``ideal_retrieved_chunks``."""
    return [b for b in load_benchmarks(path) if b.get("ideal_retrieved_chunks")]


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

def get_chunk_text(chunk_ids: list[int], chunks: dict[int, str]) -> str:
    """Concatenate texts of chunk_ids, separated by a markdown rule."""
    return "\n\n---\n\n".join(chunks[cid] for cid in chunk_ids if cid in chunks)


def retrieved_tuples(
    scores: dict[int, float],
    chunks: dict[int, str],
    k: int,
) -> list[tuple[int, str]]:
    """Return top-k (chunk_id, text) pairs sorted by score descending."""
    ranked = scores_to_ranked_ids(scores, k)
    return [(cid, chunks[cid]) for cid in ranked if cid in chunks]


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_table(rows: list[dict], columns: list[str]) -> None:
    """Print a simple ASCII table from a list of row-dicts."""
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
              for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))
