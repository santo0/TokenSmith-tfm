"""
Evaluate whether embedding-based section filtering is effective.

Two strategies are tested:
  - titles:    embed chapter/subsection titles from sections.json
  - summaries: embed subsection summaries from summary_meta.json (level 2)

For each benchmark query the script reports:
  - which sections contain the ideal chunks
  - the rank at which those sections appear when sorted by cosine similarity
  - Recall@k (k=1,3,5) and MRR over all benchmarks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAGConfig  # noqa: E402
from src.embedder import CachedEmbedder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
BENCHMARKS_FILE = PROJECT_ROOT / "tests" / "benchmarks.yaml"
SECTIONS_FILE = PROJECT_ROOT / "sections.json"


def _latest_run_dir() -> Path:
    runs = sorted((PROJECT_ROOT / "data" / "knowledge_graph" / "runs").iterdir())
    if not runs:
        raise FileNotFoundError("No KG run directories found")
    return runs[-1]


# ── data loading ──────────────────────────────────────────────────────────────

def load_benchmarks() -> list[dict]:
    with open(BENCHMARKS_FILE) as f:
        return yaml.safe_load(f)["benchmarks"]


def load_sections_titles() -> list[dict]:
    """Return list of {section_number, title, text} for chapters and subsections."""
    with open(SECTIONS_FILE) as f:
        data = json.load(f)

    entries: list[dict] = []
    for part in data["parts"]:
        for ch in part["chapters"]:
            num = str(ch["chapter_number"])
            title = ch["chapter_title"]
            entries.append({
                "section_number": num,
                "level": 1,
                "text": f"Chapter {num}: {title}",
            })
            for sub in ch.get("subsections", []):
                # sub is like "2.3 Keys"
                parts = sub.split(" ", 1)
                sub_num = parts[0]
                entries.append({
                    "section_number": sub_num,
                    "level": 2,
                    "text": sub,
                })
    return entries


def load_section_summaries(run_dir: Path) -> list[dict]:
    """Return level-2 (subsection) summary entries from summary_meta.json."""
    with open(run_dir / "summary_meta.json") as f:
        all_entries = json.load(f)
    # level 2 = one summary per subsection (e.g. "1.1", "2.3")
    seen: dict[str, dict] = {}
    for e in all_entries:
        if e["level"] == 2:
            sn = e["section_number"]
            if sn not in seen:
                seen[sn] = e
    return list(seen.values())


def build_chunk_to_section_map(run_dir: Path) -> dict[int, list[str]]:
    """Map chunk_id -> list of ancestor section_numbers (most specific first)."""
    with open(run_dir / "section_tree.json") as f:
        tree = json.load(f)

    mapping: dict[int, list[str]] = defaultdict(list)

    def _walk(node: dict, ancestors: list[str]) -> None:
        sn = node.get("section_number", "")
        path = ancestors + ([sn] if sn else [])
        for cid in node.get("chunk_ids", []):
            mapping[cid] = list(reversed(path))  # most specific first
        for child in node.get("children", []):
            _walk(child, path)

    _walk(tree, [])
    return mapping


# ── embeddings ────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of row vector a against matrix b (rows)."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norms = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norms @ a_norm


# ── evaluation ────────────────────────────────────────────────────────────────

def _target_section_numbers(
    ideal_chunk_ids: list[int],
    chunk_to_section: dict[int, list[str]],
    candidate_section_numbers: set[str],
) -> set[str]:
    """
    Return the section numbers (from the candidate pool) that contain at least
    one of the ideal chunks.
    """
    targets: set[str] = set()
    for cid in ideal_chunk_ids:
        for sn in chunk_to_section.get(cid, []):
            if sn in candidate_section_numbers:
                targets.add(sn)
                break  # use most specific ancestor that is in the pool
    return targets


def evaluate(
    benchmarks: list[dict],
    sections: list[dict],
    section_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    chunk_to_section: dict[int, list[str]],
    strategy_name: str,
    top_ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    candidate_snums = {s["section_number"] for s in sections}

    recall_hits: dict[int, int] = defaultdict(int)
    mrr_sum = 0.0
    per_query: list[dict] = []

    for bm, q_emb in zip(benchmarks, query_embeddings):
        sims = cosine_sim(q_emb, section_embeddings)
        ranked_indices = np.argsort(-sims)

        ideal_chunks = bm["ideal_retrieved_chunks"]
        targets = _target_section_numbers(ideal_chunks, chunk_to_section, candidate_snums)

        if not targets:
            logger.warning("No target sections found for benchmark '%s'", bm["id"])
            per_query.append({
                "id": bm["id"],
                "targets": [],
                "top_ranks": {},
                "top_sections": [],
            })
            continue

        # rank of each target (1-indexed)
        target_first_rank: Optional[int] = None
        top_sections = []
        for rank, idx in enumerate(ranked_indices, 1):
            sn = sections[idx]["section_number"]
            top_sections.append((rank, sn, float(sims[idx])))
            if sn in targets and target_first_rank is None:
                target_first_rank = rank

        for k in top_ks:
            top_k_snums = {sections[i]["section_number"] for i in ranked_indices[:k]}
            if targets & top_k_snums:
                recall_hits[k] += 1

        mrr_sum += 1.0 / target_first_rank if target_first_rank else 0.0

        per_query.append({
            "id": bm["id"],
            "question": bm["question"],
            "targets": sorted(targets),
            "first_hit_rank": target_first_rank,
            "top_5": top_sections[:5],
        })

    n = len(benchmarks)
    metrics = {
        "strategy": strategy_name,
        "MRR": round(mrr_sum / n, 4),
        **{f"Recall@{k}": round(recall_hits[k] / n, 4) for k in top_ks},
    }
    return {"metrics": metrics, "per_query": per_query}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark section-level embedding filter")
    parser.add_argument(
        "--run-dir",
        help="Path to KG run directory (default: latest)",
    )
    parser.add_argument(
        "--strategy",
        choices=["titles", "summaries", "both"],
        default="both",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir()
    logger.info("Using KG run: %s", run_dir)

    config = RAGConfig()
    embedder = CachedEmbedder(str(PROJECT_ROOT / config.embed_model))

    benchmarks = load_benchmarks()
    chunk_to_section = build_chunk_to_section_map(run_dir)

    logger.info("Embedding %d queries...", len(benchmarks))
    queries = [bm["question"] for bm in benchmarks]
    query_embeddings = embedder.encode(queries)

    results: dict[str, dict] = {}

    # ── titles strategy ───────────────────────────────────────────────────────
    if args.strategy in ("titles", "both"):
        logger.info("Embedding section titles...")
        title_sections = load_sections_titles()
        title_texts = [s["text"] for s in title_sections]
        title_embeddings = embedder.encode(title_texts)
        results["titles"] = evaluate(
            benchmarks, title_sections, title_embeddings, query_embeddings,
            chunk_to_section, strategy_name="titles",
        )

    # ── summaries strategy ────────────────────────────────────────────────────
    if args.strategy in ("summaries", "both"):
        logger.info("Embedding section summaries...")
        summary_sections = load_section_summaries(run_dir)
        summary_texts = [s["summary_text"] for s in summary_sections]
        summary_embeddings = embedder.encode(summary_texts)
        results["summaries"] = evaluate(
            benchmarks, summary_sections, summary_embeddings, query_embeddings,
            chunk_to_section, strategy_name="summaries",
        )

    # ── print results ─────────────────────────────────────────────────────────
    for strategy, result in results.items():
        print(f"\n{'=' * 60}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'=' * 60}")
        m = result["metrics"]
        print(f"  MRR:        {m['MRR']:.4f}")
        for k in (1, 3, 5, 10):
            key = f"Recall@{k}"
            if key in m:
                print(f"  {key}:  {m[key]:.4f}")

        print("\n  Per-query breakdown:")
        for q in result["per_query"]:
            rank_str = str(q.get("first_hit_rank", "MISS"))
            target_str = ", ".join(q["targets"]) if q["targets"] else "NONE"
            print(f"    [{q['id']}]  target sections: {target_str}  |  first hit rank: {rank_str}")
            if q.get("top_5"):
                top_str = "  ".join(f"#{r} {sn}({sc:.3f})" for r, sn, sc in q["top_5"])
                print(f"       top-5: {top_str}")


if __name__ == "__main__":
    main()
