"""Estimate the retrieval difficulty of benchmark queries using the KG analysis pipeline.

For each query in a benchmarks YAML file, computes the five difficulty dimensions
(D1–D5) and a composite score, then prints a ranked summary table and optionally
saves full results to JSON.

D2 (retrieval confidence) and D3 (context capacity) require retrieval scores and
chunks, which are not available in this offline pipeline — they default to 0.0.

Usage
-----
    python -m src.knowledge_graph.scripts.benchmark_difficulty \\
        --run-dir data/knowledge_graph/runs/latest \\
        --benchmarks tests/benchmarks.yaml \\
        --output difficulty_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import yaml

from src.knowledge_graph.analysis import analyze_query
from src.knowledge_graph.io import RUNS_DIR, load_canonicalization_data, load_graph, resolve_run_dir
from src.knowledge_graph.query import CanonicalLookup

logger = logging.getLogger(__name__)


def _load_benchmarks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("benchmarks", [])


def _print_table(rows: list[dict]) -> None:
    """Print a compact summary table sorted by composite difficulty score (desc)."""
    sorted_rows = sorted(rows, key=lambda r: r["score"], reverse=True)

    header = (
        f"{'ID':<28}  {'D1:cov':>6}  {'D4:top':>6}  {'D5:com':>6}"
        f"  {'score':>6}  {'cat':<6}  {'nodes':>5}  query"
    )
    print("\n" + header)
    print("-" * len(header))

    for r in sorted_rows:
        f = r["features"]
        query_short = r["query"][:55] + ("…" if len(r["query"]) > 55 else "")
        print(
            f"{r['id']:<28}  "
            f"{f['corpus_coverage']:>6.3f}  "
            f"{f['topological_complexity']:>6.3f}  "
            f"{f['community_dispersion']:>6.3f}  "
            f"{r['score']:>6.3f}  "
            f"{r['category']:<6}  "
            f"{f['subgraph_node_count']:>5}  "
            f"{query_short}"
        )

    print()
    easy = sum(1 for r in rows if r["category"] == "easy")
    medium = sum(1 for r in rows if r["category"] == "medium")
    hard = sum(1 for r in rows if r["category"] == "hard")
    print(f"Totals — easy: {easy}  medium: {medium}  hard: {hard}  (of {len(rows)} queries)")


def run_difficulty_benchmark(
    run_dir: str,
    benchmarks: list[dict],
    community_map: dict[str, int] | None = None,
    weights: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2),
    category_thresholds: tuple[float, float] = (0.33, 0.67),
) -> list[dict]:
    """Run difficulty analysis on all benchmark queries.

    Returns a list of result dicts, one per query, each containing the query ID,
    question text, difficulty score, category, and the full features dict.
    """
    graph_path = os.path.join(run_dir, "graph.json")
    graph = load_graph(graph_path)
    logger.info("Loaded graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

    synonym_table, canonical_keywords, canonical_embeddings = load_canonicalization_data(run_dir)
    if synonym_table is not None:
        canonical_lookup = CanonicalLookup(synonym_table, canonical_keywords, canonical_embeddings)
        logger.info("Canonicalization data loaded (%d synonyms)", len(synonym_table))
    else:
        canonical_lookup = None
        logger.warning("Canonicalization data not found — running without synonym resolution")

    results = []
    for bench in benchmarks:
        qid = bench.get("id", "?")
        query = bench.get("question", "")
        logger.debug("Analysing [%s]: %s", qid, query)

        result = analyze_query(
            query=query,
            graph=graph,
            canonical_lookup=canonical_lookup,
            community_map=community_map,
            weights=weights,
            category_thresholds=category_thresholds,
        )

        results.append({
            "id": qid,
            "mode": bench.get("mode", "standard"),
            "query": query,
            "score": result.difficulty.score,
            "category": result.difficulty.category.value,
            "components": result.difficulty.components.to_dict(),
            "features": result.features.to_dict(),
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate query difficulty for all benchmark queries."
    )
    parser.add_argument(
        "--run-dir",
        default=os.path.join(RUNS_DIR, "latest"),
        help="KG run directory (must contain graph.json and optionally canonicalization files).",
    )
    parser.add_argument(
        "--benchmarks",
        default=str(Path(__file__).parents[3] / "tests" / "benchmarks.yaml"),
        help="Path to benchmarks YAML file (default: tests/benchmarks.yaml).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write full JSON results.",
    )
    parser.add_argument(
        "--weights",
        nargs=5,
        type=float,
        metavar=("W1", "W2", "W3", "W4", "W5"),
        default=[0.2, 0.2, 0.2, 0.2, 0.2],
        help="Weights for D1–D5 in composite score (must sum to 1).",
    )
    parser.add_argument(
        "--easy-threshold",
        type=float,
        default=0.33,
        help="Score ≤ threshold → EASY (default 0.33).",
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=0.67,
        help="Score ≤ threshold → MEDIUM, else HARD (default 0.67).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_dir = resolve_run_dir(args.run_dir)
    benchmarks = _load_benchmarks(args.benchmarks)
    logger.info("Loaded %d benchmark queries from %s", len(benchmarks), args.benchmarks)

    results = run_difficulty_benchmark(
        run_dir=run_dir,
        benchmarks=benchmarks,
        weights=tuple(args.weights),
        category_thresholds=(args.easy_threshold, args.medium_threshold),
    )

    _print_table(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
