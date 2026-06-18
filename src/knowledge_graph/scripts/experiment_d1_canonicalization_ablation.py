"""D1 — Canonicalization ablation: raw graph vs canonical graph.

Compares graph topology and KG-only retrieval recall between a graph built
with the full canonicalization pipeline and one built with NullCanonicalizer
(normalize-only, no embedding clustering or LLM merging).

Topology metrics are read from each run's run_metadata.json (computed during
the build). KG recall is measured with KGNodeRetriever against benchmarks.yaml.
The canonical run uses its synonym table for query-time resolution; the raw run
has an empty synonym table, so canonical_lookup is None (direct hits only).

Usage:
    python -m src.knowledge_graph.scripts.experiment_d1_canonicalization_ablation \\
        --canonical-run-dir data/knowledge_graph/runs/latest \\
        --raw-run-dir data/knowledge_graph/runs/<raw-timestamp> \\
        --benchmarks tests/benchmarks.yaml \\
        --output results_d1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_graph_stats(run_dir: Path) -> dict:
    with open(run_dir / "run_metadata.json") as f:
        meta = json.load(f)
    return meta.get("statistics", {}).get("graph", {})


def _load_keyword_counts(run_dir: Path) -> dict:
    with open(run_dir / "synonym_table.json") as f:
        syn_table = json.load(f)
    with open(run_dir / "canonical_keywords.json") as f:
        canonical_keywords = json.load(f)
    return {
        "canonical_keywords": len(canonical_keywords),
        "synonym_table_entries": len(syn_table),
    }


def _run_kg_recall(
    run_dir: Path,
    benchmarks: list[dict],
    ks: list[int],
    num_hops: int,
    neighbor_weight: float,
) -> tuple[dict, list[dict]]:
    from src.knowledge_graph.io import load_graph_and_chunks, load_canonicalization_data, resolve_run_dir
    from src.knowledge_graph.query import CanonicalLookup, KGNodeRetriever
    from src.knowledge_graph.scripts.eval_utils import recall_at_k, scores_to_ranked_ids

    resolved = resolve_run_dir(str(run_dir))
    graph, kg_chunks = load_graph_and_chunks(resolved)
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    # Empty synonym table (raw run) → no lookup; non-empty (canonical run) → full lookup
    canonical_lookup = CanonicalLookup(syn_table, can_kw, can_emb) if syn_table else None

    kg_ret = KGNodeRetriever(
        graph, kg_chunks,
        neighbor_weight=neighbor_weight,
        num_hops=num_hops,
        canonical_lookup=canonical_lookup,
    )

    per_query: list[dict] = []
    for bm in benchmarks:
        query = bm["question"]
        gold: list[int] = bm["ideal_retrieved_chunks"]
        max_k = max(ks)
        scores = kg_ret.get_scores(query, max_k, list(kg_chunks.values()))
        ranked = scores_to_ranked_ids(scores, max_k)
        row = {"id": bm["id"]}
        for k in ks:
            row[f"R@{k}"] = round(recall_at_k(ranked, gold, k), 3)
        per_query.append(row)

    n = len(per_query)
    macro = {
        f"R@{k}": round(sum(r[f"R@{k}"] for r in per_query) / n, 4) if n else 0.0
        for k in ks
    }
    return macro, per_query


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D1: Canonicalization ablation — topology and KG recall comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--canonical-run-dir", default="data/knowledge_graph/runs/latest",
        help="Run directory for the fully canonicalized graph",
    )
    parser.add_argument(
        "--raw-run-dir", required=True,
        help="Run directory for the raw graph (built with --no-canon)",
    )
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--neighbor-weight", type=float, default=0.5)
    parser.add_argument("--output", default="results_d1.json")
    args = parser.parse_args()

    from src.knowledge_graph.scripts.eval_utils import load_labeled_benchmarks

    root = Path(__file__).parent.parent.parent.parent
    canonical_dir = Path(args.canonical_run_dir)
    if not canonical_dir.is_absolute():
        canonical_dir = root / canonical_dir
    raw_dir = Path(args.raw_run_dir)
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir

    benchmarks = load_labeled_benchmarks(args.benchmarks)
    print(f"Loaded {len(benchmarks)} labeled benchmarks")

    # ── Topology ──────────────────────────────────────────────────────────────
    print("\nLoading graph stats...")
    canon_stats = _load_graph_stats(canonical_dir)
    raw_stats = _load_graph_stats(raw_dir)
    canon_kw = _load_keyword_counts(canonical_dir)
    raw_kw = _load_keyword_counts(raw_dir)

    # ── KG Recall ─────────────────────────────────────────────────────────────
    KS = [5, 10, 20]
    print("Running KG recall on canonical graph...")
    canon_macro, canon_per_query = _run_kg_recall(
        canonical_dir, benchmarks, KS, args.num_hops, args.neighbor_weight
    )
    print("Running KG recall on raw graph...")
    raw_macro, raw_per_query = _run_kg_recall(
        raw_dir, benchmarks, KS, args.num_hops, args.neighbor_weight
    )

    # ── Print results ─────────────────────────────────────────────────────────
    SEP = "=" * 64
    print(f"\n{SEP}")
    print("D1 — Canonicalization Ablation")
    print(SEP)

    print("\nTopology comparison:")
    print(f"  {'Metric':<30} {'Canonical':>12} {'Raw':>12} {'Delta':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
    topo_fields = [
        ("nodes", "Nodes"),
        ("edges", "Edges"),
        ("avg_degree", "Mean degree"),
        ("avg_clustering", "Mean clustering"),
        ("density", "Density"),
        ("num_connected_components", "Connected components"),
        ("largest_component_size", "Largest component"),
    ]
    for field, label in topo_fields:
        c_val = canon_stats.get(field, 0)
        r_val = raw_stats.get(field, 0)
        if isinstance(c_val, float):
            delta = f"{c_val - r_val:+.4f}"
            print(f"  {label:<30} {c_val:>12.4f} {r_val:>12.4f} {delta:>10}")
        else:
            delta = f"{c_val - r_val:+d}"
            print(f"  {label:<30} {c_val:>12,} {r_val:>12,} {delta:>10}")

    print(f"\nKeyword counts:")
    print(f"  {'Metric':<30} {'Canonical':>12} {'Raw':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Canonical keywords':<30} {canon_kw['canonical_keywords']:>12,} {raw_kw['canonical_keywords']:>12,}")
    print(f"  {'Synonym table entries':<30} {canon_kw['synonym_table_entries']:>12,} {raw_kw['synonym_table_entries']:>12,}")
    reduction = (1 - canon_kw["canonical_keywords"] / raw_kw["canonical_keywords"]) * 100
    print(f"  Node reduction from merging: {reduction:.1f}%")

    print(f"\nKG-only recall (num_hops={args.num_hops}, neighbor_weight={args.neighbor_weight}):")
    print(f"  {'Config':<15} {'R@5':>8} {'R@10':>8} {'R@20':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'canonical':<15} {canon_macro['R@5']:>8.4f} {canon_macro['R@10']:>8.4f} {canon_macro['R@20']:>8.4f}")
    print(f"  {'raw':<15} {raw_macro['R@5']:>8.4f} {raw_macro['R@10']:>8.4f} {raw_macro['R@20']:>8.4f}")
    for k in KS:
        delta = canon_macro[f"R@{k}"] - raw_macro[f"R@{k}"]
        print(f"  R@{k} delta (canonical - raw): {delta:+.4f}")

    print(f"\n{SEP}")

    # ── Save ─────────────────────────────────────────────────────────────────
    result = {
        "topology": {
            "canonical": canon_stats,
            "raw": raw_stats,
        },
        "keyword_counts": {
            "canonical": canon_kw,
            "raw": raw_kw,
            "node_reduction_pct": round(reduction, 2),
        },
        "kg_recall": {
            "num_hops": args.num_hops,
            "neighbor_weight": args.neighbor_weight,
            "canonical_macro": canon_macro,
            "raw_macro": raw_macro,
            "per_query": {
                "canonical": canon_per_query,
                "raw": raw_per_query,
            },
        },
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = root / out
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
