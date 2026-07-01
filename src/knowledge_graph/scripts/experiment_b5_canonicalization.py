"""B5 — Canonicalization and retrieval quality.

Compares KGNodeRetriever (IDF-weighted, the B4 default) on the canonical graph
against the same retriever on the raw (non-canonicalized) graph.  Only
KGNodeRetriever is evaluated because FAISS/BM25/SectionSummary embed raw text
and are invariant to the graph's canonical node set.

Query-node seeding note: on the canonical graph, the synonym table resolves
query terms to canonical nodes; on the raw graph there is no synonym table, so
seeding falls back to exact n-gram matching against raw surface forms.  B5
therefore measures the *joint* effect of canonicalization on both graph structure
and query seeding — the realistic deployment comparison.

Usage:
    python -m src.knowledge_graph.scripts.experiment_b5_canonicalization \\
        --canonical-run-dir  data/knowledge_graph/runs/latest \\
        --raw-run-dir        data/knowledge_graph/runs/2026-06-17_11-58-04 \\
        --benchmarks         tests/benchmarks.yaml \\
        --output             results_b5_canonicalization.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_retriever(run_dir: str, idf_weighted: bool):
    """Load KGNodeRetriever (IDF-weighted) from a run directory.

    Returns (retriever, kg_chunks, n_nodes, n_edges, has_canon).
    """
    from src.knowledge_graph.io import (
        load_graph_and_chunks, load_canonicalization_data, resolve_run_dir,
    )
    from src.knowledge_graph.query import CanonicalLookup, KGNodeRetriever

    root = Path(__file__).parent.parent.parent.parent
    run_path = Path(run_dir) if Path(run_dir).is_absolute() else root / run_dir

    graph, kg_chunks = load_graph_and_chunks(str(run_path))
    resolved = resolve_run_dir(str(run_path))
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = CanonicalLookup(syn_table, can_kw, can_emb) if syn_table else None

    retriever = KGNodeRetriever(
        graph, kg_chunks,
        canonical_lookup=canonical_lookup,
        idf_weighted=idf_weighted,
    )
    has_canon = canonical_lookup is not None
    return retriever, kg_chunks, graph.number_of_nodes(), graph.number_of_edges(), has_canon


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B5: Canonical vs raw graph — KGNodeRetriever (IDF-weighted).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--canonical-run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--raw-run-dir",
                        default="data/knowledge_graph/runs/2026-06-17_11-58-04")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    from src.knowledge_graph.scripts.eval_utils import (
        load_labeled_benchmarks, recall_at_k, precision_at_k,
        scores_to_ranked_ids,
    )

    KS = [5, 10, 20]

    # Load both retrievers
    print("Loading canonical graph...")
    ret_can, chunks_can, n_nodes_can, n_edges_can, has_canon = _load_retriever(
        args.canonical_run_dir, idf_weighted=True
    )
    print(f"  {n_nodes_can} nodes, {n_edges_can} edges, canonical_lookup={has_canon}")

    print("Loading raw graph...")
    ret_raw, chunks_raw, n_nodes_raw, n_edges_raw, has_canon_raw = _load_retriever(
        args.raw_run_dir, idf_weighted=True
    )
    print(f"  {n_nodes_raw} nodes, {n_edges_raw} edges, canonical_lookup={has_canon_raw}")

    benchmarks = load_labeled_benchmarks(args.benchmarks)
    n = len(benchmarks)
    print(f"\nEvaluating {n} labeled benchmarks...")

    per_query: list[dict] = []

    for bm in benchmarks:
        query = bm["question"]
        gold: list[int] = bm["ideal_retrieved_chunks"]
        max_k = max(KS)

        row: dict = {"id": bm["id"]}

        for label, ret, chunks in [
            ("canonical", ret_can, chunks_can),
            ("raw",       ret_raw, chunks_raw),
        ]:
            scores = ret.get_scores(query, max_k, list(chunks.values()))
            ranked = scores_to_ranked_ids(scores, max_k)
            for k in KS:
                row[f"{label}_P@{k}"] = round(precision_at_k(ranked, gold, k), 4)
                row[f"{label}_R@{k}"] = round(recall_at_k(ranked, gold, k), 4)

        row["delta_R@10"] = round(row["canonical_R@10"] - row["raw_R@10"], 4)

        if args.verbose:
            can_str = "  ".join(f"R@{k}={row[f'canonical_R@{k}']:.3f}" for k in KS)
            raw_str = "  ".join(f"R@{k}={row[f'raw_R@{k}']:.3f}" for k in KS)
            print(f"  [{bm['id']}]")
            print(f"    canonical: {can_str}  delta_R@10={row['delta_R@10']:+.3f}")
            print(f"    raw:       {raw_str}")

        per_query.append(row)

    # Macro-averages
    def _mean(col: str) -> float:
        vals = [r[col] for r in per_query if r.get(col) is not None]
        return round(sum(vals) / len(vals), 4) if vals else float("nan")

    macro: dict = {}
    for label in ("canonical", "raw"):
        for metric in ("P", "R"):
            for k in KS:
                col = f"{label}_{metric}@{k}"
                macro[col] = _mean(col)

    # Per-query recall@10 change distribution
    improved = sum(1 for r in per_query if r["delta_R@10"] > 0.001)
    regressed = sum(1 for r in per_query if r["delta_R@10"] < -0.001)
    flat = n - improved - regressed

    # Print results
    print(f"\n{'=' * 72}")
    print(f"B5 — Canonicalization and Retrieval Quality (n={n})")
    print(f"{'=' * 72}")
    print(f"\n  {'Graph':<12} " +
          "  ".join(f"P@{k}    R@{k}" for k in KS))
    print("  " + "-" * 64)
    for label in ("canonical", "raw"):
        vals = "  ".join(
            f"{macro[f'{label}_P@{k}']:.4f}  {macro[f'{label}_R@{k}']:.4f}"
            for k in KS
        )
        print(f"  {label:<12} {vals}")

    delta_vals = "  ".join(
        f"{macro[f'canonical_R@{k}'] - macro[f'raw_R@{k}']:+.4f}        "
        for k in KS
    )
    print(f"  {'delta(R)':<12} {delta_vals}")

    print(f"\n  Per-query R@10 changes (n={n}):")
    print(f"    canonical > raw : {improved} queries")
    print(f"    canonical = raw : {flat} queries")
    print(f"    canonical < raw : {regressed} queries")

    # Per-query table (sorted by delta)
    print("\n  Per-query recall@10 (sorted by delta):")
    header = f"  {'id':<36} {'can_R@10':>9} {'raw_R@10':>9} {'Δ':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in sorted(per_query, key=lambda x: x["delta_R@10"], reverse=True):
        print(f"  {r['id']:<36} {r['canonical_R@10']:>9.4f} "
              f"{r['raw_R@10']:>9.4f} {r['delta_R@10']:>+8.4f}")

    print(f"\n{'=' * 72}")

    result = {
        "n_benchmarks": n,
        "ks": KS,
        "graphs": {
            "canonical": {
                "run_dir":  args.canonical_run_dir,
                "n_nodes":  n_nodes_can,
                "n_edges":  n_edges_can,
                "has_canonical_lookup": has_canon,
            },
            "raw": {
                "run_dir":  args.raw_run_dir,
                "n_nodes":  n_nodes_raw,
                "n_edges":  n_edges_raw,
                "has_canonical_lookup": has_canon_raw,
            },
        },
        "idf_weighted": True,
        "macro": macro,
        "per_query_summary": {
            "improved":  improved,
            "flat":      flat,
            "regressed": regressed,
        },
        "per_query": per_query,
    }

    if args.output:
        out = Path(args.output) if Path(args.output).is_absolute() else \
              Path(__file__).parent.parent.parent.parent / args.output
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {out}")


if __name__ == "__main__":
    main()
