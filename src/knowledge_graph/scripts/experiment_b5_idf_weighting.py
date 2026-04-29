"""B5 — IDF-weighting neighbor contributions reduces common-keyword bias.

Compares KGNodeRetriever(idf_weighted=False) vs KGNodeRetriever(idf_weighted=True)
on all labeled benchmarks.
Metrics: precision@k, recall@k, NDCG@k (k ∈ {5, 10}).

Usage:
    python -m src.knowledge_graph.scripts.experiment_b5_idf_weighting \\
        --run-dir data/knowledge_graph/runs/latest \\
        --output results_b5.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B5: Standard vs IDF-weighted KGNodeRetriever.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--neighbor-weight", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    from src.knowledge_graph.io import (
        load_graph_and_chunks, load_canonicalization_data, resolve_run_dir
    )
    from src.knowledge_graph.query import CanonicalLookup, KGNodeRetriever
    from src.knowledge_graph.scripts.eval_utils import (
        load_labeled_benchmarks, recall_at_k, precision_at_k, ndcg_at_k,
        scores_to_ranked_ids, print_table,
    )

    root = Path(__file__).parent.parent.parent.parent
    run_path = Path(args.run_dir)
    if not run_path.is_absolute():
        run_path = root / run_path

    print(f"Loading KG from {run_path}...")
    graph, kg_chunks = load_graph_and_chunks(str(run_path))
    resolved = resolve_run_dir(str(run_path))
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = CanonicalLookup(syn_table, can_kw, can_emb) if syn_table else None

    kg_standard = KGNodeRetriever(
        graph, kg_chunks,
        neighbor_weight=args.neighbor_weight,
        num_hops=args.num_hops,
        canonical_lookup=canonical_lookup,
        idf_weighted=False,
    )
    kg_idf = KGNodeRetriever(
        graph, kg_chunks,
        neighbor_weight=args.neighbor_weight,
        num_hops=args.num_hops,
        canonical_lookup=canonical_lookup,
        idf_weighted=True,
    )

    benchmarks = load_labeled_benchmarks(args.benchmarks)
    print(f"Evaluating on {len(benchmarks)} labeled benchmarks...")

    KS = [5, 10]
    per_query: list[dict] = []

    for bm in benchmarks[:1]:
        query = bm["question"]
        gold: list[int] = bm["ideal_retrieved_chunks"]
        chunks_list = list(kg_chunks.values())
        max_k = max(KS)

        row: dict = {"id": bm["id"]}
        for name, ret in [("standard", kg_standard), ("idf", kg_idf)]:
            scores = ret.get_scores(query, max_k, chunks_list)
            ranked = scores_to_ranked_ids(scores, max_k)
            for k in KS:
                row[f"{name}_P@{k}"] = round(precision_at_k(ranked, gold, k), 3)
                row[f"{name}_R@{k}"] = round(recall_at_k(ranked, gold, k), 3)
                row[f"{name}_NDCG@{k}"] = round(ndcg_at_k(ranked, gold, k), 3)

        if args.verbose:
            print(f"\n[{bm['id']}] {query}")
            for name in ["standard", "idf"]:
                vals = "  ".join(
                    f"P@{k}={row[f'{name}_P@{k}']:.2f} R@{k}={row[f'{name}_R@{k}']:.2f} "
                    f"NDCG@{k}={row[f'{name}_NDCG@{k}']:.2f}"
                    for k in KS
                )
                print(f"  {name:8s}: {vals}")

        per_query.append(row)

    # Macro-averages
    n = len(per_query)
    macro: dict = {}
    for name in ["standard", "idf"]:
        for metric in ["P", "R", "NDCG"]:
            for k in KS:
                col = f"{name}_{metric}@{k}"
                macro[col] = round(sum(r[col] for r in per_query) / n, 4)

    print(f"\n{'=' * 80}")
    print(f"B5 Results — Standard vs IDF-weighted KGNodeRetriever (n={n})")
    print(f"{'=' * 80}")
    display_cols = ["id"] + [f"{name}_{m}@{k}"
                             for name in ["standard", "idf"]
                             for m in ["P", "R", "NDCG"]
                             for k in KS]
    print_table(per_query, display_cols)
    print()
    print("Macro-average:")
    for name in ["standard", "idf"]:
        vals = "  ".join(
            f"{m}@{k}={macro[f'{name}_{m}@{k}']:.3f}"
            for m in ["P", "R", "NDCG"] for k in KS
        )
        print(f"  {name:8s}: {vals}")
    print(f"{'=' * 80}")

    result = {
        "n_benchmarks": n,
        "ks": KS,
        "macro": macro,
        "per_query": per_query,
    }

    if args.output:
        out = Path(args.output)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {out}")


if __name__ == "__main__":
    load_dotenv()
    main()
