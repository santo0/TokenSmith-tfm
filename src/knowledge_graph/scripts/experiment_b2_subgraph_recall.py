"""B2 — Query subgraph expansion improves recall over dense retrieval alone.

Ablation: dense-only (FAISS), KG-only, and combined (RRF fusion).
Metric: recall@k for k in {5, 10, 20}.

Usage:
    python -m src.knowledge_graph.scripts.experiment_b2_subgraph_recall \\
        --run-dir data/knowledge_graph/runs/latest \\
        --artifacts-dir index/sections \\
        --embed-model <model-path> \\
        --output results_b2.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B2: Ablation — dense vs KG vs combined retrieval recall@k.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument("--artifacts-dir", required=True,
                        help="RAG artifacts dir (FAISS + BM25 index)")
    parser.add_argument("--index-prefix", default="textbook_index")
    parser.add_argument("--embed-model", required=True,
                        help="Embedding model path or HuggingFace name")
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--neighbor-weight", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    from src.knowledge_graph.io import (
        load_graph_and_chunks, load_canonicalization_data, resolve_run_dir
    )
    from src.knowledge_graph.query import CanonicalLookup, KGNodeRetriever
    from src.retriever import FAISSRetriever, load_artifacts
    from src.ranking.ranker import EnsembleRanker
    from src.knowledge_graph.scripts.eval_utils import (
        load_labeled_benchmarks, recall_at_k, precision_at_k, ndcg_at_k,
        scores_to_ranked_ids, print_table,
    )

    root = Path(__file__).parent.parent.parent.parent
    run_path = Path(args.run_dir)
    if not run_path.is_absolute():
        run_path = root / run_path
    artifacts_path = Path(args.artifacts_dir)
    if not artifacts_path.is_absolute():
        artifacts_path = root / artifacts_path

    print(f"Loading KG from {run_path}...")
    graph, kg_chunks = load_graph_and_chunks(str(run_path))
    resolved = resolve_run_dir(str(run_path))
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = CanonicalLookup(syn_table, can_kw, can_emb) if syn_table else None

    print(f"Loading FAISS artifacts from {artifacts_path}...")
    faiss_idx, _, raw_chunks, _, _ = load_artifacts(str(artifacts_path), args.index_prefix)

    faiss_ret = FAISSRetriever(faiss_idx, args.embed_model)
    kg_ret = KGNodeRetriever(
        graph, kg_chunks,
        neighbor_weight=args.neighbor_weight,
        num_hops=args.num_hops,
        canonical_lookup=canonical_lookup,
    )

    benchmarks = load_labeled_benchmarks(args.benchmarks)
    print(f"Evaluating on {len(benchmarks)} labeled benchmarks...")

    KS = [5, 10, 20]
    configs = {
        "dense": {"faiss": 1.0},
        "kg": {"kg_node": 1.0},
        "combined": {"faiss": 0.5, "kg_node": 0.5},
    }

    per_query = []
    for bm in benchmarks:
        query = bm["question"]
        gold: list[int] = bm["ideal_retrieved_chunks"]
        max_k = max(KS)

        # Retrieve scores from each retriever
        faiss_scores = faiss_ret.get_scores(query, max_k, raw_chunks)
        kg_scores = kg_ret.get_scores(query, max_k, list(kg_chunks.values()))

        raw_scores = {"faiss": faiss_scores, "kg_node": kg_scores}

        row = {"id": bm["id"]}
        for cfg_name, weights in configs.items():
            if len(weights) == 1:
                name = next(iter(weights))
                scores = raw_scores.get(name, {})
                ranked = scores_to_ranked_ids(scores, max_k)
            else:
                ranker = EnsembleRanker("rrf", weights)
                ranked_ids, _ = ranker.rank(raw_scores)
                ranked = ranked_ids[:max_k]

            for k in KS:
                row[f"{cfg_name}_R@{k}"] = round(recall_at_k(ranked, gold, k), 3)

        if args.verbose:
            print(f"\n[{bm['id']}] {query}")
            for cfg_name in configs:
                vals = "  ".join(f"R@{k}={row[f'{cfg_name}_R@{k}']:.2f}" for k in KS)
                print(f"  {cfg_name:10s}: {vals}")

        per_query.append(row)

    # Macro-averages
    n = len(per_query)
    cols = [f"{c}_R@{k}" for c in configs for k in KS]
    macro = {c: round(sum(r[c] for r in per_query) / n, 4) for c in cols if per_query}

    print(f"\n{'=' * 80}")
    print("B2 Results — Dense vs KG vs Combined retrieval recall@k")
    print(f"{'=' * 80}")
    display_cols = ["id"] + [f"{c}_R@{k}" for c in configs for k in KS]
    print_table(per_query, display_cols)
    print()
    print("Macro-average:")
    for c in configs:
        vals = "  ".join(f"R@{k}={macro.get(f'{c}_R@{k}', 0):.3f}" for k in KS)
        print(f"  {c:10s}: {vals}")
    print(f"{'=' * 80}")

    result = {
        "n_benchmarks": n,
        "ks": KS,
        "configs": list(configs.keys()),
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
