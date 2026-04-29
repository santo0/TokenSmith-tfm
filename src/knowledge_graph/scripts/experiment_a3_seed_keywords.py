"""A3 — Embedding-based seed keywords vs keyword matching for query nodes.

For each labeled benchmark query, compares two query-node extraction strategies:
  1. extract_query_nodes      — exact n-gram match + canonical lookup
  2. extract_query_nodes_embedding — pure embedding-based match

Evaluation: precision, recall, F1 against the benchmark's gold ``keywords`` field
(after normalization via the same Normalizer used in the graph).

Usage:
    python -m src.knowledge_graph.scripts.experiment_a3_seed_keywords \\
        --run-dir data/knowledge_graph/runs/latest \\
        --benchmarks tests/benchmarks.yaml \\
        --output results_a3.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _prf(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    tp = len(predicted & gold)
    p = tp / len(predicted)
    r = tp / len(gold)
    return p, r, _f1(p, r)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A3: Compare exact-match vs embedding seed-keyword extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for embedding extraction (must match KG build)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for embedding extraction")
    parser.add_argument("--sim-threshold", type=float, default=0.40,
                        help="Minimum cosine similarity for embedding match")
    parser.add_argument("--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    from src.knowledge_graph.io import load_graph_and_chunks, load_canonicalization_data, load_keyword_index, resolve_run_dir
    from src.knowledge_graph.query import CanonicalLookup, extract_query_nodes, extract_query_nodes_embedding
    from src.knowledge_graph.normalizer import Normalizer
    from src.knowledge_graph.scripts.eval_utils import load_benchmarks, print_table

    root = Path(__file__).parent.parent.parent.parent
    run_dir_path = Path(args.run_dir)
    if not run_dir_path.is_absolute():
        run_dir_path = root / run_dir_path

    print(f"Loading KG from {run_dir_path}...")
    graph, chunks = load_graph_and_chunks(str(run_dir_path))
    resolved = resolve_run_dir(str(run_dir_path))
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = CanonicalLookup(syn_table, can_kw, can_emb) if syn_table else None
    keyword_index = load_keyword_index(resolved)

    normalizer = Normalizer()
    benchmarks = load_benchmarks(args.benchmarks)
    # Use all benchmarks that have keywords
    labeled = [b for b in benchmarks if b.get("keywords")]

    print(f"Evaluating on {len(labeled)} benchmarks with gold keywords...")

    rows = []
    for bm in labeled:
        query = bm["question"]
        gold_raw: list[str] = bm["keywords"]
        gold_norm = set(normalizer.normalize(gold_raw))

        # Strategy 1: exact n-gram match
        exact_nodes = set(extract_query_nodes(query, graph, canonical_lookup))
        exact_norm = set(normalizer.normalize(list(exact_nodes)))

        # Strategy 2: embedding-based
        emb_nodes: set[str] = set()
        if keyword_index is not None and can_kw:
            emb_nodes = set(extract_query_nodes_embedding(
                query, graph, keyword_index, can_kw,
                embedding_model=args.embed_model,
                top_k=args.top_k,
                similarity_threshold=args.sim_threshold,
            ))
        emb_norm = set(normalizer.normalize(list(emb_nodes)))

        p_ex, r_ex, f_ex = _prf(exact_norm, gold_norm)
        p_em, r_em, f_em = _prf(emb_norm, gold_norm)

        if args.verbose:
            print(f"\n[{bm['id']}] {query}")
            print(f"  Gold:          {sorted(gold_norm)}")
            print(f"  Exact matched: {sorted(exact_norm)}")
            print(f"  Emb matched:   {sorted(emb_norm)}")
            print(f"  Exact  P={p_ex:.2f} R={r_ex:.2f} F1={f_ex:.2f}")
            print(f"  Emb    P={p_em:.2f} R={r_em:.2f} F1={f_em:.2f}")

        rows.append({
            "id": bm["id"],
            "gold_n": len(gold_norm),
            "exact_P": round(p_ex, 3),
            "exact_R": round(r_ex, 3),
            "exact_F1": round(f_ex, 3),
            "emb_P": round(p_em, 3),
            "emb_R": round(r_em, 3),
            "emb_F1": round(f_em, 3),
        })

    # Macro-averages
    n = len(rows)
    macro = {
        "exact_P": sum(r["exact_P"] for r in rows) / n,
        "exact_R": sum(r["exact_R"] for r in rows) / n,
        "exact_F1": sum(r["exact_F1"] for r in rows) / n,
        "emb_P": sum(r["emb_P"] for r in rows) / n,
        "emb_R": sum(r["emb_R"] for r in rows) / n,
        "emb_F1": sum(r["emb_F1"] for r in rows) / n,
    }

    print(f"\n{'=' * 70}")
    print("A3 Results — seed keyword extraction precision/recall/F1 vs gold keywords")
    print(f"{'=' * 70}")
    print_table(rows, ["id", "gold_n", "exact_P", "exact_R", "exact_F1",
                        "emb_P", "emb_R", "emb_F1"])
    print()
    print(f"  Macro-average  exact: P={macro['exact_P']:.3f}  R={macro['exact_R']:.3f}  F1={macro['exact_F1']:.3f}")
    print(f"  Macro-average  embed: P={macro['emb_P']:.3f}  R={macro['emb_R']:.3f}  F1={macro['emb_F1']:.3f}")
    better = "embedding" if macro["emb_F1"] > macro["exact_F1"] else "exact-match"
    print(f"  Higher macro-F1: {better}")
    print(f"{'=' * 70}")

    result = {
        "n_benchmarks": n,
        "macro": {k: round(v, 4) for k, v in macro.items()},
        "per_query": rows,
        "config": {
            "embed_model": args.embed_model,
            "top_k": args.top_k,
            "sim_threshold": args.sim_threshold,
        },
    }

    if args.output:
        out = Path(args.output)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {out}")


if __name__ == "__main__":
    load_dotenv()
    main()
