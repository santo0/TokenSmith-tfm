"""B3 — Summary tree retrieval produces better context on broad queries.

Splits benchmarks by the ``broad`` field. For broad and specific queries separately,
compares SectionSummaryRetriever vs FAISSRetriever using LLM-as-judge quality scores.

Usage:
    python -m src.knowledge_graph.scripts.experiment_b3_summary_broad \\
        --run-dir data/knowledge_graph/runs/latest \\
        --artifacts-dir index/sections \\
        --embed-model <model-path> \\
        --output results_b3.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B3: Summary tree vs dense for broad vs specific queries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--index-prefix", default="textbook_index")
    parser.add_argument("--embed-model", required=True)
    parser.add_argument(
        "--summary-embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model used for summary index (must match build time)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm-model", default="google/gemini-3-flash-preview")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("No OPENROUTER_API_KEY — LLM-as-judge requires an API key.")

    from src.knowledge_graph.io import (
        load_graph_and_chunks, load_summary_data, resolve_run_dir
    )
    from src.knowledge_graph.query import SectionSummaryRetriever
    from src.retriever import FAISSRetriever, load_artifacts
    from src.knowledge_graph.openrouter_client import OpenRouterClient
    from src.knowledge_graph.scripts.eval_utils import (
        load_benchmarks, llm_judge, scores_to_ranked_ids, retrieved_tuples, print_table,
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
    summary_index, summary_entries = load_summary_data(str(run_path))

    if summary_index is None:
        raise SystemExit("No summary index found — run the summary tree builder first.")

    print(f"Loading FAISS from {artifacts_path}...")
    faiss_idx, _, raw_chunks, _, _ = load_artifacts(str(artifacts_path), args.index_prefix)

    faiss_ret = FAISSRetriever(faiss_idx, args.embed_model)
    summary_ret = SectionSummaryRetriever(
        summary_index, summary_entries, embed_model=args.summary_embed_model
    )
    client = OpenRouterClient(api_key, retries=2)

    benchmarks = load_benchmarks(args.benchmarks)
    # Use all benchmarks that have expected_answer (for LLM judge reference)
    labeled = [b for b in benchmarks if b.get("expected_answer")]
    print(f"Evaluating on {len(labeled)} benchmarks (with LLM judge)...")

    per_query = []
    for bm in labeled:
        query = bm["question"]
        ref_answer = bm.get("expected_answer", "")
        is_broad = bm.get("broad", False)

        # Dense retrieval
        dense_scores = faiss_ret.get_scores(query, args.top_k, raw_chunks)
        dense_tuples = retrieved_tuples(dense_scores, {i: raw_chunks[i] for i in range(len(raw_chunks))}, args.top_k)

        # Summary retrieval
        sum_scores = summary_ret.get_scores(query, args.top_k, list(kg_chunks.values()))
        sum_tuples = retrieved_tuples(sum_scores, kg_chunks, args.top_k)

        dense_judge, sum_judge = 0.0, 0.0
        try:
            if dense_tuples:
                dense_judge = llm_judge(client, args.llm_model, query, dense_tuples)
            if sum_tuples:
                sum_judge = llm_judge(client, args.llm_model, query, sum_tuples)
        except Exception as e:
            print(f"  [{bm['id']}] LLM judge failed: {e}")

        row = {
            "id": bm["id"],
            "broad": is_broad,
            "dense_judge": round(dense_judge, 3),
            "summary_judge": round(sum_judge, 3),
            "delta": round(sum_judge - dense_judge, 3),
        }

        if args.verbose:
            label = "BROAD" if is_broad else "specific"
            print(f"\n[{bm['id']}] [{label}] {query}")
            print(f"  dense={dense_judge:.3f}  summary={sum_judge:.3f}  Δ={row['delta']:+.3f}")

        per_query.append(row)

    # Group analysis
    broad_rows = [r for r in per_query if r["broad"]]
    specific_rows = [r for r in per_query if not r["broad"]]

    def _mean(lst: list[dict], key: str) -> float:
        vals = [r[key] for r in lst if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n{'=' * 60}")
    print("B3 Results — Summary vs Dense, broad vs specific queries")
    print(f"{'=' * 60}")
    print_table(per_query, ["id", "broad", "dense_judge", "summary_judge", "delta"])
    print()
    for group_name, group in [("broad", broad_rows), ("specific", specific_rows)]:
        if group:
            d = _mean(group, "dense_judge")
            s = _mean(group, "summary_judge")
            print(f"  {group_name:8s} (n={len(group)}):  dense={d:.3f}  summary={s:.3f}  Δ={s-d:+.3f}")
    print(f"{'=' * 60}")

    result = {
        "n_benchmarks": len(per_query),
        "top_k": args.top_k,
        "broad_summary": {
            "n": len(broad_rows),
            "mean_dense": round(_mean(broad_rows, "dense_judge"), 4),
            "mean_summary": round(_mean(broad_rows, "summary_judge"), 4),
        },
        "specific_summary": {
            "n": len(specific_rows),
            "mean_dense": round(_mean(specific_rows, "dense_judge"), 4),
            "mean_summary": round(_mean(specific_rows, "summary_judge"), 4),
        },
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
