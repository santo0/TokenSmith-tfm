"""B3 — Summary tree retrieval produces better context on broad queries.

Splits benchmarks by the ``broad`` field. For broad and specific queries separately,
compares SectionSummaryRetriever vs FAISSRetriever using precision@k, recall@k, and
optionally LLM-as-judge quality scores.

Usage:
    python -m src.knowledge_graph.scripts.experiment_b3_summary_broad \\
        --run-dir data/knowledge_graph/runs/latest \\
        --artifacts-dir index/sections \\
        --embed-model <model-path> \\
        --output results_b3.json

    # With LLM judge (requires OPENROUTER_API_KEY):
    python -m src.knowledge_graph.scripts.experiment_b3_summary_broad \\
        --run-dir data/knowledge_graph/runs/latest \\
        --artifacts-dir index/sections \\
        --embed-model <model-path> \\
        --llm \\
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
    parser.add_argument("--llm", action="store_true", default=False,
                        help="Enable LLM-as-judge scoring (requires OPENROUTER_API_KEY).")
    parser.add_argument("--llm-model", default="google/gemini-3-flash-preview")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if args.llm and not api_key:
        raise SystemExit("No OPENROUTER_API_KEY — LLM-as-judge requires an API key.")

    from src.knowledge_graph.io import (
        load_graph_and_chunks, load_summary_data, resolve_run_dir
    )
    from src.knowledge_graph.query import SectionSummaryRetriever
    from src.retriever import FAISSRetriever, load_artifacts
    from src.knowledge_graph.scripts.eval_utils import (
        load_benchmarks, llm_judge, scores_to_ranked_ids, retrieved_tuples, print_table,
        recall_at_k, precision_at_k,
    )
    if args.llm:
        from src.knowledge_graph.openrouter_client import OpenRouterClient

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
    faiss_idx, _, raw_chunks, _, metadata = load_artifacts(str(artifacts_path), args.index_prefix)
    chunk_id_map = [m["chunk_id"] for m in metadata]

    faiss_ret = FAISSRetriever(faiss_idx, args.embed_model, chunk_id_map=chunk_id_map)
    summary_ret = SectionSummaryRetriever(
        summary_index, summary_entries, embed_model=args.summary_embed_model
    )
    client = OpenRouterClient(api_key, retries=2) if args.llm else None

    benchmarks = load_benchmarks(args.benchmarks)
    # Use all benchmarks that have ideal chunks; LLM judge additionally needs expected_answer
    labeled = [b for b in benchmarks if b.get("ideal_retrieved_chunks")]
    print(f"Evaluating on {len(labeled)} benchmarks...")

    raw_chunk_map = {i: raw_chunks[i] for i in range(len(raw_chunks))}

    per_query = []
    for bm in labeled:
        query = bm["question"]
        is_broad = bm.get("broad", False)
        gold_ids: list[int] = bm.get("ideal_retrieved_chunks", [])

        # Dense retrieval
        dense_scores = faiss_ret.get_scores(query, args.top_k, raw_chunks)
        dense_ids = scores_to_ranked_ids(dense_scores, args.top_k)
        dense_tuples = retrieved_tuples(dense_scores, raw_chunk_map, args.top_k)

        # Summary retrieval
        sum_scores = summary_ret.get_scores(query, args.top_k, list(kg_chunks.values()))
        sum_ids = scores_to_ranked_ids(sum_scores, args.top_k)
        sum_tuples = retrieved_tuples(sum_scores, kg_chunks, args.top_k)

        # Precision and recall from ground-truth chunk IDs
        dense_recall = recall_at_k(dense_ids, gold_ids, args.top_k)
        dense_precision = precision_at_k(dense_ids, gold_ids, args.top_k)
        sum_recall = recall_at_k(sum_ids, gold_ids, args.top_k)
        sum_precision = precision_at_k(sum_ids, gold_ids, args.top_k)

        row: dict = {
            "id": bm["id"],
            "broad": is_broad,
            "dense_recall": round(dense_recall, 3),
            "dense_precision": round(dense_precision, 3),
            "summary_recall": round(sum_recall, 3),
            "summary_precision": round(sum_precision, 3),
            "recall_delta": round(sum_recall - dense_recall, 3),
        }

        if args.llm and bm.get("expected_answer"):
            dense_judge, sum_judge = 0.0, 0.0
            try:
                if dense_tuples:
                    dense_judge = llm_judge(client, args.llm_model, query, dense_tuples)
                if sum_tuples:
                    sum_judge = llm_judge(client, args.llm_model, query, sum_tuples)
            except Exception as e:
                print(f"  [{bm['id']}] LLM judge failed: {e}")
            row["dense_judge"] = round(dense_judge, 3)
            row["summary_judge"] = round(sum_judge, 3)
            row["judge_delta"] = round(sum_judge - dense_judge, 3)

        if args.verbose:
            label = "BROAD" if is_broad else "specific"
            print(f"\n[{bm['id']}] [{label}] {query}")
            print(
                f"  dense:   recall={dense_recall:.3f}  precision={dense_precision:.3f}"
            )
            print(
                f"  summary: recall={sum_recall:.3f}  precision={sum_precision:.3f}"
                f"  Δrecall={row['recall_delta']:+.3f}"
            )
            if args.llm and "dense_judge" in row:
                print(
                    f"  judge:   dense={row['dense_judge']:.3f}"
                    f"  summary={row['summary_judge']:.3f}"
                    f"  Δ={row['judge_delta']:+.3f}"
                )

        per_query.append(row)

    # Group analysis
    broad_rows = [r for r in per_query if r["broad"]]
    specific_rows = [r for r in per_query if not r["broad"]]

    def _mean(lst: list[dict], key: str) -> float:
        vals = [r[key] for r in lst if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    base_cols = ["id", "broad", "dense_recall", "dense_precision", "summary_recall", "summary_precision", "recall_delta"]
    llm_cols = ["dense_judge", "summary_judge", "judge_delta"] if args.llm else []

    print(f"\n{'=' * 72}")
    print("B3 Results — Summary vs Dense, broad vs specific queries")
    print(f"{'=' * 72}")
    print_table(per_query, base_cols + llm_cols)
    print()
    for group_name, group in [("broad", broad_rows), ("specific", specific_rows)]:
        if not group:
            continue
        dr = _mean(group, "dense_recall")
        sr = _mean(group, "summary_recall")
        dp = _mean(group, "dense_precision")
        sp = _mean(group, "summary_precision")
        print(
            f"  {group_name:8s} (n={len(group)}): "
            f"recall  dense={dr:.3f}  summary={sr:.3f}  Δ={sr-dr:+.3f}  |  "
            f"precision  dense={dp:.3f}  summary={sp:.3f}  Δ={sp-dp:+.3f}"
        )
        if args.llm:
            dj = _mean(group, "dense_judge")
            sj = _mean(group, "summary_judge")
            print(f"           judge   dense={dj:.3f}  summary={sj:.3f}  Δ={sj-dj:+.3f}")
    print(f"{'=' * 72}")

    def _group_stats(group: list[dict]) -> dict:
        stats = {
            "n": len(group),
            "mean_dense_recall": round(_mean(group, "dense_recall"), 4),
            "mean_summary_recall": round(_mean(group, "summary_recall"), 4),
            "mean_dense_precision": round(_mean(group, "dense_precision"), 4),
            "mean_summary_precision": round(_mean(group, "summary_precision"), 4),
        }
        if args.llm:
            stats["mean_dense_judge"] = round(_mean(group, "dense_judge"), 4)
            stats["mean_summary_judge"] = round(_mean(group, "summary_judge"), 4)
        return stats

    result = {
        "n_benchmarks": len(per_query),
        "top_k": args.top_k,
        "llm_scoring": args.llm,
        "broad_summary": _group_stats(broad_rows),
        "specific_summary": _group_stats(specific_rows),
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
