"""B4 — Full ablation: hybrid retrieval outperforms any single method.

Tests 8 retriever configurations on all labeled benchmarks.
Metrics: recall@10 (ground-truth chunks) + LLM-as-judge quality score.

Usage:
    python -m src.knowledge_graph.scripts.experiment_b4_full_ablation \\
        --run-dir data/knowledge_graph/runs/latest \\
        --artifacts-dir index/sections \\
        --embed-model <model-path> \\
        --output results_b4.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


CONFIGS: dict[str, dict[str, float]] = {
    "faiss":          {"faiss": 1.0},
    "bm25":           {"bm25": 1.0},
    "kg_node":        {"kg_node": 1.0},
    "section_tree":   {"section_tree": 1.0},
    "section_summary":{"section_summary": 1.0},
    "faiss+kg":       {"faiss": 0.5, "kg_node": 0.5},
    "faiss+summary":  {"faiss": 0.5, "section_summary": 0.5},
    "all_five":       {"faiss": 0.2, "bm25": 0.2, "kg_node": 0.2,
                       "section_tree": 0.2, "section_summary": 0.2},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B4: Full retriever ablation — recall@k + LLM quality.",
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
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--neighbor-weight", type=float, default=0.5)
    parser.add_argument("--llm-model", default="google/gemini-3-flash-preview")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM grading (recall only)")
    parser.add_argument("--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    use_llm = not args.no_llm and bool(api_key)

    from src.knowledge_graph.io import (
        load_graph_chunks_and_tree, load_summary_data,
        load_canonicalization_data, resolve_run_dir,
    )
    from src.knowledge_graph.query import (
        CanonicalLookup, KGNodeRetriever, SectionTreeRetriever, SectionSummaryRetriever,
    )
    from src.retriever import FAISSRetriever, BM25Retriever, load_artifacts
    from src.ranking.ranker import EnsembleRanker
    from src.knowledge_graph.openrouter_client import OpenRouterClient
    from src.knowledge_graph.scripts.eval_utils import (
        load_labeled_benchmarks, recall_at_k, scores_to_ranked_ids,
        retrieved_tuples, llm_judge, print_table,
    )

    root = Path(__file__).parent.parent.parent.parent
    run_path = Path(args.run_dir)
    if not run_path.is_absolute():
        run_path = root / run_path
    artifacts_path = Path(args.artifacts_dir)
    if not artifacts_path.is_absolute():
        artifacts_path = root / artifacts_path

    print(f"Loading KG from {run_path}...")
    graph, kg_chunks, section_tree = load_graph_chunks_and_tree(str(run_path))
    resolved = resolve_run_dir(str(run_path))
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = CanonicalLookup(syn_table, can_kw, can_emb) if syn_table else None
    summary_index, summary_entries = load_summary_data(str(run_path))

    print(f"Loading artifacts from {artifacts_path}...")
    faiss_idx, bm25_idx, raw_chunks, _, _ = load_artifacts(str(artifacts_path), args.index_prefix)
    raw_chunks_dict = {i: t for i, t in enumerate(raw_chunks)}

    # Build all individual retrievers
    retrievers: dict = {
        "faiss": FAISSRetriever(faiss_idx, args.embed_model),
        "bm25": BM25Retriever(bm25_idx),
        "kg_node": KGNodeRetriever(
            graph, kg_chunks,
            neighbor_weight=args.neighbor_weight,
            num_hops=args.num_hops,
            canonical_lookup=canonical_lookup,
        ),
    }
    if section_tree is not None:
        retrievers["section_tree"] = SectionTreeRetriever(
            section_tree, graph, canonical_lookup=canonical_lookup
        )
    if summary_index is not None:
        retrievers["section_summary"] = SectionSummaryRetriever(
            summary_index, summary_entries, embed_model=args.summary_embed_model
        )

    llm_client = OpenRouterClient(api_key, retries=2) if use_llm else None

    benchmarks = load_labeled_benchmarks(args.benchmarks)
    print(f"Evaluating {len(benchmarks)} benchmarks × {len(CONFIGS)} configs...")

    per_query: list[dict] = []
    for bm in benchmarks:
        query = bm["question"]
        gold: list[int] = bm["ideal_retrieved_chunks"]

        # Collect raw scores from available individual retrievers
        raw_scores: dict[str, dict[int, float]] = {}
        for rname, ret in retrievers.items():
            try:
                raw_scores[rname] = ret.get_scores(query, args.top_k, raw_chunks)
            except Exception as e:
                print(f"  [{bm['id']}] {rname} failed: {e}")
                raw_scores[rname] = {}

        row: dict = {"id": bm["id"]}
        for cfg_name, weights in CONFIGS.items():
            # Skip configs that require unavailable retrievers
            missing = [r for r in weights if r not in retrievers and r not in raw_scores]
            if missing:
                row[f"{cfg_name}_recall"] = None
                row[f"{cfg_name}_judge"] = None
                continue

            # Select only the relevant scores
            cfg_scores = {r: raw_scores.get(r, {}) for r in weights}

            if len(weights) == 1:
                name = next(iter(weights))
                ranked = scores_to_ranked_ids(cfg_scores[name], args.top_k)
            else:
                ranker = EnsembleRanker("rrf", weights)
                ranked, _ = ranker.rank(cfg_scores)
                ranked = ranked[:args.top_k]

            row[f"{cfg_name}_recall"] = round(recall_at_k(ranked, gold, args.top_k), 3)

            if llm_client:
                chunks_for_judge = [(cid, kg_chunks.get(cid, raw_chunks_dict.get(cid, "")))
                                     for cid in ranked if cid in kg_chunks or cid in raw_chunks_dict]
                try:
                    score = llm_judge(llm_client, args.llm_model, query, chunks_for_judge)
                    row[f"{cfg_name}_judge"] = round(score, 3)
                except Exception as e:
                    print(f"  [{bm['id']}] [{cfg_name}] LLM judge failed: {e}")
                    row[f"{cfg_name}_judge"] = None

        if args.verbose:
            print(f"\n[{bm['id']}]")
            for cfg_name in CONFIGS:
                r = row.get(f"{cfg_name}_recall", "N/A")
                j = row.get(f"{cfg_name}_judge", "N/A")
                print(f"  {cfg_name:18s}: recall={r}  judge={j}")

        per_query.append(row)

    # Summary table
    n = len(per_query)
    print(f"\n{'=' * 90}")
    print(f"B4 Results — Full ablation: recall@{args.top_k} + LLM judge (n={n} queries)")
    print(f"{'=' * 90}")

    summary_rows = []
    for cfg_name in CONFIGS:
        recall_vals = [r[f"{cfg_name}_recall"] for r in per_query if r.get(f"{cfg_name}_recall") is not None]
        judge_vals = [r[f"{cfg_name}_judge"] for r in per_query if r.get(f"{cfg_name}_judge") is not None]
        summary_rows.append({
            "config": cfg_name,
            f"recall@{args.top_k}": round(sum(recall_vals) / len(recall_vals), 3) if recall_vals else None,
            "llm_judge": round(sum(judge_vals) / len(judge_vals), 3) if judge_vals else None,
        })

    print_table(summary_rows, ["config", f"recall@{args.top_k}", "llm_judge"])
    print(f"{'=' * 90}")

    result = {
        "n_benchmarks": n,
        "top_k": args.top_k,
        "configs": list(CONFIGS.keys()),
        "macro": {
            cfg_name: {
                "recall": next((r[f"recall@{args.top_k}"] for r in summary_rows if r["config"] == cfg_name), None),
                "judge": next((r["llm_judge"] for r in summary_rows if r["config"] == cfg_name), None),
            }
            for cfg_name in CONFIGS
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
