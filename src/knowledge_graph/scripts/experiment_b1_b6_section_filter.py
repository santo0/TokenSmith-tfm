"""B1/B6 — Section title/summary embeddings as a pre-filter for vector search.

B1: For each benchmark query, embed the query and compare against section heading
    embeddings (SectionTreeRetriever). Check if the gold sections appear in the
    top-k retrieved sections. Metric: recall@k (k ∈ {1, 3, 5, 10}).

B6: Same but using the summary FAISS index (SectionSummaryRetriever).

Gold sections come from the `sections` field in benchmarks.yaml (populated by
populate_benchmark_sections.py). Entries without that field are skipped.

Also computes a random baseline.

Usage:
    python -m src.knowledge_graph.scripts.experiment_b1_b6_section_filter \\
        --run-dir data/knowledge_graph/runs/latest \\
        --output results_b1_b6.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv


def _chunk_scores_to_section_scores(
    chunk_scores: dict[int, float],
    node_index: dict,
) -> dict[str, float]:
    """Aggregate chunk-level scores to section-level by taking max per section."""
    section_scores: dict[str, float] = {}
    for heading, node in node_index.items():
        if node.chunk_ids:
            section_scores[heading] = max(
                chunk_scores.get(cid, 0.0) for cid in node.chunk_ids
            )
        else:
            section_scores[heading] = 0.0
    return section_scores


def _recall_at_k_str(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B1/B6: Section heading vs summary pre-filter recall@k (section level).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model (must match KG build)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    from src.knowledge_graph.io import (
        load_graph_chunks_and_tree, load_summary_data, load_canonicalization_data, resolve_run_dir
    )
    from src.knowledge_graph.query import (
        CanonicalLookup, SectionTreeRetriever, SectionSummaryRetriever
    )
    from src.knowledge_graph.scripts.eval_utils import (
        load_benchmarks, print_table,
    )

    root = Path(__file__).parent.parent.parent.parent
    run_path = Path(args.run_dir)
    if not run_path.is_absolute():
        run_path = root / run_path

    print(f"Loading KG and tree from {run_path}...")
    graph, kg_chunks, section_tree = load_graph_chunks_and_tree(str(run_path))
    resolved = resolve_run_dir(str(run_path))
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = CanonicalLookup(syn_table, can_kw, can_emb) if syn_table else None
    summary_index, summary_entries = load_summary_data(str(run_path))

    if section_tree is None:
        print("WARNING: no section_tree.json found — B1 heading filter will be skipped.")
    if summary_index is None:
        print("WARNING: no summary_index.faiss found — B6 summary filter will be skipped.")

    # Build retrievers
    section_ret = None
    if section_tree is not None:
        section_ret = SectionTreeRetriever(section_tree, graph, canonical_lookup=canonical_lookup)

    summary_ret = None
    if summary_index is not None:
        summary_ret = SectionSummaryRetriever(
            summary_index, summary_entries, embed_model=args.embed_model
        )

    all_benchmarks = load_benchmarks(args.benchmarks)
    benchmarks = [b for b in all_benchmarks if b.get("sections")]
    n_skipped = len(all_benchmarks) - len(benchmarks)
    if n_skipped:
        print(f"Skipping {n_skipped} entries without a `sections` field.")
    print(f"Evaluating on {len(benchmarks)} benchmarks...")

    KS = [1, 3, 5, 10]
    random.seed(args.seed)

    node_index = section_tree.node_index if section_tree is not None else {}
    all_section_headings = list(node_index.keys())
    total_sections = len(all_section_headings)

    per_query = []
    for bm in benchmarks:
        query = bm["question"]
        gold: list[str] = bm["sections"]
        gold_set = set(gold)
        max_k = max(KS)

        row = {"id": bm["id"]}

        # B1: section heading retriever
        if section_ret is not None:
            chunk_scores = section_ret.get_scores(query, max_k, list(kg_chunks.values()))
            section_scores = _chunk_scores_to_section_scores(chunk_scores, node_index)
            ranked = sorted(section_scores, key=section_scores.__getitem__, reverse=True)
            for k in KS:
                row[f"heading_R@{k}"] = round(_recall_at_k_str(ranked, gold, k), 3)
        else:
            for k in KS:
                row[f"heading_R@{k}"] = None

        # B6: summary retriever
        if summary_ret is not None:
            chunk_scores = summary_ret.get_scores(query, max_k, list(kg_chunks.values()))
            section_scores = _chunk_scores_to_section_scores(chunk_scores, node_index)
            ranked = sorted(section_scores, key=section_scores.__getitem__, reverse=True)
            for k in KS:
                row[f"summary_R@{k}"] = round(_recall_at_k_str(ranked, gold, k), 3)
        else:
            for k in KS:
                row[f"summary_R@{k}"] = None

        # Random baseline
        random_ranked = random.sample(all_section_headings, min(max_k, total_sections))
        for k in KS:
            row[f"random_R@{k}"] = round(_recall_at_k_str(random_ranked, gold, k), 3)

        if args.verbose:
            print(f"\n[{bm['id']}] {query}")
            print(f"  gold sections: {gold}")
            for label in ["heading", "summary", "random"]:
                vals = "  ".join(f"R@{k}={row.get(f'{label}_R@{k}', 'N/A')}" for k in KS)
                print(f"  {label:10s}: {vals}")

        per_query.append(row)

    # Macro-averages (skip None values)
    n = len(per_query)
    print(f"\n{'=' * 80}")
    print("B1/B6 Results — Section pre-filter recall@k")
    print(f"{'=' * 80}")

    display_cols = ["id"] + [f"{lbl}_R@{k}"
                              for lbl in ["heading", "summary", "random"]
                              for k in KS]
    available = [c for c in display_cols if any(c in r and r[c] is not None for r in per_query)]
    print_table(per_query, ["id"] + [c for c in available if c != "id"])

    print("\nMacro-average:")
    for lbl in ["heading", "summary", "random"]:
        vals_parts = []
        for k in KS:
            col = f"{lbl}_R@{k}"
            valid = [r[col] for r in per_query if r.get(col) is not None]
            if valid:
                vals_parts.append(f"R@{k}={sum(valid)/len(valid):.3f}")
        if vals_parts:
            print(f"  {lbl:10s}: {'  '.join(vals_parts)}")
    print(f"{'=' * 80}")

    result = {
        "n_benchmarks": n,
        "ks": KS,
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
