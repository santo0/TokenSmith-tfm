"""B1/B6 — Section title/summary embeddings as a pre-filter for vector search.

B1: For each benchmark query, score sections directly using SectionTree heading/KG
    keyword overlap and check if the gold sections appear in the top-k ranked
    sections. Metric: recall@k (k ∈ {1, 3, 5, 10}).

B6: Same but using direct FAISS similarity against section summary embeddings.
    Sections are ranked by their own summary similarity (chapter-level entries
    are excluded to avoid inflating scores for all sections in a matched chapter).

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

import faiss
from dotenv import load_dotenv


def _recall_at_k_str(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def _summary_section_scores(
    query: str,
    summary_index,
    summary_entries: list,
    number_index: dict,
    embed_model_name: str,
    top_k: int,
) -> dict[str, float]:
    """Score sections directly from FAISS summary similarities.

    Searches the summary index and maps each hit to its section heading via
    section_number. Chapter-level entries (level == 1) are skipped because they
    would inflate scores for every section in a matched chapter.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embed_model_name)

    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)

    k = min(top_k, summary_index.ntotal)
    similarities, indices = summary_index.search(q_emb, k)

    section_scores: dict[str, float] = {}
    for sim, idx in zip(similarities[0], indices[0]):
        if idx < 0 or sim <= 0:
            continue
        entry = summary_entries[idx]
        if entry.level == 1:  # chapter-level: too coarse, skip
            continue
        node = number_index.get(entry.section_number)
        if node is None:
            continue
        heading = node.heading
        section_scores[heading] = max(section_scores.get(heading, 0.0), float(sim))

    return section_scores


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
        CanonicalLookup, SectionTreeRetriever, extract_query_nodes
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

    section_ret = None
    if section_tree is not None:
        section_ret = SectionTreeRetriever(section_tree, graph, canonical_lookup=canonical_lookup)

    all_benchmarks = load_benchmarks(args.benchmarks)
    benchmarks = [b for b in all_benchmarks if b.get("sections")]
    n_skipped = len(all_benchmarks) - len(benchmarks)
    if n_skipped:
        print(f"Skipping {n_skipped} entries without a `sections` field.")
    print(f"Evaluating on {len(benchmarks)} benchmarks...")

    KS = [1, 3, 5, 10]
    random.seed(args.seed)

    node_index = section_tree.node_index if section_tree is not None else {}
    number_index = section_tree._number_index if section_tree is not None else {}

    # Only level 2-3 sections are valid retrieval targets and can be gold annotations.
    # Chapters (level 1) are excluded so they never waste a top-k slot in B1 or the random baseline.
    eligible_headings: set[str] = {
        h for h, node in node_index.items() if node.level in (2, 3)
    }
    all_section_headings = list(eligible_headings)
    total_sections = len(all_section_headings)

    # Warn about any gold section that falls outside the eligible pool.
    for bm in benchmarks:
        bad = [s for s in bm.get("sections", []) if s not in eligible_headings]
        if bad:
            print(f"WARNING [{bm['id']}]: gold sections not in eligible pool (level 2-3): {bad}")

    # Lazy-load SentenceTransformer once for B6
    _st_model = None

    per_query = []
    for bm in benchmarks:
        query = bm["question"]
        gold: list[str] = bm["sections"]
        max_k = max(KS)

        row = {"id": bm["id"]}

        # B1: section tree — score sections directly (no chunk roundtrip)
        ranked_b1: list[str] = []
        if section_ret is not None:
            query_keywords = set(extract_query_nodes(query, graph, canonical_lookup))
            section_scores = section_tree.get_section_scores(
                query_keywords,
                query=query,
                heading_alpha=section_ret.heading_alpha,
                inheritance_decay=section_ret.inheritance_decay,
            )
            ranked_b1 = [
                h for h in sorted(section_scores, key=section_scores.__getitem__, reverse=True)
                if h in eligible_headings
            ]
            for k in KS:
                row[f"heading_R@{k}"] = round(_recall_at_k_str(ranked_b1, gold, k), 3)
        else:
            for k in KS:
                row[f"heading_R@{k}"] = None

        # B6: summary FAISS — score sections directly from their summary embeddings
        ranked_b6: list[str] = []
        if summary_index is not None:
            if _st_model is None:
                from sentence_transformers import SentenceTransformer
                _st_model = SentenceTransformer(args.embed_model)

            q_emb = _st_model.encode([query]).astype("float32")
            faiss.normalize_L2(q_emb)
            k_search = min(max_k * 5, summary_index.ntotal)
            similarities, indices = summary_index.search(q_emb, k_search)

            section_scores = {}
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < 0 or sim <= 0:
                    continue
                entry = summary_entries[idx]
                if entry.level == 1:  # skip chapter-level entries
                    continue
                node = number_index.get(entry.section_number)
                if node is None:
                    continue
                heading = node.heading
                section_scores[heading] = max(section_scores.get(heading, 0.0), float(sim))

            ranked_b6 = sorted(section_scores, key=section_scores.__getitem__, reverse=True)
            for k in KS:
                row[f"summary_R@{k}"] = round(_recall_at_k_str(ranked_b6, gold, k), 3)
        else:
            for k in KS:
                row[f"summary_R@{k}"] = None

        # Random baseline
        random_ranked = random.sample(all_section_headings, min(max_k, total_sections))
        for k in KS:
            row[f"random_R@{k}"] = round(_recall_at_k_str(random_ranked, gold, k), 3)

        if args.verbose:
            gold_set = set(gold)
            print(f"\n[{bm['id']}] {query}")
            print(f"  gold ({len(gold)}): {gold}")
            for label, ranked_list in [("heading", ranked_b1), ("summary", ranked_b6)]:
                if ranked_list:
                    top = ranked_list[:max_k]
                    hits = [f"[HIT] {s}" if s in gold_set else s for s in top]
                    print(f"  {label} top-{max_k}: {hits}")
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
