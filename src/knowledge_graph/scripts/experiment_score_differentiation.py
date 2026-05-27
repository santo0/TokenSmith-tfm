"""Score differentiation probe: two cases.

Hypothesis: when the top-k retrieval scores show no clear peak, the system is
either (a) burying the relevant chunks in noise — a retriever quality problem a
stronger embedder might fix — or (b) the answer genuinely spans many chunks and
the flat distribution is expected.

The two benchmark queries used here represent opposite ends of that spectrum:

  aggregation_grouping  2 gold chunks, single section, simple factual question
  aries_atomicity       5 gold chunks, multi-section procedural explanation

For each query the script runs all available retrievers and the RRF ensemble,
prints the top-k score distribution as a bar chart with gold chunks marked, and
reports two differentiation metrics per retriever:
  - NQC   std(top-k) / mean(top-k)            high = peaked, low = flat
  - H     normalized entropy of softmax(top-k) low  = peaked, high = flat

Retrievers included (those whose artifacts are available):
  faiss                dense FAISS cosine similarity
  bm25                 sparse BM25Okapi
  kg_node              BFS expansion from query nodes in the KG
  section_tree         heading-overlap score propagated down the section tree
  section_summary      dense similarity to pre-computed section summaries
  section_filtered_faiss  section-summary pre-filter → FAISS over candidates
  kg_filtered_bm25     BM25 over KG-matched + bridge-expanded query tokens
  ensemble             RRF fusion of all available individual retrievers

Usage:
    python -m src.knowledge_graph.scripts.experiment_score_differentiation \\
        --run-dir data/knowledge_graph/runs/latest \\
        --artifacts-dir index/sections \\
        --embed-model models/embedders/Qwen3-Embedding-4B-Q5_K_M.gguf
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


CASES = [
    {
        "id": "aggregation_grouping",
        "question": "How do aggregation with grouping work?",
        "gold": [148, 149],
    },
    {
        "id": "aries_atomicity",
        "question": "How does the recovery manager use ARIES to ensure atomicity",
        "gold": [1300, 1301, 1302, 1303, 1304],
    },
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def norm_entropy(scores: list[float]) -> float:
    """Normalized Shannon entropy of softmax(scores). 0=peaked, 1=uniform."""
    k = len(scores)
    if k < 2:
        return 0.0
    probs = _softmax(scores)
    h = -sum(p * math.log(p + 1e-12) for p in probs)
    return h / math.log(k)


def nqc(scores: list[float]) -> float:
    """Normalized query commitment: std / mean. High=peaked, low=flat."""
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    if mean == 0:
        return 0.0
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return math.sqrt(variance) / mean


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _bar(value: float, max_value: float, width: int = 28) -> str:
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def print_distribution(
    label: str,
    ranked: list[tuple[int, float]],
    gold: set[int],
    top_k: int,
) -> None:
    top = ranked[:top_k]
    scores = [s for _, s in top]
    if not scores:
        print(f"\n  {label}  — no scores returned")
        return
    max_score = max(scores)

    print(f"\n  {label}")
    print(f"  {'rank':>4}  {'chunk_id':>8}  {'score':>8}  {'':28}  tag")
    print(f"  {'----':>4}  {'--------':>8}  {'--------':>8}  {'':28}  ---")
    for rank, (cid, score) in enumerate(top, 1):
        tag = "[GOLD]" if cid in gold else ""
        bar = _bar(score, max_score)
        print(f"  {rank:>4}  {cid:>8}  {score:>8.4f}  {bar:<28}  {tag}")

    h = norm_entropy(scores)
    n = nqc(scores)
    gold_ranks = [r for r, (cid, _) in enumerate(top, 1) if cid in gold]
    missing = sorted(g for g in gold if g not in {cid for cid, _ in top})

    print(f"\n  H (entropy) = {h:.3f}   NQC = {n:.3f}")
    print(f"  gold ranks  : {gold_ranks if gold_ranks else '—'}   "
          f"missing from top-{top_k}: {missing if missing else '—'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score differentiation probe for aggregation_grouping vs aries_atomicity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--artifacts-dir", default="index/sections")
    parser.add_argument("--index-prefix", default="textbook_index")
    parser.add_argument(
        "--embed-model",
        default="models/embedders/Qwen3-Embedding-4B-Q5_K_M.gguf",
    )
    parser.add_argument(
        "--summary-embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--neighbor-weight", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent.parent

    run_path = Path(args.run_dir)
    if not run_path.is_absolute():
        run_path = root / run_path
    artifacts_path = Path(args.artifacts_dir)
    if not artifacts_path.is_absolute():
        artifacts_path = root / artifacts_path

    # ── Load artifacts ────────────────────────────────────────────────────────
    from src.retriever import FAISSRetriever, BM25Retriever, load_artifacts
    from src.knowledge_graph.io import (
        load_graph_chunks_and_tree, load_summary_data,
        load_canonicalization_data, resolve_run_dir,
    )
    from src.knowledge_graph.query import (
        CanonicalLookup, KGNodeRetriever, SectionTreeRetriever, SectionSummaryRetriever,
    )
    from src.knowledge_graph.experimental_retriever import SectionFilteredFAISSRetriever
    from src.knowledge_graph.kg_filtered_bm25 import KGFilteredBM25Retriever
    from src.ranking.ranker import EnsembleRanker

    print(f"Loading KG from {run_path} ...")
    graph, kg_chunks, section_tree = load_graph_chunks_and_tree(str(run_path))
    resolved = resolve_run_dir(str(run_path))
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = CanonicalLookup(syn_table, can_kw, can_emb) if syn_table else None
    summary_index, summary_entries = load_summary_data(str(run_path))

    print(f"Loading FAISS/BM25 artifacts from {artifacts_path} ...")
    faiss_idx, bm25_idx, raw_chunks, _, metadata = load_artifacts(
        str(artifacts_path), args.index_prefix
    )
    chunk_id_map = [m["chunk_id"] for m in metadata]

    # ── Build retrievers ──────────────────────────────────────────────────────
    retrievers: dict[str, object] = {
        "faiss": FAISSRetriever(faiss_idx, args.embed_model, chunk_id_map=chunk_id_map),
        # "bm25": BM25Retriever(bm25_idx),
        "kg_node": KGNodeRetriever(
            graph, kg_chunks,
            neighbor_weight=args.neighbor_weight,
            num_hops=args.num_hops,
            canonical_lookup=canonical_lookup,
        ),
        # "kg_filtered_bm25": KGFilteredBM25Retriever(
        #     graph, kg_chunks, bm25_idx, canonical_lookup=canonical_lookup
        # ),
    }
    # if section_tree is not None:
    #     retrievers["section_tree"] = SectionTreeRetriever(
    #         section_tree, graph, canonical_lookup=canonical_lookup
    #     )
    # else:
    #     print("  section_tree not available — skipping")

    if summary_index is not None:
        retrievers["section_summary"] = SectionSummaryRetriever(
            summary_index, summary_entries, embed_model=args.summary_embed_model
        )
        retrievers["section_filtered_faiss"] = SectionFilteredFAISSRetriever(
            dense_index=faiss_idx,
            dense_embed_model=args.embed_model,
            chunks=raw_chunks,
            summary_index=summary_index,
            summary_entries=summary_entries,
            summary_embed_model=args.summary_embed_model,
            chunk_id_map=chunk_id_map,
        )
    else:
        print("  section_summary / section_filtered_faiss not available — skipping")

    # ── Run per case ──────────────────────────────────────────────────────────
    sep = "=" * 70
    for case in CASES:
        query = case["question"]
        gold = set(case["gold"])
        print(f"\n{sep}")
        print(f"Query : {case['id']}")
        print(f"Text  : {query}")
        print(f"Gold  : {sorted(gold)}  ({len(gold)} chunk{'s' if len(gold) > 1 else ''})")
        print(sep)

        all_scores: dict[str, dict[int, float]] = {}
        for name, ret in retrievers.items():
            try:
                scores = ret.get_scores(query, args.top_k, raw_chunks)
                all_scores[name] = scores
            except Exception as exc:
                print(f"  [{name}] failed: {exc}")
                all_scores[name] = {}

        for name, scores in all_scores.items():
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            print_distribution(name, ranked, gold, args.top_k)

        # RRF ensemble over all retrievers that returned scores
        available = {n: s for n, s in all_scores.items() if s}
        if len(available) >= 2:
            weights = {n: 1.0 / len(available) for n in available}
            ranker = EnsembleRanker("rrf", weights)
            rrf_ids, rrf_scores = ranker.rank(available)
            rrf_ranked = list(zip(rrf_ids[:args.top_k], rrf_scores[:args.top_k]))
            print_distribution("ensemble (RRF)", rrf_ranked, gold, args.top_k)

    # ── Interpretation guide ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("Interpretation guide")
    print(sep)
    print("  Low H + Low NQC  + gold at rank 1-3  → retriever found it cleanly")
    print("  High H + Low NQC + gold present       → noise case: answer is there but")
    print("                                           retriever can't isolate it;")
    print("                                           stronger embedder / reranker may fix")
    print("  High H + gold absent                  → coverage/scope case: relevant chunks")
    print("                                           not surfacing at all; more context")
    print("                                           or delegation to a larger model needed")
    print()


if __name__ == "__main__":
    main()
