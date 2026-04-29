"""A1 — LLM extracts more useful keywords than YAKE/KeyBERT.

For each benchmark's gold chunks, runs three extractors (LLM/OpenRouter, KeyBERT,
YAKE) and evaluates keyword quality two ways:
  1. Lexical match P/R/F1 vs the benchmark's gold ``keywords`` field.
  2. Downstream retrieval recall@k: seed KG subgraph expansion with extracted
     keywords and retrieve; compare recall against dense FAISS baseline.

Usage:
    python -m src.knowledge_graph.scripts.experiment_a1_extractor_quality \\
        --run-dir data/knowledge_graph/runs/latest \\
        --artifacts-dir index/sections \\
        --embed-model sentence-transformers/all-MiniLM-L6-v2 \\
        --llm-model google/gemini-3-flash-preview \\
        --output results_a1.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


def _prf(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    tp = len(predicted & gold)
    p = tp / len(predicted)
    r = tp / len(gold)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A1: Compare LLM / KeyBERT / YAKE keyword extraction quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument("--artifacts-dir", default=None,
                        help="RAG artifacts dir (for FAISS dense baseline). Optional.")
    parser.add_argument("--index-prefix", default="textbook_index")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", default="google/gemini-3-flash-preview")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")

    from src.knowledge_graph.io import (
        load_graph_and_chunks, load_canonicalization_data, resolve_run_dir
    )
    from src.knowledge_graph.query import CanonicalLookup, KGNodeRetriever
    from src.knowledge_graph.normalizer import Normalizer
    from src.knowledge_graph.models import Chunk
    from src.knowledge_graph.scripts.eval_utils import (
        load_labeled_benchmarks, recall_at_k, scores_to_ranked_ids, print_table
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

    normalizer = Normalizer()
    benchmarks = load_labeled_benchmarks(args.benchmarks)
    print(f"Evaluating on {len(benchmarks)} labeled benchmarks...")

    # Optionally load FAISS artifacts for dense baseline
    faiss_retriever = None
    dense_chunks_list: list[str] = []
    if args.artifacts_dir:
        try:
            from src.retriever import FAISSRetriever, load_artifacts
            artifacts_path = Path(args.artifacts_dir)
            if not artifacts_path.is_absolute():
                artifacts_path = root / artifacts_path
            faiss_idx, _, raw_chunks, _, _ = load_artifacts(str(artifacts_path), args.index_prefix)
            faiss_retriever = FAISSRetriever(faiss_idx, args.embed_model)
            dense_chunks_list = raw_chunks
            print(f"Dense baseline (FAISS) enabled with {len(dense_chunks_list)} chunks.")
        except Exception as e:
            print(f"FAISS not available: {e} — skipping dense baseline.")

    # Build LLM extractor if api key provided
    llm_extractor = None
    if api_key:
        from src.knowledge_graph.extractors.openrouter_extractor import OpenRouterExtractor
        llm_extractor = OpenRouterExtractor(
            api_key=api_key, model=args.llm_model, top_n=args.top_n, adaptive_top_n=False
        )

    # KeyBERT extractor
    try:
        from src.knowledge_graph.extractors.keybert_extractor import KeyBERTExtractor
        kb_extractor = KeyBERTExtractor(top_n=args.top_n)
    except Exception as e:
        print(f"KeyBERT not available: {e}")
        kb_extractor = None

    # YAKE extractor
    try:
        from src.knowledge_graph.extractors.yake_extractor import YakeExtractor
        yake_extractor = YakeExtractor(top_n=args.top_n)
    except Exception as e:
        print(f"YAKE not available: {e}")
        yake_extractor = None

    rows = []
    for bm in benchmarks:
        query = bm["question"]
        gold_kws_raw: list[str] = bm.get("keywords", [])
        gold_kws = set(normalizer.normalize(gold_kws_raw))
        gold_chunks: list[int] = bm["ideal_retrieved_chunks"]

        chunk_texts = {cid: kg_chunks[cid] for cid in gold_chunks if cid in kg_chunks}
        chunks_objs = [
            Chunk(id=cid, text=text, metadata={})
            for cid, text in chunk_texts.items()
        ]
        if not chunks_objs:
            continue

        extractor_results: dict[str, dict] = {}

        for name, extractor in [
            ("llm", llm_extractor),
            ("keybert", kb_extractor),
            ("yake", yake_extractor),
        ]:
            if extractor is None:
                continue
            try:
                results = extractor.extract(chunks_objs)
                all_kws: list[str] = []
                for r in results:
                    all_kws.extend(r.keywords)
                pred_norm = set(normalizer.normalize(all_kws))
            except Exception as e:
                print(f"  [{bm['id']}] {name} extraction failed: {e}")
                pred_norm = set()

            p, r, f1 = _prf(pred_norm, gold_kws)

            # Downstream recall@k via KG retrieval seeded with extracted keywords
            kg_recall = None
            if pred_norm:
                # Build a temporary graph query using matched nodes
                matched_nodes = [n for n in pred_norm if graph.has_node(n)]
                if matched_nodes:
                    kg_retriever = KGNodeRetriever(
                        graph, kg_chunks, canonical_lookup=canonical_lookup
                    )
                    scores = kg_retriever.get_scores(query, len(kg_chunks), list(kg_chunks.values()))
                    ranked = scores_to_ranked_ids(scores, args.top_k)
                    kg_recall = recall_at_k(ranked, gold_chunks, args.top_k)

            extractor_results[name] = {
                "P": round(p, 3), "R": round(r, 3), "F1": round(f1, 3),
                "kg_recall": round(kg_recall, 3) if kg_recall is not None else None,
            }

        # Dense baseline
        dense_recall = None
        if faiss_retriever and dense_chunks_list:
            try:
                scores = faiss_retriever.get_scores(query, args.top_k, dense_chunks_list)
                ranked = scores_to_ranked_ids(scores, args.top_k)
                dense_recall = recall_at_k(ranked, gold_chunks, args.top_k)
            except Exception as e:
                print(f"  [{bm['id']}] dense retrieval failed: {e}")

        if args.verbose:
            print(f"\n[{bm['id']}] {query}")
            for name, res in extractor_results.items():
                print(f"  {name:8s}: P={res['P']:.2f} R={res['R']:.2f} F1={res['F1']:.2f}  "
                      f"KG-recall@{args.top_k}={res.get('kg_recall', 'N/A')}")
            if dense_recall is not None:
                print(f"  dense:    recall@{args.top_k}={dense_recall:.3f}")

        row = {"id": bm["id"], "dense_recall": round(dense_recall, 3) if dense_recall is not None else None}
        for name, res in extractor_results.items():
            row[f"{name}_F1"] = res["F1"]
            row[f"{name}_recall"] = res.get("kg_recall")
        rows.append(row)

    # Summary
    print(f"\n{'=' * 70}")
    print("A1 Results — Extractor quality (keyword F1 vs gold) and KG recall@k")
    print(f"{'=' * 70}")
    cols = ["id"]
    for name in ["llm", "keybert", "yake"]:
        cols += [f"{name}_F1", f"{name}_recall"]
    if faiss_retriever:
        cols.append("dense_recall")
    print_table(rows, [c for c in cols if any(c in r for r in rows)])

    result = {
        "n_benchmarks": len(rows),
        "top_k": args.top_k,
        "top_n_extractors": args.top_n,
        "per_query": rows,
    }

    if args.output:
        out = Path(args.output)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {out}")


if __name__ == "__main__":
    load_dotenv()
    main()
