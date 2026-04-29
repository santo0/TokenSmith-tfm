"""Debug tool — run a retriever and ask the LLM to grade the retrieved chunks.

Usage:
    python -m src.knowledge_graph.scripts.debug_chunk_inspector \\
        --query "What is a B+ tree?" \\
        --retriever kg \\
        --top-k 5

Retriever choices: faiss, bm25, kg, section_tree, section_summary, hybrid, section_kg
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

RELEVANCE_PROMPT = """\
You are evaluating a retrieval system for a question-answering application.

Question: {query}

Retrieved passages:
{passages}

For each passage decide how relevant it is to the question, then write a concise \
answer using only the relevant ones.

Return ONLY a JSON object in this exact shape:
{{
  "grades": [
    {{"chunk_id": <int>, "score": <0|1|2>, "reason": "<one sentence>"}}
  ],
  "relevant_ids": [<chunk_id>, ...],
  "answer": "<answer text>"
}}

Scoring: 0 = irrelevant, 1 = partially relevant, 2 = directly answers the question."""

SCORE_LABEL = {0: "irrelevant", 1: "partial  ", 2: "RELEVANT "}


def _build_retriever(name: str, args: argparse.Namespace, root: Path):
    from src.knowledge_graph.io import (
        load_graph_and_chunks, load_graph_chunks_and_tree,
        load_summary_data, load_canonicalization_data, resolve_run_dir,
    )
    from src.knowledge_graph.query import (
        CanonicalLookup, KGNodeRetriever, SectionTreeRetriever, SectionSummaryRetriever,
    )
    from src.retriever import FAISSRetriever, BM25Retriever, load_artifacts

    run_path = Path(args.run_dir)
    if not run_path.is_absolute():
        run_path = root / run_path
    artifacts_path = Path(args.artifacts_dir)
    if not artifacts_path.is_absolute():
        artifacts_path = root / artifacts_path

    # ── Index-based retrievers (faiss, bm25, hybrid) ──────────────────────
    if name in ("faiss", "bm25", "hybrid"):
        import faiss as _faiss
        _, bm25_idx, raw_chunks, _, _ = load_artifacts(str(artifacts_path), args.index_prefix)
        faiss_idx = _faiss.read_index(str(artifacts_path / f"{args.index_prefix}.faiss"))
        chunks_map: dict[int, str] = {i: t for i, t in enumerate(raw_chunks)}

        if name == "faiss":
            retriever = FAISSRetriever(faiss_idx, args.embed_model)
        elif name == "bm25":
            retriever = BM25Retriever(bm25_idx)
        else:
            from src.knowledge_graph.experimental_retriever import HybridRetriever
            resolved = resolve_run_dir(str(run_path))
            syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
            canonical_lookup = CanonicalLookup(syn_table, can_kw, can_emb) if syn_table else None
            retriever = HybridRetriever.from_config(
                faiss_index=faiss_idx,
                chunks=raw_chunks,
                dense_embed_model=args.embed_model,
                run_dir=str(run_path),
                summary_embed_model=args.summary_embed_model,
                canonical_lookup=canonical_lookup,
            )
        return retriever, raw_chunks, chunks_map

    # ── KG-based retrievers ───────────────────────────────────────────────
    if name in ("section_tree", "section_kg"):
        graph, kg_chunks, section_tree = load_graph_chunks_and_tree(str(run_path))
    else:
        graph, kg_chunks = load_graph_and_chunks(str(run_path))
        section_tree = None

    resolved = resolve_run_dir(str(run_path))
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = CanonicalLookup(syn_table, can_kw, can_emb) if syn_table else None
    chunks_list = list(kg_chunks.values())

    if name == "kg":
        retriever = KGNodeRetriever(
            graph, kg_chunks,
            neighbor_weight=args.neighbor_weight,
            num_hops=args.num_hops,
            canonical_lookup=canonical_lookup,
        )
    elif name == "section_tree":
        if section_tree is None:
            raise SystemExit("No section_tree.json found in run dir.")
        retriever = SectionTreeRetriever(section_tree, graph, canonical_lookup=canonical_lookup)
    elif name == "section_summary":
        summary_index, summary_entries = load_summary_data(str(run_path))
        if summary_index is None:
            raise SystemExit("No summary index found in run dir.")
        retriever = SectionSummaryRetriever(
            summary_index, summary_entries, embed_model=args.embed_model
        )
    elif name == "section_kg":
        summary_index, summary_entries = load_summary_data(str(run_path))
        if summary_index is None:
            raise SystemExit("No summary index found in run dir.")
        if section_tree is None:
            raise SystemExit("No section_tree.json found in run dir.")
        from src.knowledge_graph.experimental_retriever import SectionKGRetriever
        retriever = SectionKGRetriever(
            summary_index=summary_index,
            summary_entries=summary_entries,
            summary_embed_model=args.summary_embed_model,
            section_tree=section_tree,
            graph=graph,
            kg_chunks=kg_chunks,
            chunks=chunks_list,
            canonical_lookup=canonical_lookup,
        )
    else:
        raise SystemExit(f"Unknown retriever: {name}")

    return retriever, chunks_list, kg_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a retriever on a query and grade results with an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--retriever",
        required=True,
        choices=["faiss", "bm25", "kg", "section_tree", "section_summary", "hybrid", "section_kg"],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--artifacts-dir", default="index/sections")
    parser.add_argument("--index-prefix", default="textbook_index")
    parser.add_argument("--embed-model", default="models/embedders/Qwen3-Embedding-4B-Q5_K_M.gguf")
    parser.add_argument(
        "--summary-embed-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--neighbor-weight", type=float, default=0.5)
    parser.add_argument("--llm-model", default="google/gemini-3-flash-preview")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM grading")
    args = parser.parse_args()

    load_dotenv()
    root = Path(__file__).parent.parent.parent.parent

    # ── Build retriever and run ──────────────────────────────────────────
    print(f"\nBuilding '{args.retriever}' retriever...")
    retriever, chunks_input, chunks_map = _build_retriever(args.retriever, args, root)

    print(f"Retrieving top-{args.top_k} chunks for: {args.query!r}")
    scores: dict[int, float] = retriever.get_scores(args.query, args.top_k, chunks_input)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: args.top_k]

    if not ranked:
        print("No chunks retrieved — check that the query matches the index vocabulary.")
        return

    # ── Print retrieved chunks ───────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"QUERY : {args.query}")
    print(f"RETRIEVER : {args.retriever}   TOP-K : {args.top_k}")
    print(sep)

    for rank, (cid, score) in enumerate(ranked, 1):
        text = chunks_map.get(cid, "<chunk not found>")
        print(f"\n[{rank}] chunk_id={cid}  score={score:.4f}")
        print("-" * 40)
        print(text)

    # ── LLM grading ─────────────────────────────────────────────────────
    if args.no_llm:
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("\nOPENROUTER_API_KEY not set — skipping LLM grading.")
        return

    from src.knowledge_graph.openrouter_client import OpenRouterClient
    client = OpenRouterClient(api_key, retries=2)

    passages_lines: list[str] = []
    for rank, (cid, score) in enumerate(ranked, 1):
        text = chunks_map.get(cid, "<chunk not found>")
        passages_lines.append(f"(chunk_id={cid}, score={score:.4f})\n{text}")
    passages_block = "\n\n".join(passages_lines)

    prompt = RELEVANCE_PROMPT.format(query=args.query, passages=passages_block)

    print(f"\n{sep}")
    print(f"Calling {args.llm_model} for relevance grading...")
    print(sep)

    raw = client.chat(
        args.llm_model,
        [{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print("LLM returned non-JSON:\n")
        print(raw)
        return

    print("\nGrades:")
    for g in result.get("grades", []):
        cid = g.get("chunk_id", "?")
        score = g.get("score", "?")
        reason = g.get("reason", "")
        label = SCORE_LABEL.get(score, f"score={score}")
        print(f"  [{label}] chunk_id={cid!s:>5}  {reason}")

    relevant = result.get("relevant_ids", [])
    print(f"\nRelevant chunk IDs: {relevant}")

    print(f"\n{sep}")
    print("LLM answer:")
    print(sep)
    print(result.get("answer", "<no answer returned>"))
    print()


if __name__ == "__main__":
    main()
