"""
Rerun the canonicalization step (and everything downstream) on an existing run.

Reads raw extractions from input/extractions.json if available; falls back to
reconstructing them from the graph for runs that predate checkpoint persistence.
Does NOT re-extract keywords or rebuild the summary index.

Usage:
    python -m src.knowledge_graph.scripts.rerun_canonicalization
    python -m src.knowledge_graph.scripts.rerun_canonicalization --run-dir data/knowledge_graph/runs/2026-04-29_16-00-17
    python -m src.knowledge_graph.scripts.rerun_canonicalization --config config/config.yaml
"""
import argparse
import json
import logging
import os

import numpy as np
from dotenv import load_dotenv

from src.knowledge_graph.build import RUNS_DIR, PROJECT_ROOT
from src.knowledge_graph.canonicalizer import Canonicalizer
from src.knowledge_graph.io import (
    resolve_run_dir,
    load_run_chunks,
    load_graph,
    build_keyword_index,
)
from src.knowledge_graph.linkers import CooccurrenceLinker
from src.knowledge_graph.models import ExtractionResult, KGPipelineConfig
from src.knowledge_graph.section_tree import build_section_tree, save_section_tree

logger = logging.getLogger(__name__)


def _load_extractions(run_dir: str, graph) -> list[ExtractionResult]:
    path = os.path.join(run_dir, "input", "extractions.json")
    if os.path.isfile(path):
        logger.info("Loading raw extractions from %s", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ExtractionResult(chunk_id=e["chunk_id"], keywords=e["keywords"]) for e in data]

    logger.warning(
        "input/extractions.json not found — reconstructing from graph (canonical keywords only)"
    )
    chunk_to_kws: dict[int, list[str]] = {}
    for node, data in graph.nodes(data=True):
        for cid in data.get("chunk_ids", []):
            chunk_to_kws.setdefault(cid, []).append(node)
    return [
        ExtractionResult(chunk_id=cid, keywords=kws)
        for cid, kws in sorted(chunk_to_kws.items())
    ]


def _save_extractions(extractions: list[ExtractionResult], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"chunk_id": er.chunk_id, "keywords": er.keywords} for er in extractions], f)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    load_dotenv()

    parser = argparse.ArgumentParser(description="Rerun canonicalization on an existing run.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Path to a run directory or runs/ parent. Defaults to latest run.",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "config", "config.yaml"),
        help="Project config YAML (default: config/config.yaml)",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir or RUNS_DIR)
    logger.info("Run dir: %s", run_dir)

    # ── Load existing artifacts ───────────────────────────────────────────────

    graph = load_graph(os.path.join(run_dir, "graph.json"))
    chunks_text = load_run_chunks(os.path.join(run_dir, "chunks.json"))

    run_config_path = os.path.join(run_dir, "config.json")
    with open(run_config_path, "r", encoding="utf-8") as f:
        run_config = json.load(f)

    extractions = _load_extractions(run_dir, graph)
    logger.info("Loaded %d extractions", len(extractions))

    # ── Build pipeline objects ────────────────────────────────────────────────

    cfg = KGPipelineConfig.from_yaml(args.config)
    c = cfg.canonicalization

    canonicalizer = Canonicalizer(
        corpus_description=cfg.corpus_description,
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        embedding_model=cfg.embed_model,
        llm_model=c.llm_model,
        similarity_threshold=c.similarity_threshold,
        max_group_size=c.max_group_size,
        batch_size=c.batch_size,
    )

    min_cooccurrence = run_config.get("linker", {}).get("min_cooccurrence", 0)
    linker = CooccurrenceLinker(min_cooccurrence=min_cooccurrence)

    # ── Canonicalize ─────────────────────────────────────────────────────────

    logger.info("Canonicalizing...")
    updated_extractions, canon_result = canonicalizer.canonicalize(extractions)
    s = canon_result.stats
    logger.info(
        "  %d → %d keywords, %d merges, %d LLM calls",
        s["keywords_after_stage1"], s["canonical_keywords_final"],
        s["merges_performed"], s["llm_calls"],
    )

    _save_extractions(updated_extractions, os.path.join(run_dir, "canonical_extractions.json"))
    logger.info("Saved canonical_extractions.json")

    # ── Re-link ───────────────────────────────────────────────────────────────

    logger.info("Linking...")
    graph = linker.link(updated_extractions)
    logger.info("  %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

    import networkx as nx
    graph_data = nx.node_link_data(graph)
    with open(os.path.join(run_dir, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(run_dir, "synonym_table.json"), "w", encoding="utf-8") as f:
        json.dump(canon_result.synonym_table, f, indent=2, ensure_ascii=False)

    with open(os.path.join(run_dir, "canonical_keywords.json"), "w", encoding="utf-8") as f:
        json.dump(canon_result.canonical_keywords, f, indent=2, ensure_ascii=False)

    np.save(os.path.join(run_dir, "canonical_embeddings.npy"), canon_result.canonical_embeddings)
    logger.info("Saved graph, synonym_table, canonical_keywords, canonical_embeddings")

    # ── Rebuild keyword index ────────────────────────────────────────────────

    build_keyword_index(canon_result.canonical_embeddings, run_dir)
    logger.info("Rebuilt keyword_index.faiss")

    # ── Rebuild section tree ─────────────────────────────────────────────────

    from src.knowledge_graph.models import Chunk
    import pickle

    chunks_pkl = os.path.join(run_dir, "input", "chunks.pkl")
    meta_pkl = os.path.join(run_dir, "input", "meta.pkl")
    with open(chunks_pkl, "rb") as f:
        raw_chunks = pickle.load(f)
    with open(meta_pkl, "rb") as f:
        meta = pickle.load(f)

    chunks = [
        Chunk(id=m["chunk_id"], text=text, metadata=m)
        for text, m in zip(raw_chunks, meta)
    ]

    tree = build_section_tree(chunks, graph)
    tree_path = save_section_tree(tree, run_dir)
    logger.info("Rebuilt section_tree → %s", tree_path)

    logger.info("Done. Summary index not modified.")


if __name__ == "__main__":
    main()
