"""Build a semantic knowledge graph from an existing cooccurrence KG run.

Usage:
    python -m src.knowledge_graph.scripts.run_semantic_kg_pipeline \\
        [--config config/config.yaml] \\
        [--run_dir data/knowledge_graph/runs/2026-04-13_10-00-00] \\
        [--chapters 1 3 5]

The script reads an existing run directory (or the ``latest`` symlink) and a
pre-built canonicalization cache, then runs :class:`SemanticLinker` over the
chunks and persists ``semantic_graph.json`` + ``semantic_triples.json`` into
the same run directory.

Use ``--chapters`` to restrict extraction to a subset of chapters (e.g.
``--chapters 1 2 3``).  By default all chapters are processed.
"""

import argparse
import json
import logging
import os

from dotenv import load_dotenv

from src.knowledge_graph.build import (
    CHUNKS_PKL,
    JSON_KW_PATH,
    META_PKL,
    OUTPUT_DIR,
    PROJECT_ROOT,
    load_chunks,
)
from src.knowledge_graph.canonicalizer import MockCanonicalizer
from src.knowledge_graph.extractors import JsonExtractor
from src.knowledge_graph.io import resolve_run_dir
from src.knowledge_graph.linkers import SemanticLinker
from src.knowledge_graph.models import KGPipelineConfig
from src.knowledge_graph.persisters import NetworkxJsonPersister
from src.knowledge_graph.pipeline import Pipeline

logger = logging.getLogger(__name__)

DEFAULT_CANON_CACHE = os.path.join(
    PROJECT_ROOT, "debug", "canonicalization_cache.json")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build the semantic knowledge graph.",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "config", "config.yaml"),
        help="Path to project config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--run_dir",
        default=os.path.join(OUTPUT_DIR, "runs"),
        help=(
            "Path to an existing KG run directory or runs/ parent with a "
            "'latest' symlink. The semantic graph is written into this directory. "
            "(default: data/knowledge_graph/runs)"
        ),
    )
    parser.add_argument(
        "--chapters",
        nargs="+",
        metavar="N",
        default=None,
        help=(
            "Space-separated list of chapter numbers to process "
            "(e.g. --chapters 1 2 3). Default: all chapters."
        ),
    )
    args = parser.parse_args()

    cfg = KGPipelineConfig.from_yaml(args.config)
    logger.info("Loaded config from %s", args.config)

    # Resolve the concrete run directory
    try:
        run_dir = resolve_run_dir(args.run_dir)
    except FileNotFoundError:
        logger.error(
            "Could not resolve a valid run directory from %r. "
            "Run the cooccurrence pipeline first.",
            args.run_dir,
        )
        raise

    logger.info("Writing semantic graph to run directory: %s", run_dir)

    # Build chapter prefix list if --chapters was supplied
    chapter_prefixes: list[str] | None = None
    if args.chapters:
        chapter_prefixes = [f"Chapter {n} " for n in args.chapters]
        logger.info(
            "Chapter filter active — will process chapters: %s",
            ", ".join(args.chapters),
        )

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY environment variable must be set."
        )

    # Load all chunks (chapter filtering is applied inside the pipeline)
    logger.info("Loading chunks from:\n  %s\n  %s", CHUNKS_PKL, META_PKL)
    chunks = load_chunks(CHUNKS_PKL, META_PKL)
    logger.info("Loaded %d chunks", len(chunks))

    sl_cfg = cfg.semantic_linker
    linker = SemanticLinker(
        chunks={c.id: c.text for c in chunks},
        api_key=api_key,
        llm_model=sl_cfg.llm_model,
        corpus_description=cfg.corpus_description,
        allowed_relations=sl_cfg.allowed_relations or None,
        batch_size=sl_cfg.batch_size,
        min_occurrence=sl_cfg.min_occurrence,
    )

    # c = cfg.canonicalization
    # canonicalizer = Canonicalizer(
    #     embedding_model=c.embed_model,
    #     corpus_description=cfg.corpus_description,
    #     api_key=api_key,
    #     llm_model=c.llm_model,
    #     similarity_threshold=c.similarity_threshold,
    #     max_group_size=c.max_group_size,
    #     batch_size=c.batch_size,
    # )
    canonicalizer = MockCanonicalizer(DEFAULT_CANON_CACHE)

    pipeline = Pipeline(
        extractor=JsonExtractor(input_path=JSON_KW_PATH),
        linker=linker,
        persister=NetworkxJsonPersister(graph_filename="semantic_graph.json"),
        canonicalizer=canonicalizer,
        chapter_filter=chapter_prefixes,
    )

    logger.info("Running semantic linker pipeline...")
    graph = pipeline.run(chunks=chunks, output_dir=run_dir)

    triples_path = os.path.join(run_dir, "semantic_triples.json")
    triples: list[dict] = [
        {
            "subject": u,
            "relation": data.get("relation", ""),
            "object": v,
            "weight": data.get("weight", 1),
            "chunk_ids": data.get("chunk_ids", []),
        }
        for u, v, data in graph.edges(data=True)
    ]
    with open(triples_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d triples to %s", len(triples), triples_path)

    logger.info(
        "Semantic graph: %d nodes, %d edges → %s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        os.path.join(run_dir, "semantic_graph.json"),
    )
    logger.info("Done.")


if __name__ == "__main__":
    load_dotenv()
    main()
