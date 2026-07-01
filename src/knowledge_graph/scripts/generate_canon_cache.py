import argparse
import json
import logging
import os

from dotenv import load_dotenv

from src.knowledge_graph.build import (
    PROJECT_ROOT,
    get_index_paths,
    get_latest_extractions_path,
    load_chunks,
)
from src.knowledge_graph.canonicalizer import Canonicalizer
from src.knowledge_graph.extractors import JsonExtractor
from src.knowledge_graph.scripts.run_kg_pipeline import KGPipelineConfig

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = os.path.join(PROJECT_ROOT, "debug", "canonicalization_cache.json")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run LLM canonicalization once and save the full result to a cache file."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "config", "config.yaml"),
        help="Path to project config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_CACHE_PATH,
        help=f"Path to write the cache JSON (default: {DEFAULT_CACHE_PATH})",
    )
    parser.add_argument(
        "--extractions",
        default=None,
        metavar="PATH",
        help="Path to a keyword extractions JSON (default: extractions/latest.json)",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        default=False,
        help="Use the partial index (index/partial_sections/) instead of the full index",
    )
    args = parser.parse_args()

    cfg = KGPipelineConfig.from_yaml(args.config)
    logger.info("Loaded config from %s", args.config)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable must be set.")

    chunks_pkl, meta_pkl = get_index_paths(args.partial)
    logger.info("Loading chunks from:\n  %s\n  %s", chunks_pkl, meta_pkl)
    chunks = load_chunks(chunks_pkl, meta_pkl)
    logger.info("Loaded %d chunks", len(chunks))

    extractions_path = args.extractions or get_latest_extractions_path()
    extractor = JsonExtractor(input_path=extractions_path)
    extractions = extractor.extract(chunks)
    logger.info("Extracted %d results", len(extractions))

    c = cfg.canonicalization
    canonicalizer = Canonicalizer(
        embedding_model=c.embed_model,
        corpus_description=cfg.corpus_description,
        api_key=api_key,
        llm_model=c.llm_model,
        similarity_threshold=c.similarity_threshold,
        max_group_size=c.max_group_size,
        batch_size=c.batch_size,
    )

    updated_extractions, canon_result = canonicalizer.canonicalize(extractions)

    cache = {
        "updated_extractions": [
            {"chunk_id": e.chunk_id, "keywords": e.keywords}
            for e in updated_extractions
        ],
        "synonym_table": canon_result.synonym_table,
        "canonical_keywords": canon_result.canonical_keywords,
        "canonical_embeddings": canon_result.canonical_embeddings.tolist(),
        "stats": canon_result.stats,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    logger.info("Saved cache to %s", args.output)


if __name__ == "__main__":
    load_dotenv()
    main()
