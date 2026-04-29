import argparse
import logging
import os

from dotenv import load_dotenv

from src.knowledge_graph.build import (
    RUNS_DIR,
    PROJECT_ROOT,
    build_extractor,
    create_run_dir,
    setup_input_dir,
    write_config,
    update_latest_symlink,
    load_chunks,
    get_index_paths,
)
from src.knowledge_graph.models import KGPipelineConfig
from src.knowledge_graph.pipeline import build_kg
from src.knowledge_graph.summary_tree import build_summary_index
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.io import load_run_chunks, load_canonicalization_data, build_keyword_index
from src.knowledge_graph.section_tree import build_section_tree, save_section_tree
from src.knowledge_graph.canonicalizer import Canonicalizer
from src.knowledge_graph.linkers import CooccurrenceLinker


logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build the knowledge graph.")
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "config", "config.yaml"),
        help="Path to project config YAML (default: config/config.yaml)",
    )
    args = parser.parse_args()

    cfg = KGPipelineConfig.from_yaml(args.config)
    logger.info("Loaded config from %s", args.config)

    extractor, extractor_config = build_extractor(cfg)
    logger.info("Using extractor: %s", extractor_config["class"])

    chunks_pkl, meta_pkl = get_index_paths(cfg.partial)
    logger.info("Using chunks: %s", chunks_pkl)

    run_dir = create_run_dir()
    logger.info("Run directory: %s", run_dir)

    extractions_path = (
        extractor_config.get("input_path")
        if extractor_config["class"] == "JsonExtractor"
        else None
    )
    setup_input_dir(run_dir, extractions_path, chunks_pkl=chunks_pkl, meta_pkl=meta_pkl)
    write_config(run_dir, cfg, extractor_config, extractions_path,
                 chunks_pkl=chunks_pkl, meta_pkl=meta_pkl)

    chapter_filter = f"Chapter {cfg.chapter} " if cfg.chapter else None
    exclude_chapters = [f"Chapter {c} " for c in cfg.exclude_chapters]

    c = cfg.canonicalization
    canonicalizer = Canonicalizer(
        embedding_model=cfg.embed_model,
        corpus_description=cfg.corpus_description,
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        llm_model=c.llm_model,
        similarity_threshold=c.similarity_threshold,
        max_group_size=c.max_group_size,
        batch_size=c.batch_size,
    )

    linker = CooccurrenceLinker(min_cooccurrence=cfg.min_cooccurrence)

    chunks = load_chunks(
        chunks_pkl,
        meta_pkl,
        chapter_filter=chapter_filter,
        exclude_chapters=exclude_chapters,
    )
    logger.info("Loaded %d chunks", len(chunks))
    graph = build_kg(
        output_dir=run_dir,
        chunks=chunks,
        extractor=extractor,
        linker=linker,
        canonicalizer=canonicalizer,
    )

    logger.info("Building keyword FAISS index...")
    _, _canon_kws, _canon_embs = load_canonicalization_data(run_dir)
    if _canon_embs is not None:
        build_keyword_index(_canon_embs, run_dir)
        logger.info("Keyword index saved.")
    else:
        logger.warning("Canonicalization data missing; keyword index not built.")

    logger.info("Building section tree...")
    tree = build_section_tree(chunks, graph)
    tree_path = save_section_tree(tree, run_dir)
    level_counts: dict[int, int] = {}
    for node in tree.node_index.values():
        level_counts[node.level] = level_counts.get(node.level, 0) + 1
    level_labels = {1: "chapters", 2: "sections", 3: "subsections"}
    for level, count in sorted(level_counts.items()):
        label = level_labels.get(level, f"level-{level} nodes")
        logger.info("  %4d %s", count, label)
    logger.info("  Saved: %s", tree_path)

    st = cfg.summary_tree
    logger.info(
        "Building summary index (model=%s, chunk_window=%d)...",
        st.summary_model,
        st.chunk_window,
    )
    chunk_texts = load_run_chunks(os.path.join(run_dir, "chunks.json"))
    client = OpenRouterClient(os.environ.get("OPENROUTER_API_KEY", ""), retries=2)
    build_summary_index(
        client=client,
        summary_model=st.summary_model,
        section_tree=tree,
        chunks=chunk_texts,
        embed_model=cfg.embed_model,
        chunk_window=st.chunk_window,
        run_dir=run_dir,
    )
    logger.info("Summary index saved to %s", run_dir)

    update_latest_symlink(run_dir)
    logger.info("Updated: %s -> %s", os.path.join(RUNS_DIR, "latest"), run_dir)


if __name__ == "__main__":
    load_dotenv()
    main()
