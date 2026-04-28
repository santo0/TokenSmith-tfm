"""
Evaluate whether embedding-based section filtering is effective.

Three strategies are compared for each benchmark query:

  - titles:    embed chapter/subsection titles from sections.json
               (all-MiniLM-L6-v2, IndexFlatIP)
  - summaries: use the pre-built summary_index.faiss from the KG run directory
               (all-MiniLM-L6-v2, IndexFlatIP)
  - vector:    use the pre-built chunk index from the artifact directory
               (Qwen3 GGUF via CachedEmbedder, IndexFlatL2)
               Results are shown at chunk level with their section path.

Titles index is cached as titles_index.faiss / titles_meta.json inside the
run directory so embeddings are computed only once.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAGConfig  # noqa: E402
from src.embedder import CachedEmbedder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TITLES_INDEX_FILE = "titles_index.faiss"
TITLES_META_FILE = "titles_meta.json"

ARTIFACT_DIR = PROJECT_ROOT / "index" / "sections"
ARTIFACT_PREFIX = "textbook_index"


# ── paths ─────────────────────────────────────────────────────────────────────

def _latest_run_dir() -> Path:
    runs = sorted((PROJECT_ROOT / "data" / "knowledge_graph" / "runs").iterdir())
    if not runs:
        raise FileNotFoundError("No KG run directories found")
    return runs[-1]


# ── data loading ──────────────────────────────────────────────────────────────

def load_benchmarks() -> list[dict]:
    with open(PROJECT_ROOT / "tests" / "benchmarks.yaml") as f:
        return yaml.safe_load(f)["benchmarks"]


def _build_title_entries() -> list[dict]:
    """Return one entry per chapter and per subsection from sections.json."""
    with open(PROJECT_ROOT / "sections.json") as f:
        data = json.load(f)

    entries: list[dict] = []
    for part in data["parts"]:
        for ch in part["chapters"]:
            num = str(ch["chapter_number"])
            title = ch["chapter_title"]
            entries.append({
                "section_number": num,
                "level": 1,
                "text": f"Chapter {num}: {title}",
            })
            for sub in ch.get("subsections", []):
                sub_num = sub.split(" ", 1)[0]
                entries.append({
                    "section_number": sub_num,
                    "level": 2,
                    "text": sub,
                })
    return entries


# ── FAISS index helpers (MiniLM / IP) ─────────────────────────────────────────

def _build_and_save_titles_index(
    run_dir: Path,
    model: SentenceTransformer,
) -> tuple[faiss.Index, list[dict]]:
    logger.info("Building titles index (will be cached in %s)…", run_dir)
    entries = _build_title_entries()
    texts = [e["text"] for e in entries]
    embeddings = model.encode(texts, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(run_dir / TITLES_INDEX_FILE))
    with open(run_dir / TITLES_META_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    logger.info("Titles index saved (%d entries, dim=%d)", len(entries), embeddings.shape[1])
    return index, entries


def load_or_build_titles_index(
    run_dir: Path,
    model: SentenceTransformer,
) -> tuple[faiss.Index, list[dict]]:
    index_path = run_dir / TITLES_INDEX_FILE
    meta_path = run_dir / TITLES_META_FILE

    if index_path.exists() and meta_path.exists():
        logger.info("Loading cached titles index from %s", run_dir)
        index = faiss.read_index(str(index_path))
        with open(meta_path, encoding="utf-8") as f:
            entries = json.load(f)
        return index, entries

    return _build_and_save_titles_index(run_dir, model)


def load_summary_index(run_dir: Path) -> tuple[faiss.Index, list[dict]]:
    index_path = run_dir / "summary_index.faiss"
    meta_path = run_dir / "summary_meta.json"

    if not index_path.exists():
        raise FileNotFoundError(f"summary_index.faiss not found in {run_dir}")

    index = faiss.read_index(str(index_path))
    with open(meta_path, encoding="utf-8") as f:
        entries = json.load(f)
    return index, entries


# ── chunk FAISS index (Qwen3 / L2) ───────────────────────────────────────────

def load_chunk_index(
    artifact_dir: Path = ARTIFACT_DIR,
    prefix: str = ARTIFACT_PREFIX,
) -> tuple[faiss.Index, list[str], list[dict]]:
    index = faiss.read_index(str(artifact_dir / f"{prefix}.faiss"))
    chunks = pickle.load(open(artifact_dir / f"{prefix}_chunks.pkl", "rb"))
    meta = pickle.load(open(artifact_dir / f"{prefix}_meta.pkl", "rb"))
    return index, chunks, meta


# ── search helpers ────────────────────────────────────────────────────────────

def search_ip(
    query_embedding: np.ndarray,
    index: faiss.Index,
    entries: list[dict],
    top_k: int,
    text_field: str,
) -> list[dict]:
    """Search an IndexFlatIP index (L2-normalise query first)."""
    q = query_embedding.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    scores, ids = index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        e = entries[idx]
        results.append({
            "section_number": e.get("section_number", ""),
            "text": e.get(text_field, e.get("text", "")),
            "score": float(score),
        })
    return results


def search_l2_chunks(
    query_embedding: np.ndarray,
    index: faiss.Index,
    chunks: list[str],
    meta: list[dict],
    top_k: int,
) -> list[dict]:
    """Search an IndexFlatL2 chunk index; convert L2 distance to similarity."""
    q = query_embedding.reshape(1, -1).astype("float32")
    distances, ids = index.search(q, top_k)
    results = []
    for dist, idx in zip(distances[0], ids[0]):
        if idx < 0:
            continue
        m = meta[idx]
        results.append({
            "chunk_id": idx,
            "section": m.get("section", ""),
            "section_path": m.get("section_path", ""),
            "score": float(1.0 / (1.0 + dist)),
            "preview": chunks[idx][:120].replace("\n", " "),
        })
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Section-level embedding filter benchmark")
    parser.add_argument("--run-dir", help="KG run directory (default: latest)")
    parser.add_argument(
        "--artifact-dir",
        help="Artifact directory with chunk index "
             "(default: index/sections, or index/partial_sections with --partial)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--strategy",
        choices=["titles", "summaries", "vector", "all"],
        default="all",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        default=False,
        help="Use the partial index (index/partial_sections/) instead of the full index",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir()
    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir)
    elif args.partial:
        artifact_dir = PROJECT_ROOT / "index" / "partial_sections"
    else:
        artifact_dir = ARTIFACT_DIR
    do_titles = args.strategy in ("titles", "all")
    do_summaries = args.strategy in ("summaries", "all")
    do_vector = args.strategy in ("vector", "all")

    logger.info("KG run dir:    %s", run_dir)
    logger.info("Artifact dir:  %s", artifact_dir)

    benchmarks = load_benchmarks()
    queries = [bm["question"] for bm in benchmarks]

    # ── MiniLM for titles / summaries ─────────────────────────────────────────
    minilm_query_embeddings = None
    titles_index = titles_entries = None
    summary_index = summary_entries = None

    if do_titles or do_summaries:
        logger.info("Loading MiniLM model: %s", MINILM_MODEL)
        minilm = SentenceTransformer(MINILM_MODEL)
        logger.info("Embedding queries with MiniLM…")
        minilm_query_embeddings = minilm.encode(queries, show_progress_bar=False).astype("float32")

        if do_titles:
            titles_index, titles_entries = load_or_build_titles_index(run_dir, minilm)
        if do_summaries:
            summary_index, summary_entries = load_summary_index(run_dir)

    # ── Qwen3 GGUF for chunk vector search ────────────────────────────────────
    qwen_query_embeddings = None
    chunk_index = chunk_texts = chunk_meta = None

    if do_vector:
        config = RAGConfig()
        logger.info("Loading Qwen3 embedder: %s", config.embed_model)
        qwen_embedder = CachedEmbedder(str(PROJECT_ROOT / config.embed_model))
        logger.info("Embedding queries with Qwen3…")
        qwen_query_embeddings = qwen_embedder.encode(queries)
        chunk_index, chunk_texts, chunk_meta = load_chunk_index(artifact_dir)

    # ── results ───────────────────────────────────────────────────────────────
    for i, bm in enumerate(benchmarks):
        print(f"\n{'=' * 70}")
        print(f"[{bm['id']}]  {bm['question']}")
        print(f"{'=' * 70}")

        if do_titles:
            hits = search_ip(minilm_query_embeddings[i], titles_index, titles_entries, args.top_k, "text")
            print(f"\n  ── Titles (top {args.top_k}) ──")
            for j, h in enumerate(hits, 1):
                print(f"    {j}. §{h['section_number']:10s}  [{h['score']:.4f}]  {h['text']}")

        if do_summaries:
            hits = search_ip(minilm_query_embeddings[i], summary_index, summary_entries, args.top_k, "summary_text")
            print(f"\n  ── Summaries (top {args.top_k}) ──")
            for j, h in enumerate(hits, 1):
                preview = h["text"][:120].replace("\n", " ")
                print(f"    {j}. §{h['section_number']:10s}  [{h['score']:.4f}]  {preview}…")

        if do_vector:
            hits = search_l2_chunks(qwen_query_embeddings[i], chunk_index, chunk_texts, chunk_meta, args.top_k)
            print(f"\n  ── Vector search / chunks (top {args.top_k}) ──")
            for j, h in enumerate(hits, 1):
                print(f"    {j}. chunk {h['chunk_id']:4d}  [{h['score']:.4f}]  {h['section']}")
                print(f"           {h['preview']}…")


if __name__ == "__main__":
    main()
