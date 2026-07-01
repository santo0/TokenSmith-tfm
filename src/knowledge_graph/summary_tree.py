from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import faiss

from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.section_tree import SectionNode, SectionTree
from src.knowledge_graph.prompts import (
    CHUNK_SUMMARY_PROMPT,
    SECTION_SUMMARY_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
)

SUMMARY_INDEX_FILE = "summary_index.faiss"
SUMMARY_META_FILE = "summary_meta.json"


@dataclass
class SummaryEntry:
    section_number: str
    level: int  # 0 = chunk-group; matches SectionNode.level for section nodes
    chunk_ids: list[int]
    summary_text: str


def _windowed(items: list[int], window: int) -> list[list[int]]:
    """Split *items* into consecutive non-overlapping groups of size *window*."""
    return [items[i: i + window] for i in range(0, len(items), window)]


def _all_chunk_ids(node: SectionNode) -> list[int]:
    ids = list(node.chunk_ids)
    for child in node.children:
        ids.extend(_all_chunk_ids(child))
    return ids


def _collect_nodes_by_height(root: SectionNode) -> dict[int, list[SectionNode]]:
    """Group all descendant nodes by height (0 = leaf, increasing toward root)."""
    by_height: dict[int, list[SectionNode]] = {}

    def walk(node: SectionNode) -> int:
        h = 0 if not node.children else 1 + max(walk(c) for c in node.children)
        by_height.setdefault(h, []).append(node)
        return h

    for child in root.children:
        walk(child)
    return by_height


def build_summary_index(
    client: OpenRouterClient,
    summary_model: str,
    section_tree: SectionTree,
    chunks: dict[int, str],
    embed_model: str,
    chunk_window: int,
    run_dir: str,
) -> tuple[faiss.Index, list[SummaryEntry]]:
    """Build LLM summaries for all tree levels and persist as a FAISS index.

    Args:
        section_tree:  Pre-built ``SectionTree`` for the corpus.
        chunks:        Mapping of chunk ID → raw text.

        embed_model:   SentenceTransformer model name for embedding summaries.
        chunk_window:  Number of adjacent chunks summarized together at the
                       leaf level (level=0).  Larger values → fewer LLM calls
                       but coarser granularity.
        run_dir:       Directory where ``summary_index.faiss`` and
                       ``summary_meta.json`` will be written.

    Returns:
        ``(index, entries)`` — the populated FAISS index and the parallel list
        of ``SummaryEntry`` objects (index position == FAISS row).
    """
    entries: list[SummaryEntry] = []
    section_summary_cache: dict[str, str] = {}

    by_height = _collect_nodes_by_height(section_tree.root)
    if not by_height:
        raise ValueError("No summaries generated — section tree may be empty.")
    max_height = max(by_height)

    # ── Height 0: leaf nodes ──────────────────────────────────────────────
    leaf_nodes = by_height.get(0, [])

    # Phase A1: all chunk-group summaries across all leaves in one batch
    chunk_tasks: list[tuple[SectionNode, list[int]]] = []
    chunk_requests: list[dict] = []
    for node in leaf_nodes:
        for group in _windowed(node.chunk_ids, chunk_window):
            text = "\n\n".join(chunks[cid] for cid in group if cid in chunks).strip()
            if text:
                chunk_tasks.append((node, group))
                chunk_requests.append({"messages": [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": CHUNK_SUMMARY_PROMPT.format(text=text)},
                ]})

    chunk_summaries_by_section: dict[str, list[str]] = {}
    if chunk_requests:
        outcomes = client.chat_many(chunk_requests, model=summary_model)
        for (node, group), outcome in zip(chunk_tasks, outcomes):
            if isinstance(outcome, Exception):
                continue
            entries.append(SummaryEntry(
                section_number=node.section_number,
                level=0,
                chunk_ids=list(group),
                summary_text=outcome,
            ))
            chunk_summaries_by_section.setdefault(node.section_number, []).append(outcome)

    # Phase A2: leaf section summaries in one batch
    leaf_section_tasks: list[SectionNode] = []
    leaf_section_requests: list[dict] = []
    for node in leaf_nodes:
        group_summaries = chunk_summaries_by_section.get(node.section_number, [])
        if group_summaries:
            leaf_section_tasks.append(node)
            leaf_section_requests.append({"messages": [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": SECTION_SUMMARY_PROMPT.format(
                    heading=node.heading,
                    summaries="\n\n".join(group_summaries),
                )},
            ]})

    if leaf_section_requests:
        outcomes = client.chat_many(leaf_section_requests, model=summary_model)
        for node, outcome in zip(leaf_section_tasks, outcomes):
            if isinstance(outcome, Exception):
                continue
            entries.append(SummaryEntry(
                section_number=node.section_number,
                level=node.level,
                chunk_ids=list(node.chunk_ids),
                summary_text=outcome,
            ))
            section_summary_cache[node.section_number] = outcome

    # ── Heights 1…max: internal nodes, bottom-up ─────────────────────────
    for height in range(1, max_height + 1):
        internal_tasks: list[SectionNode] = []
        internal_requests: list[dict] = []
        for node in by_height.get(height, []):
            child_summaries = [
                section_summary_cache[child.section_number]
                for child in node.children
                if child.section_number in section_summary_cache
            ]
            if child_summaries:
                internal_tasks.append(node)
                internal_requests.append({"messages": [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": SECTION_SUMMARY_PROMPT.format(
                        heading=node.heading,
                        summaries="\n\n".join(child_summaries),
                    )},
                ]})

        if internal_requests:
            outcomes = client.chat_many(internal_requests, model=summary_model)
            for node, outcome in zip(internal_tasks, outcomes):
                if isinstance(outcome, Exception):
                    continue
                entries.append(SummaryEntry(
                    section_number=node.section_number,
                    level=node.level,
                    chunk_ids=_all_chunk_ids(node),
                    summary_text=outcome,
                ))
                section_summary_cache[node.section_number] = outcome

    if not entries:
        raise ValueError("No summaries generated — section tree may be empty.")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embed_model)
    texts = [e.summary_text for e in entries]
    embeddings = model.encode(texts, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(run_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(run_dir, SUMMARY_INDEX_FILE))
    with open(os.path.join(run_dir, SUMMARY_META_FILE), "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in entries], f, indent=2, ensure_ascii=False)

    return index, entries


def load_summary_index(run_dir: str) -> tuple[faiss.Index, list[SummaryEntry]]:
    """Load a persisted summary FAISS index and its metadata from *run_dir*.

    Raises:
        FileNotFoundError: If either artifact is absent.
    """
    index_path = os.path.join(run_dir, SUMMARY_INDEX_FILE)
    meta_path = os.path.join(run_dir, SUMMARY_META_FILE)

    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Summary index not found: {index_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Summary metadata not found: {meta_path}")

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        entries = [SummaryEntry(**d) for d in json.load(f)]

    return index, entries
