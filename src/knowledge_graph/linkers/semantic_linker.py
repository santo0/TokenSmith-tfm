"""Semantic linker: extracts typed directed triples via LLM to build a MultiDiGraph."""

import json
import logging
import re
from typing import Any

import networkx as nx

from src.knowledge_graph.linkers import BaseLinker
from src.knowledge_graph.models import ExtractionResult
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.utils.normalizer import Normalizer
from src.knowledge_graph.utils.semantic_prompts import (
    RELATION_TYPES,
    TRIPLE_EXTRACTION_SYSTEM_PROMPT,
    TRIPLE_EXTRACTION_USER_TEMPLATE,
    KEYWORD_SECTION_TEMPLATE,
    CHUNK_ENTRY_TEMPLATE,
    _RELATION_LIST_STR,
)

logger = logging.getLogger(__name__)


class SemanticLinker(BaseLinker):
    """Build a directed knowledge graph by extracting typed semantic triples via LLM.

    For each batch of chunks the LLM is prompted to return ``(subject, relation,
    object)`` triples.  The result is a :class:`networkx.MultiDiGraph` whose
    edges carry ``relation``, ``weight``, and ``chunk_ids`` attributes.

    Args:
        chunks: Mapping of ``chunk_id → raw text`` used as LLM input.
        api_key: OpenRouter API key.
        llm_model: OpenRouter model identifier.
        corpus_description: Human-readable corpus context injected into the prompt.
        allowed_relations: Subset of relation types to extract.  Empty list means
            all 10 default types.
        allowed_nodes: If provided, triples where either subject or object is not
            in this set are dropped (keeps the graph anchored to canonical vocab).
        batch_size: Number of chunks sent per LLM call.
        min_occurrence: Minimum number of times a triple must appear to be kept.
        retries: Extra LLM retries on failure.
        normalizer: Optional shared :class:`Normalizer` instance.
    """

    def __init__(
        self,
        chunks: dict[int, str],
        api_key: str,
        llm_model: str = "openai/gpt-4o-mini",
        corpus_description: str = "",
        allowed_relations: list[str] | None = None,
        allowed_nodes: set[str] | None = None,
        batch_size: int = 5,
        min_occurrence: int = 1,
        retries: int = 1,
        normalizer: Normalizer | None = None,
    ):
        super().__init__()
        self.chunks = chunks
        self.llm_model = llm_model
        self.corpus_description = corpus_description
        self.allowed_relations: set[str] = (
            set(allowed_relations) if allowed_relations else set(RELATION_TYPES)
        )
        self.allowed_nodes = allowed_nodes  # None = unconstrained
        self.batch_size = batch_size
        self.min_occurrence = min_occurrence
        self._client = OpenRouterClient(api_key, retries=retries)
        self._normalizer = normalizer or Normalizer()

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "llm_model": self.llm_model,
                "corpus_description": self.corpus_description,
                "allowed_relations": sorted(self.allowed_relations),
                "batch_size": self.batch_size,
                "min_occurrence": self.min_occurrence,
            }
        )
        return config

    # ── public API ────────────────────────────────────────────────────────────

    def link(self, extractions: list[ExtractionResult]) -> nx.MultiDiGraph:
        """Extract triples from chunks and return a typed directed multi-graph.

        Args:
            extractions: One :class:`ExtractionResult` per chunk (used to know
                which chunk IDs to process and optionally to supply keywords).

        Returns:
            :class:`networkx.MultiDiGraph` with node attribute ``chunk_ids`` and
            edge attributes ``relation``, ``weight``, ``chunk_ids``.
        """
        chunk_ids = [e.chunk_id for e in extractions]
        keyword_map: dict[int, list[str]] = {e.chunk_id: e.keywords for e in extractions}

        # Batch chunk IDs
        batches: list[list[int]] = [
            chunk_ids[i: i + self.batch_size]
            for i in range(0, len(chunk_ids), self.batch_size)
        ]

        all_triples: list[tuple[int, str, str, str]] = []
        for batch_idx, batch in enumerate(batches):
            logger.info(
                "SemanticLinker: batch %d/%d (%d chunks)",
                batch_idx + 1,
                len(batches),
                len(batch),
            )
            batch_pairs = [
                (cid, self.chunks[cid])
                for cid in batch
                if cid in self.chunks
            ]
            if not batch_pairs:
                continue
            # Collect keywords for the batch as context
            batch_keywords: list[str] = []
            for cid in batch:
                batch_keywords.extend(keyword_map.get(cid, []))
            triples = self._extract_triples_batch(batch_pairs, batch_keywords)
            all_triples.extend(triples)

        logger.info("SemanticLinker: extracted %d raw triples", len(all_triples))
        graph = self._build_graph(all_triples, keyword_map)
        logger.info(
            "SemanticLinker: graph has %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    # ── private helpers ───────────────────────────────────────────────────────

    def _extract_triples_batch(
        self,
        batch: list[tuple[int, str]],
        keywords: list[str],
    ) -> list[tuple[int, str, str, str]]:
        """Call the LLM on one batch; return validated ``(chunk_id, subj, rel, obj)`` tuples."""
        chunks_section = "\n".join(
            CHUNK_ENTRY_TEMPLATE.format(chunk_id=cid, text=text[:1200])
            for cid, text in batch
        )
        keyword_section = (
            KEYWORD_SECTION_TEMPLATE.format(keywords=", ".join(sorted(set(keywords))))
            if keywords
            else ""
        )
        # Inject corpus description into system prompt if provided
        system = TRIPLE_EXTRACTION_SYSTEM_PROMPT
        if self.corpus_description:
            system = system.replace(
                "a technical textbook",
                f"'{self.corpus_description}'",
            )

        user = TRIPLE_EXTRACTION_USER_TEMPLATE.format(
            relation_list=_RELATION_LIST_STR,
            keyword_section=keyword_section,
            chunks_section=chunks_section,
        )

        try:
            raw = self._client.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            return self._parse_response(raw, {cid for cid, _ in batch})
        except Exception as exc:
            logger.warning("SemanticLinker: LLM call failed — %s", exc)
            return []

    def _parse_response(
        self,
        raw: str,
        valid_chunk_ids: set[int],
    ) -> list[tuple[int, str, str, str]]:
        """Parse and validate LLM JSON output into clean triples."""
        # Strip optional markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("SemanticLinker: JSON parse failed (%s) — raw=%r", exc, raw[:200])
            return []

        triples: list[tuple[int, str, str, str]] = []
        for chunk_entry in data.get("chunks", []):
            try:
                cid = int(chunk_entry["chunk_id"])
            except (KeyError, ValueError, TypeError):
                continue
            if cid not in valid_chunk_ids:
                continue
            for triple in chunk_entry.get("triples", []):
                subj = str(triple.get("subject", "")).strip().lower()
                rel = str(triple.get("relation", "")).strip().upper()
                obj = str(triple.get("object", "")).strip().lower()

                # Validate
                if not subj or not obj or subj == obj:
                    continue
                if rel not in self.allowed_relations:
                    logger.debug("SemanticLinker: skipping unknown relation %r", rel)
                    continue

                # Normalize via existing normalizer
                [norm_subj] = self._normalizer.normalize([subj])
                [norm_obj] = self._normalizer.normalize([obj])
                if not norm_subj or not norm_obj or norm_subj == norm_obj:
                    continue

                # Filter against canonical vocabulary if provided
                if self.allowed_nodes is not None:
                    if norm_subj not in self.allowed_nodes or norm_obj not in self.allowed_nodes:
                        logger.debug(
                            "SemanticLinker: dropping triple (%r, %r, %r) — not in allowed_nodes",
                            norm_subj,
                            rel,
                            norm_obj,
                        )
                        continue

                triples.append((cid, norm_subj, rel, norm_obj))

        return triples

    def _build_graph(
        self,
        triples: list[tuple[int, str, str, str]],
        keyword_map: dict[int, list[str]],
    ) -> nx.MultiDiGraph:
        """Aggregate validated triples into a MultiDiGraph with weights."""
        graph = nx.MultiDiGraph()

        # First pass: add all keyword nodes so isolated concept nodes exist
        for cid, keywords in keyword_map.items():
            for kw in keywords:
                [norm_kw] = self._normalizer.normalize([kw])
                if not norm_kw:
                    continue
                if self.allowed_nodes is not None and norm_kw not in self.allowed_nodes:
                    continue
                if graph.has_node(norm_kw):
                    if cid not in graph.nodes[norm_kw]["chunk_ids"]:
                        graph.nodes[norm_kw]["chunk_ids"].append(cid)
                else:
                    graph.add_node(norm_kw, chunk_ids=[cid])

        # Track (subj, rel, obj) → {weight, chunk_ids} for deduplication
        edge_counts: dict[tuple[str, str, str], dict] = {}
        for cid, subj, rel, obj in triples:
            key = (subj, rel, obj)
            if key not in edge_counts:
                edge_counts[key] = {"weight": 0, "chunk_ids": []}
            edge_counts[key]["weight"] += 1
            edge_counts[key]["chunk_ids"].append(cid)

        # Second pass: build graph from aggregated triples
        deleted = 0
        for (subj, rel, obj), data in edge_counts.items():
            if data["weight"] < self.min_occurrence:
                deleted += 1
                continue

            # Ensure subject node exists with chunk_ids
            for node, cids in ((subj, data["chunk_ids"]), (obj, data["chunk_ids"])):
                if not graph.has_node(node):
                    graph.add_node(node, chunk_ids=list(set(cids)))
                else:
                    existing = graph.nodes[node]["chunk_ids"]
                    for cid in cids:
                        if cid not in existing:
                            existing.append(cid)

            graph.add_edge(
                subj,
                obj,
                relation=rel,
                weight=data["weight"],
                chunk_ids=list(set(data["chunk_ids"])),
            )

        self.metadata["deleted_edges"] = deleted
        self.metadata["total_raw_triples"] = len(triples)
        logger.info(
            "SemanticLinker: pruned %d edges below min_occurrence=%d",
            deleted,
            self.min_occurrence,
        )
        return graph
