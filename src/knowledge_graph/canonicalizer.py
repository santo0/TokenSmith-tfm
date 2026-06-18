import json
import logging
from collections import Counter
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from src.knowledge_graph.models import ExtractionResult, CanonicalizationResult
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.normalizer import Normalizer
from src.knowledge_graph.prompts import SYNONYM_PROMPT, SYNONYM_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Canonicalizer:
    """Semantic canonicalization of KG keywords.

    Args:
        corpus_description: Human-readable description of the corpus
            (e.g. Title of the textbook or main topic of the document).
            Injected into the LLM system prompt as domain context.
        api_key: OpenRouter API key for the LLM verification step.
        embedding_model: Sentence-transformer model name for keyword embedding.
        similarity_threshold: Cosine similarity threshold for complete-linkage
            clustering. A group forms only when ALL pairs in it exceed this value.
        max_group_size: Maximum keywords per LLM call. Oversized clusters are
            force-split into fixed-size chunks before the LLM step.
        llm_model: OpenRouter model identifier.
        batch_size: Number of small groups (≤5 keywords) to batch per LLM call.
        fallback_threshold: Cosine similarity threshold used at query time when a
            keyword is not in the synonym table (embedding-based fallback).
    """

    def __init__(
        self,
        corpus_description: str,
        api_key: str,
        embedding_model: str,
        similarity_threshold: float = 0.78,
        max_group_size: int = 30,
        llm_model: str = "google/gemini-3-flash-preview",
        batch_size: int = 15,
        fallback_threshold: float = 0.85,
        retries: int = 1,
        normalizer: Normalizer | None = None,
    ):
        self.corpus_description = corpus_description
        self.similarity_threshold = similarity_threshold
        self.max_group_size = max_group_size
        self.llm_model = llm_model
        self.batch_size = batch_size
        self.fallback_threshold = fallback_threshold
        self._normalizer = normalizer or Normalizer()
        self.retries = retries
        self._client = OpenRouterClient(api_key, retries=retries)

        logger.info("Loading embedding model: %s", embedding_model)
        self._model = SentenceTransformer(embedding_model)
        self._embedding_model_name = embedding_model
        self._llm_calls = 0

    def get_config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "corpus_description": self.corpus_description,
            "embedding_model": self._embedding_model_name,
            "similarity_threshold": self.similarity_threshold,
            "max_group_size": self.max_group_size,
            "llm_model": self.llm_model,
            "batch_size": self.batch_size,
            "fallback_threshold": self.fallback_threshold,
            "retries": self.retries,
        }

    def canonicalize(
        self, extractions: list[ExtractionResult]
    ) -> tuple[list[ExtractionResult], CanonicalizationResult]:
        """Run canonicalization on a list of extraction results.

        Returns:
            Updated extractions (nodes replaced by canonical forms) and a
            CanonicalizationResult carrying the artifacts and run statistics.
        """
        all_keywords = self._collect_keywords(extractions)
        n = len(all_keywords)
        logger.info("Canonicalizing %d unique keywords…", n)

        # 2a — embed
        logger.info("  [2a] Embedding keywords…")
        embeddings = self._embed(all_keywords)

        # 2b — cluster
        logger.info("  [2b] Complete-linkage clustering (θ=%.2f)…",
                    self.similarity_threshold)
        groups = self._cluster(all_keywords, embeddings)
        singletons = [g[0] for g in groups if len(g) == 1]
        non_singletons = [g for g in groups if len(g) > 1]
        logger.info(
            "       %d singletons, %d candidate groups", len(
                singletons), len(non_singletons)
        )

        # 2c — LLM verification
        logger.info("  [2c] LLM verification (%d groups)…",
                    len(non_singletons))
        self._llm_calls = 0
        synonym_table = self._verify_with_llm(non_singletons)

        # 2d — build structures
        canonical_keywords = sorted(
            set(synonym_table.values()) | set(singletons) | set(all_keywords))

        logger.info("  [2d] Embedding %d canonical keywords…",
                    len(canonical_keywords))
        canonical_embeddings = self._embed(canonical_keywords)

        counts = Counter(synonym_table.values())
        merges_performed = sum(c - 1 for c in counts.values() if c > 1)

        stats = {
            "keywords_after_stage1": n,
            "candidate_groups": len(non_singletons),
            "singletons": len(singletons),
            "merges_performed": merges_performed,
            "canonical_keywords_final": len(canonical_keywords),
            "llm_calls": self._llm_calls,
        }

        logger.info(
            "Canonicalization done: %d → %d keywords, %d merges, %d LLM calls",
            n, len(canonical_keywords), merges_performed, self._llm_calls,
        )

        updated = self._apply(extractions, synonym_table)
        result = CanonicalizationResult(
            synonym_table=synonym_table,
            canonical_keywords=canonical_keywords,
            canonical_embeddings=canonical_embeddings,
            stats=stats,
        )
        return updated, result

    def _collect_keywords(self, extractions: list[ExtractionResult]) -> list[str]:
        # Normalize at collection time so clustering and the synonym table operate
        # on the same forms that _apply will look up later.
        seen: set[str] = set()
        keywords: list[str] = []
        for er in extractions:
            for kw in er.keywords:
                norm = self._normalize_kw(kw)
                if norm and norm not in seen:
                    keywords.append(norm)
                    seen.add(norm)
        return keywords

    def _embed(self, keywords: list[str]) -> np.ndarray:
        return self._model.encode(keywords, show_progress_bar=False)

    def _cluster(self, keywords: list[str], embeddings: np.ndarray) -> list[list[str]]:
        """Complete-linkage clustering.

        A group forms only when ALL pairs within it have cosine similarity ≥
        self.similarity_threshold (equivalently, distance ≤ 1 − threshold).
        Oversized groups are force-split into max_group_size chunks.
        """
        n = len(keywords)
        if n == 1:
            return [keywords]

        sim = cosine_similarity(embeddings)
        np.fill_diagonal(sim, 1.0)
        dist = np.clip(1.0 - sim, 0.0, None)

        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="complete")
        labels = fcluster(Z, t=1.0 - self.similarity_threshold,
                          criterion="distance")

        raw_groups: dict[int, list[str]] = {}
        for kw, label in zip(keywords, labels):
            raw_groups.setdefault(int(label), []).append(kw)

        result: list[list[str]] = []
        for group in raw_groups.values():
            if len(group) <= self.max_group_size:
                result.append(group)
            else:
                for i in range(0, len(group), self.max_group_size):
                    result.append(group[i: i + self.max_group_size])
        return result

    def _verify_with_llm(self, groups: list[list[str]]) -> dict[str, str]:
        """Return a partial synonym table for all keywords in non-singleton groups."""
        small = [g for g in groups if len(g) <= 5]
        large = [g for g in groups if len(g) > 5]

        batches: list[list[list[str]]] = [
            small[i: i + self.batch_size] for i in range(0, len(small), self.batch_size)
        ] + [[g] for g in large]

        if not batches:
            return {}

        requests_ = [
            {"messages": self._build_llm_messages(b), "response_format": {"type": "json_object"}}
            for b in batches
        ]
        outcomes = self._client.chat_many(requests_, model=self.llm_model)
        self._llm_calls += len(requests_)

        partial: dict[str, str] = {}
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                logger.warning("LLM call failed after all attempts (%s) — batch skipped", outcome)
                continue
            try:
                partial.update(self._parse_llm_response(outcome))
            except Exception as e:
                logger.warning("LLM response parse failed: %s — batch skipped", e)
        return partial

    def _normalize_kw(self, kw: str) -> str:
        """Normalize a single keyword using the configured Normalizer or strip+lower."""
        result = self._normalizer.normalize([kw])
        return result[0] if result else kw.strip().lower()

    def _build_llm_messages(self, groups: list[list[str]]) -> list[dict]:
        groups_text = "\n".join(
            f"Group {i + 1}: {json.dumps(g)}" for i, g in enumerate(groups)
        )
        return [
            {"role": "system", "content": SYNONYM_SYSTEM_PROMPT.format(
                corpus_description=self.corpus_description)},
            {"role": "user", "content": SYNONYM_PROMPT.format(groups_text=groups_text)},
        ]

    def _parse_llm_response(self, content: str) -> dict[str, str]:
        partial: dict[str, str] = {}
        for group_result in json.loads(content).get("groups", []):
            for sg in group_result.get("synonym_groups", []):
                canonical = self._normalize_kw(sg.get("canonical", ""))
                for member in sg.get("members", []):
                    if member:
                        partial[self._normalize_kw(member)] = canonical
        return partial

    def _llm_call(self, groups: list[list[str]]) -> dict[str, str]:
        try:
            content = self._client.chat(
                model=self.llm_model,
                messages=self._build_llm_messages(groups),
                response_format={"type": "json_object"},
            )
            self._llm_calls += 1
            return self._parse_llm_response(content)
        except Exception as e:
            logger.warning("LLM call failed after all attempts (%s) — batch skipped", e)
            return {}

    def _apply(
        self, extractions: list[ExtractionResult], synonym_table: dict[str, str]
    ) -> list[ExtractionResult]:
        updated = []
        for er in extractions:
            seen: set[str] = set()
            canonical_nodes: list[str] = []
            for kw in er.keywords:
                norm = self._normalize_kw(kw)
                # synonym_table keys are normalized; fall back to normalized form,
                # not the raw string, so singletons are also normalized in the graph.
                canonical = synonym_table.get(norm, norm)
                if canonical not in seen:
                    canonical_nodes.append(canonical)
                    seen.add(canonical)
            updated.append(ExtractionResult(
                chunk_id=er.chunk_id, keywords=canonical_nodes))
        return updated


class NullCanonicalizer:
    """Drop-in replacement for Canonicalizer that skips all merging.

    Normalizes keywords (lowercase + lemmatize via Normalizer) but performs
    no embedding clustering or LLM verification. Produces an empty synonym
    table — every normalized keyword maps only to itself. Used for the
    canonicalization ablation experiment (D1).

    Args:
        embedding_model: Sentence-transformer model name. Embeddings are still
            generated so the keyword FAISS index remains functional at query time.
    """

    def __init__(self, embedding_model: str, normalizer: Normalizer | None = None):
        self._normalizer = normalizer or Normalizer()
        logger.info("Loading embedding model: %s", embedding_model)
        self._model = SentenceTransformer(embedding_model)
        self._embedding_model_name = embedding_model

    def get_config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "embedding_model": self._embedding_model_name,
        }

    def _normalize_kw(self, kw: str) -> str:
        result = self._normalizer.normalize([kw])
        return result[0] if result else kw.strip().lower()

    def canonicalize(
        self, extractions: list[ExtractionResult]
    ) -> tuple[list[ExtractionResult], CanonicalizationResult]:
        # Collect globally unique normalized keywords (mirrors Canonicalizer._collect_keywords)
        seen: set[str] = set()
        all_keywords: list[str] = []
        for er in extractions:
            for kw in er.keywords:
                norm = self._normalize_kw(kw)
                if norm and norm not in seen:
                    all_keywords.append(norm)
                    seen.add(norm)

        logger.info(
            "NullCanonicalizer: %d unique normalized keywords, 0 merges", len(all_keywords)
        )
        embeddings = self._model.encode(all_keywords, show_progress_bar=False)

        # Apply normalization and per-chunk deduplication; no synonym remapping
        updated: list[ExtractionResult] = []
        for er in extractions:
            chunk_seen: set[str] = set()
            chunk_kws: list[str] = []
            for kw in er.keywords:
                norm = self._normalize_kw(kw)
                if norm and norm not in chunk_seen:
                    chunk_kws.append(norm)
                    chunk_seen.add(norm)
            updated.append(ExtractionResult(chunk_id=er.chunk_id, keywords=chunk_kws))

        stats = {
            "keywords_after_stage1": len(all_keywords),
            "candidate_groups": 0,
            "singletons": len(all_keywords),
            "merges_performed": 0,
            "canonical_keywords_final": len(all_keywords),
            "llm_calls": 0,
        }
        return updated, CanonicalizationResult(
            synonym_table={},
            canonical_keywords=all_keywords,
            canonical_embeddings=np.array(embeddings, dtype=np.float32),
            stats=stats,
        )


class MockCanonicalizer:
    """Drop-in replacement for Canonicalizer that replays a pre-saved result.

    Loads a cache file produced by generate_canon_cache.py and returns the
    stored extractions and CanonicalizationResult without running any model
    or LLM. Useful for iterating on pipeline stages that follow canonicalization.

    Args:
        cache_path: Path to the JSON cache file (relative to repo root or absolute).
    """

    def __init__(self, cache_path: str):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._updated_extractions = [
            ExtractionResult(chunk_id=e["chunk_id"], keywords=e["keywords"])
            for e in data["updated_extractions"]
        ]
        self._result = CanonicalizationResult(
            synonym_table=data["synonym_table"],
            canonical_keywords=data["canonical_keywords"],
            canonical_embeddings=np.array(data["canonical_embeddings"], dtype=np.float32),
            stats=data.get("stats", {}),
        )
        logger.warning("MockCanonicalizer: loaded cache from %s", cache_path)

    def get_config(self) -> dict[str, Any]:
        return {"class": self.__class__.__name__}

    def canonicalize(
        self, extractions: list[ExtractionResult]
    ) -> tuple[list[ExtractionResult], CanonicalizationResult]:
        logger.warning("MockCanonicalizer: returning cached canonicalization, input ignored")
        return self._updated_extractions, self._result
