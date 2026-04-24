from __future__ import annotations

import logging

import faiss
import networkx as nx
from src.retriever import FAISSRetriever
from src.knowledge_graph.query import (
    KGNodeRetriever,
    SectionSummaryRetriever,
    SectionTreeRetriever,
    CanonicalLookup,
)
from src.knowledge_graph.section_tree import SectionTree
from src.knowledge_graph.summary_tree import SummaryEntry

logger = logging.getLogger(__name__)


def _rrf_fuse(
    signals: list[dict[int, float]],
    candidate_ids: set[int],
    k: int = 60,
) -> dict[int, float]:
    """Reciprocal Rank Fusion over *candidate_ids* for an arbitrary number of score dicts.

    Each signal contributes ``1 / (k + rank)`` per candidate, where rank is
    determined by descending score within *candidate_ids*. Candidates absent
    from a signal are ranked last.
    """
    ranks_per_signal: list[dict[int, int]] = []
    for scores in signals:
        ordered = sorted(candidate_ids, key=lambda c: scores.get(c, 0.0), reverse=True)
        ranks_per_signal.append({cid: r + 1 for r, cid in enumerate(ordered)})

    return {
        cid: sum(1.0 / (k + ranks[cid]) for ranks in ranks_per_signal)
        for cid in candidate_ids
    }


class HybridRetriever:
    """Three-stage hybrid retriever: dense → section filter → KG boost, fused via RRF.

    Implements the standard ``Retriever`` interface (``name`` + ``get_scores``) so it
    can be dropped into the existing benchmark and ensemble machinery.

    Stage 1 — Dense (Qwen3 GGUF via CachedEmbedder):
        FAISS search over chunk embeddings produces the initial candidate pool.

    Stage 2 — Section summary filter (all-MiniLM SentenceTransformer):
        The query is embedded with the summary model and compared against
        LLM-generated section/subsection summaries. Chunks covered by at least
        one matching summary entry are kept; chunks in the dense pool but outside
        all matching sections are discarded. Chunks from matching sections that
        were not in the dense pool are added (candidate expansion).

    Stage 3 — KG boost (graph BFS):
        KGNodeRetriever scores the filtered candidates via graph topology.

    Final scores are produced by Reciprocal Rank Fusion (RRF) over all three
    signals.

    Args:
        faiss_index: Dense FAISS index built with the Qwen3 embedder.
        dense_embed_model: Path/name of the dense embedding model (must match
            the model used to build *faiss_index*).
        chunks: All chunk texts ordered by chunk_id (used for FAISS bounds).
        summary_index: FAISS IndexFlatIP built from section summary embeddings
            (all-MiniLM L2-normalised).
        summary_entries: Metadata aligned with *summary_index* rows.
        summary_embed_model: HuggingFace model name used for summary embeddings
            (must match the model used to build *summary_index*).
        graph: Knowledge graph (NetworkX) with keyword nodes carrying chunk_ids.
        kg_chunks: Mapping of chunk_id → text used by the KG retriever.
        canonical_lookup: Optional synonym/canonical resolver for KG node matching.
        dense_top_k: Number of dense candidates to retrieve in Stage 1.
        top_sections: Number of summary entries to retrieve in Stage 2.
        rrf_k: RRF smoothing constant (default 60).
        kg_neighbor_weight: BFS hop decay for KGNodeRetriever.
        kg_num_hops: BFS depth for KGNodeRetriever.
    """

    def __init__(
        self,
        faiss_index: faiss.Index,
        dense_embed_model: str,
        chunks: list[str],
        summary_index: faiss.Index,
        summary_entries: list[SummaryEntry],
        summary_embed_model: str,
        graph: nx.Graph,
        kg_chunks: dict[int, str],
        canonical_lookup: CanonicalLookup | None = None,
        dense_top_k: int = 100,
        top_sections: int = 10,
        rrf_k: int = 60,
        kg_neighbor_weight: float = 0.5,
        kg_num_hops: int = 1,
    ) -> None:
        self.name = "hybrid"
        self._dense = FAISSRetriever(faiss_index, dense_embed_model)
        self._kg = KGNodeRetriever(
            graph, kg_chunks, kg_neighbor_weight, kg_num_hops, canonical_lookup
        )
        self._summary_index = summary_index
        self._summary_entries = summary_entries
        self._summary_model_name = summary_embed_model
        self._summary_model = None  # lazy-loaded on first call

        self.chunks = chunks
        self.dense_top_k = dense_top_k
        self.top_sections = top_sections
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Run the hybrid pipeline and return the top-*top_k* (chunk_id, score) pairs.

        Scores are RRF-fused values (higher = more relevant). Not normalised to [0,1].
        """
        # Stage 1: Dense retrieval
        dense_scores: dict[int, float] = self._dense.get_scores(
            query, self.dense_top_k, self.chunks
        )
        logger.debug("Dense candidates: %d", len(dense_scores))

        # Stage 2: Section summary search
        section_scores = self._search_sections(query)
        logger.debug("Section-covered chunks: %d", len(section_scores))

        # Candidate pool: (dense intersection section-covered) filtered to chunks with a section score
        candidate_ids: set[int] = {
            cid
            for cid in (set(dense_scores) & set(section_scores))
            if section_scores.get(cid, 0.0) > 0.0
        }
        if not candidate_ids:
            # No section signal — fall back to dense results only
            logger.debug("No section signal; using dense candidates as fallback.")
            candidate_ids = set(dense_scores)

        logger.debug("Candidate pool after section filter: %d", len(candidate_ids))

        # Stage 3: KG boost
        kg_scores: dict[int, float] = self._kg.get_scores(
            query, len(candidate_ids), self.chunks
        )
        logger.debug("KG-scored chunks: %d", len(kg_scores))

        # Stage 4: RRF fusion
        fused = self._rrf(dense_scores, section_scores, kg_scores, candidate_ids)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def get_scores(self, query: str, pool_size: int, _chunks: list[str]) -> dict[int, float]:
        """Retriever-interface shim: returns RRF scores for the top *pool_size* chunks."""
        return dict(self.retrieve(query, top_k=pool_size))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        faiss_index: faiss.Index,
        chunks: list[str],
        dense_embed_model: str,
        run_dir: str,
        summary_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        canonical_lookup: CanonicalLookup | None = None,
        dense_top_k: int = 100,
        top_sections: int = 10,
        rrf_k: int = 60,
        kg_neighbor_weight: float = 0.5,
        kg_num_hops: int = 1,
    ) -> "HybridRetriever":
        """Construct a HybridRetriever by loading KG artifacts from *run_dir*.

        Args:
            faiss_index: Pre-loaded dense FAISS index.
            chunks: All chunk texts (aligned with faiss_index rows).
            dense_embed_model: Dense embedding model path (e.g. Qwen3 GGUF).
            run_dir: KG run directory containing graph.json, chunks.json,
                summary_index.faiss, and summary_entries.json.
            summary_embed_model: SentenceTransformer model for summary embeddings.
        """
        from src.knowledge_graph.io import (
            load_graph_and_chunks,
            load_summary_data,
        )

        graph, kg_chunks = load_graph_and_chunks(run_dir)

        summary_index, summary_entries = load_summary_data(run_dir)
        if summary_index is None or summary_entries is None:
            raise FileNotFoundError(
                f"Summary artifacts not found in {run_dir!r}. "
                "Run the KG summary pipeline first."
            )

        return cls(
            faiss_index=faiss_index,
            dense_embed_model=dense_embed_model,
            chunks=chunks,
            summary_index=summary_index,
            summary_entries=summary_entries,
            summary_embed_model=summary_embed_model,
            graph=graph,
            kg_chunks=kg_chunks,
            canonical_lookup=canonical_lookup,
            dense_top_k=dense_top_k,
            top_sections=top_sections,
            rrf_k=rrf_k,
            kg_neighbor_weight=kg_neighbor_weight,
            kg_num_hops=kg_num_hops,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_summary_model(self):
        if self._summary_model is None:
            from sentence_transformers import SentenceTransformer
            self._summary_model = SentenceTransformer(self._summary_model_name)
        return self._summary_model

    def _search_sections(self, query: str) -> dict[int, float]:
        """Return per-chunk scores derived from section summary similarity.

        Each chunk's score is the maximum cosine similarity across all summary
        entries that cover it, among the top *top_sections* FAISS hits.
        """
        model = self._get_summary_model()
        q_emb = model.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)

        k = min(self.top_sections, self._summary_index.ntotal)
        sims, idxs = self._summary_index.search(q_emb, k)

        chunk_scores: dict[int, float] = {}
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0 or sim <= 0.0:
                continue
            for cid in self._summary_entries[idx].chunk_ids:
                chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), float(sim))

        return chunk_scores

    def _rrf(
        self,
        dense_scores: dict[int, float],
        section_scores: dict[int, float],
        kg_scores: dict[int, float],
        candidate_ids: set[int],
    ) -> dict[int, float]:
        return _rrf_fuse([dense_scores, section_scores, kg_scores], candidate_ids, self.rrf_k)


class SectionKGRetriever:
    """Section-only retriever: summary embedding + section tree → KG boost, fused via RRF.

    Implements the standard ``Retriever`` interface (``name`` + ``get_scores``).

    Unlike ``HybridRetriever`` there is no dense (Qwen3) retrieval stage. The
    candidate pool is built entirely from structural signals:

    Stage 1 — Section summary (all-MiniLM):
        FAISS search over LLM-generated section/subsection summary embeddings.
        All chunks covered by at least one hit become candidates.

    Stage 2 — Section tree (keyword overlap):
        KG-keyword and heading-keyword overlap scores propagated top-down through
        the section hierarchy. All chunks with a non-zero tree score are added to
        the candidate pool.

    Stage 3 — KG boost (graph BFS):
        KGNodeRetriever scores the candidate pool via graph topology.

    Final scores are produced by RRF over the three signals.
    """

    name = "section_kg"

    def __init__(
        self,
        summary_index: faiss.Index,
        summary_entries: list[SummaryEntry],
        summary_embed_model: str,
        section_tree: SectionTree,
        graph: nx.Graph,
        kg_chunks: dict[int, str],
        chunks: list[str],
        canonical_lookup: CanonicalLookup | None = None,
        top_sections: int = 10,
        rrf_k: int = 60,
        kg_neighbor_weight: float = 0.5,
        kg_num_hops: int = 1,
        heading_alpha: float = 0.5,
        inheritance_decay: float = 0.5,
    ) -> None:
        self._summary = SectionSummaryRetriever(
            summary_index, summary_entries, summary_embed_model, top_sections
        )
        self._tree = SectionTreeRetriever(
            section_tree, graph, canonical_lookup, heading_alpha, inheritance_decay
        )
        self._kg = KGNodeRetriever(
            graph, kg_chunks, kg_neighbor_weight, kg_num_hops, canonical_lookup
        )
        self.chunks = chunks
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return top-*top_k* (chunk_id, score) pairs fused from all three signals."""
        n = len(self.chunks)

        # Stage 1 & 2: section signals
        summary_scores = self._summary.get_scores(query, n, self.chunks)
        tree_scores = self._tree.get_scores(query, n, self.chunks)

        # Candidate pool: any chunk touched by either section signal
        candidate_ids: set[int] = {
            cid for cid in (set(summary_scores) | set(tree_scores))
            if summary_scores.get(cid, 0.0) > 0.0 or tree_scores.get(cid, 0.0) > 0.0
        }
        if not candidate_ids:
            logger.debug("SectionKGRetriever: no section signal for query %r.", query)
            return []

        logger.debug("SectionKGRetriever candidates: %d", len(candidate_ids))

        # Stage 3: KG boost
        kg_scores = self._kg.get_scores(query, len(candidate_ids), self.chunks)
        logger.debug("SectionKGRetriever KG-scored: %d", len(kg_scores))

        # RRF fusion
        fused = _rrf_fuse([summary_scores, tree_scores, kg_scores], candidate_ids, self.rrf_k)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def get_scores(self, query: str, pool_size: int, _chunks: list[str]) -> dict[int, float]:
        """Retriever-interface shim: returns RRF scores for the top *pool_size* chunks."""
        return dict(self.retrieve(query, top_k=pool_size))
