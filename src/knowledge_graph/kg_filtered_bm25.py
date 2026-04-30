"""
KGFilteredBM25Retriever — BM25 retrieval with KG-guided query expansion.

Algorithm:
  1. Match query tokens against KG nodes → core_tokens.
  2. For each pair of disconnected core_tokens, find an IDF-path-aware bridge
     node via weighted shortest path (edge weight = avg inverse-IDF of endpoints).
     Inject at most 1 intermediate node per pair; skip pairs with path > 3 hops.
  3. Score documents with BM25 twice: using core_tokens and core_tokens +
     expansion_tokens.  Final = 0.7 * norm(core) + 0.3 * norm(expanded).
  4. Fallback: if core_tokens has fewer than 2 terms, run standard BM25 on the
     original query string.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Dict, List

import networkx as nx
import numpy as np

from src.retriever import Retriever
from src.index_builder import preprocess_for_bm25
from src.knowledge_graph.query import CanonicalLookup, extract_query_nodes

logger = logging.getLogger(__name__)


class KGFilteredBM25Retriever(Retriever):
    """BM25 retriever that uses the KG to filter and expand the query.

    Scores are the weighted average of two BM25 runs:
    - *core*: BM25 over KG-matched query keywords only.
    - *expanded*: BM25 over those keywords plus bridge nodes found via
      IDF-weighted shortest paths between disconnected keyword pairs.

    Final score = 0.7 * norm(core) + 0.3 * norm(expanded), where both
    vectors are normalised to [0, 1] before combining.

    Fallback: fewer than 2 core_tokens → standard BM25 on the raw query.
    """

    name = "kg_filtered_bm25"

    def __init__(
        self,
        graph: nx.Graph,
        kg_chunks: dict[int, str],
        bm25_index,
        canonical_lookup: CanonicalLookup | None = None,
    ):
        """
        Args:
            graph:            Knowledge graph (nodes carry ``chunk_ids`` lists).
            kg_chunks:        Mapping of chunk ID → chunk text from the KG run.
            bm25_index:       A ``rank_bm25.BM25Okapi`` instance whose corpus
                              positions align with the integer chunk IDs in
                              *kg_chunks*.
            canonical_lookup: Optional synonym / embedding resolver used during
                              query-node extraction.
        """
        self.graph = graph
        self.kg_chunks = kg_chunks
        self.bm25_index = bm25_index
        self.canonical_lookup = canonical_lookup
        self._idf: dict[str, float] | None = None

    # ── IDF helpers ──────────────────────────────────────────────────────────

    def _get_idf(self) -> dict[str, float]:
        """Compute and cache IDF scores (normalised to [0, 1]) for every node."""
        if self._idf is not None:
            return self._idf
        n_chunks = len(self.kg_chunks)
        if n_chunks == 0:
            self._idf = {}
            return self._idf
        raw: dict[str, float] = {}
        for node, data in self.graph.nodes(data=True):
            df = len(data.get("chunk_ids", []))
            raw[node] = math.log(n_chunks / max(df, 1))
        max_idf = max(raw.values(), default=1.0)
        max_idf = max(max_idf, 1e-9)
        self._idf = {n: v / max_idf for n, v in raw.items()}
        return self._idf

    def _inv_idf_edge_weight(
        self, u: str, v: str, edge_data: dict, idf: dict[str, float]
    ) -> float:
        """Dijkstra edge weight combining inverse-IDF of endpoints and co-occurrence strength.

        weight = avg_inverse_IDF(u, v) / edge_co_occurrence_weight

        - Rare nodes (high IDF) → small inverse-IDF → lower cost → preferred bridges.
        - Strongly co-occurring edges (high weight) → larger divisor → lower cost → preferred.
        """
        inv_u = 1.0 / max(idf.get(u, 1e-9), 1e-9)
        inv_v = 1.0 / max(idf.get(v, 1e-9), 1e-9)
        co_occurrence = max(float(edge_data.get("weight", 1)), 1.0)
        return (inv_u + inv_v) / (2.0 * co_occurrence)

    # ── Bridge-node discovery ────────────────────────────────────────────────

    def _find_bridge_nodes(self, query_nodes: list[str]) -> list[str]:
        """Return at most one bridge node per disconnected query-node pair.

        For each pair (u, v) that share no direct edge, this runs Dijkstra
        with IDF-weighted edge costs.  The intermediate node with the highest
        IDF score is selected and injected as an expansion token.  Pairs whose
        shortest path exceeds 3 hops are skipped.
        """
        idf = self._get_idf()
        query_set = set(query_nodes)
        bridge_nodes: list[str] = []

        for u, v in combinations(query_nodes, 2):
            if self.graph.has_edge(u, v):
                print("Pair (%s, %s): direct edge — no bridge needed.", u, v)
                continue

            try:
                path = nx.shortest_path(
                    self.graph,
                    u,
                    v,
                    weight=lambda a, b, d: self._inv_idf_edge_weight(a, b, d, idf),
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                print("Pair (%s, %s): no path in graph — skipped.", u, v)
                continue

            hops = len(path) - 1
            if hops > 3:
                print(
                    "Pair (%s, %s): path length %d > 3 hops — skipped.", u, v, hops
                )
                continue

            intermediates = path[1:-1]
            if not intermediates:
                continue

            # Inject the single most specific (highest-IDF) intermediate.
            best = max(intermediates, key=lambda n: idf.get(n, 0.0))
            if best not in query_set:
                bridge_nodes.append(best)
                print(
                    "Pair (%s, %s): injecting bridge '%s' (IDF=%.4f, path=%s).",
                    u, v, best, idf.get(best, 0.0), path,
                )

        # Deduplicate while preserving insertion order.
        seen: set[str] = set()
        deduped: list[str] = []
        for n in bridge_nodes:
            if n not in seen:
                seen.add(n)
                deduped.append(n)
        return deduped

    # ── BM25 helpers ─────────────────────────────────────────────────────────

    def _bm25_scores(self, query: str) -> np.ndarray:
        """Return the raw BM25 score array for a token list."""
        tokenized = preprocess_for_bm25(query)
        return np.array(self.bm25_index.get_scores(tokenized), dtype=float)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        max_score = scores.max()
        if max_score <= 0:
            return scores
        return scores / max_score

    # ── Public interface ──────────────────────────────────────────────────────

    def get_scores(self, query: str, pool_size: int, chunks: List[str]) -> Dict[int, float]:
        """Return KG-filtered BM25 relevance scores keyed by chunk ID.

        Args:
            query:     Natural-language query string.
            pool_size: Maximum number of chunks to return.
            chunks:    The pipeline's full chunk list (used only for bounds checking).

        Returns:
            ``Dict[chunk_id, score]`` for up to *pool_size* chunks, scores in
            the range [0, 1].
        """
        core_tokens = extract_query_nodes(query, self.graph, self.canonical_lookup)
        print("Query: %r", query)
        print("Core tokens (%d): %s", len(core_tokens), core_tokens)

        # ── Fallback ─────────────────────────────────────────────────────────
        if len(core_tokens) < 2:
            print(
                "Fewer than 2 core tokens — falling back to standard BM25 on raw query."
            )
            raw = self._normalize(
                np.array(self.bm25_index.get_scores(preprocess_for_bm25(query)), dtype=float)
            )
            return self._top_k_from_array(raw, pool_size)

        # ── Expansion via bridge nodes ────────────────────────────────────────
        expansion_tokens = self._find_bridge_nodes(core_tokens)
        print("Expansion tokens (%d): %s", len(expansion_tokens), expansion_tokens)

        # ── Dual BM25 scoring ─────────────────────────────────────────────────
        core_scores = self._normalize(self._bm25_scores(query))
        expanded_scores = self._normalize(self._bm25_scores(" ".join(expansion_tokens)))

        final_scores = 0.7 * core_scores + 0.3 * expanded_scores

        print(
            "Score stats — core max: %.4f, expanded max: %.4f, final max: %.4f",
            core_scores.max(), expanded_scores.max(), final_scores.max(),
        )

        return self._top_k_from_array(final_scores, pool_size)

    def _top_k_from_array(self, scores: np.ndarray, pool_size: int) -> Dict[int, float]:
        """Convert a full-corpus score array to a {chunk_id: score} dict."""
        n = min(pool_size, len(scores))
        if n == 0:
            return {}
        top_indices = np.argpartition(-scores, kth=n - 1)[:n]
        result: Dict[int, float] = {}
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0 and int(idx) in self.kg_chunks:
                result[int(idx)] = score
        print(
            "Returning %d chunks (top-5): %s",
            len(result),
            dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:5]),
        )
        return result


# ── Self-contained smoke test ────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import pickle

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    from src.knowledge_graph.build import RUNS_DIR
    from src.knowledge_graph.io import (
        load_graph_and_chunks,
        load_canonicalization_data,
        resolve_run_dir,
    )
    from src.knowledge_graph.query import CanonicalLookup

    QUERY = "How does ARIES use write-ahead logging to ensure atomicity during crash recovery?"
    TOP_K = 3

    # ── Load KG artifacts ────────────────────────────────────────────────────
    run_dir = resolve_run_dir(RUNS_DIR)
    print(f"Run dir : {run_dir}")

    graph, kg_chunks = load_graph_and_chunks(run_dir)
    print(f"Graph   : {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Chunks  : {len(kg_chunks)}")

    syn_table, can_kw, can_emb = load_canonicalization_data(run_dir)
    canonical_lookup = (
        CanonicalLookup(syn_table, can_kw, can_emb) if syn_table is not None else None
    )

    # ── Load BM25 index ──────────────────────────────────────────────────────
    BM25_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "index", "sections", "textbook_index_bm25.pkl",
    )
    with open(BM25_PATH, "rb") as fh:
        bm25_index = pickle.load(fh)
    print(f"BM25    : {bm25_index.corpus_size} documents\n")

    # ── Build retriever and score ─────────────────────────────────────────────
    retriever = KGFilteredBM25Retriever(
        graph=graph,
        kg_chunks=kg_chunks,
        bm25_index=bm25_index,
        canonical_lookup=canonical_lookup,
    )

    chunk_list = list(kg_chunks.values())
    scores = retriever.get_scores(QUERY, pool_size=TOP_K, chunks=chunk_list)

    results = sorted(
        [(cid, kg_chunks[cid], score) for cid, score in scores.items() if cid in kg_chunks],
        key=lambda x: x[2],
        reverse=True,
    )[:TOP_K]

    print(f"\n{'='*70}")
    print(f"Query: {QUERY!r}")
    print(f"{'='*70}\n")
    for rank, (cid, text, score) in enumerate(results, 1):
        print(f"#{rank}  chunk_id={cid}  score={score:.4f}")
        print(f"    {text[:300].strip()}")
        print()
