import logging
import math
from collections import Counter
from itertools import combinations

import networkx as nx
import numpy as np

from src.knowledge_graph.models import (
    Chunk,
    DifficultyCategory,
    DifficultyComponents,
    DifficultyScore,
    QueryAnalysisResult,
    QueryFeatures,
)
from src.knowledge_graph.ngrams import KW_PATTERN, extract_ngrams
from src.knowledge_graph.normalizer import Normalizer
from src.knowledge_graph.query import CanonicalLookup, extract_query_nodes

logger = logging.getLogger(__name__)

_normalizer = Normalizer()


# ---------------------------------------------------------------------------
# Subgraph extraction (shared by D1 and D4)
# ---------------------------------------------------------------------------

def extract_query_subgraph(query_nodes: list[str], graph: nx.Graph) -> nx.Graph:
    """Return the subgraph spanning *query_nodes* and the shortest paths between them."""
    subgraph_nodes = set(query_nodes)
    for u, v in combinations(query_nodes, 2):
        if nx.has_path(graph, u, v):
            try:
                path = nx.shortest_path(graph, u, v)
                subgraph_nodes.update(path)
            except nx.NetworkXNoPath:
                pass
    return graph.subgraph(subgraph_nodes).copy()


# ---------------------------------------------------------------------------
# D1 — Corpus Coverage
# ---------------------------------------------------------------------------

def _compute_d1_coverage(
    query: str,
    graph: nx.Graph,
    canonical_lookup: CanonicalLookup | None,
) -> tuple[float, int, int]:
    """δ_cov = 1 − (matched_concepts / total_concepts). Returns (δ_cov, total, matched)."""
    terms = extract_ngrams(query, KW_PATTERN)
    normalized = _normalizer.normalize(terms)

    if canonical_lookup is not None:
        resolved = {canonical_lookup.resolve(t) for t in normalized}
    else:
        resolved = set(normalized)

    total = len(resolved)
    if total == 0:
        return 0.0, 0, 0

    matched = sum(1 for t in resolved if graph.has_node(t))
    delta_cov = 1.0 - matched / total
    return delta_cov, total, matched


# ---------------------------------------------------------------------------
# D2 — Retrieval Confidence
# ---------------------------------------------------------------------------

def _compute_d2_confidence(
    retrieval_scores: dict[int, float] | None,
    top_k: int = 10,
) -> float:
    """δ_conf = normalized Shannon entropy of softmax(top-k scores). Range [0, 1]."""
    if not retrieval_scores:
        return 0.0

    scores = sorted(retrieval_scores.values(), reverse=True)[:top_k]
    k = len(scores)
    if k < 2:
        return 0.0

    arr = np.array(scores, dtype=float)
    arr -= arr.max()  # numerical stability
    exp_arr = np.exp(arr)
    probs = exp_arr / exp_arr.sum()

    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return entropy / math.log(k)


# ---------------------------------------------------------------------------
# D3 — Context Capacity
# ---------------------------------------------------------------------------

def _compute_d3_capacity(
    retrieval_scores: dict[int, float] | None,
    chunks: list[Chunk] | None,
    context_window: int,
    threshold: float,
) -> tuple[float, int]:
    """δ_cap = max(0, total_tokens/context_window − 1). Returns (δ_cap, total_tokens)."""
    if not retrieval_scores or not chunks:
        return 0.0, 0

    chunk_map = {c.id: c for c in chunks}
    total_tokens = sum(
        len(chunk_map[cid].text.split())
        for cid, score in retrieval_scores.items()
        if score >= threshold and cid in chunk_map
    )
    delta_cap = max(0.0, total_tokens / context_window - 1.0)
    return delta_cap, total_tokens


# ---------------------------------------------------------------------------
# D4 — Topological Complexity
# ---------------------------------------------------------------------------

def _compute_d4_topology(
    query_nodes: list[str],
    graph: nx.Graph,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    max_components: float = 10.0,
    max_diameter: float = 10.0,
) -> tuple[float, int, float, float]:
    """δ_top ∈ [0, 1] from κ (components), ρ (density), diam. Returns (δ_top, κ, ρ, diam)."""
    if not query_nodes:
        return 0.0, 0, 0.0, 0.0

    subgraph = extract_query_subgraph(query_nodes, graph)
    n_nodes = subgraph.number_of_nodes()
    if n_nodes == 0:
        return 0.0, 0, 0.0, 0.0

    kappa = nx.number_connected_components(subgraph)
    rho = nx.density(subgraph)

    # Diameter of the largest connected component
    largest_cc = max(nx.connected_components(subgraph), key=len)
    cc_sub = subgraph.subgraph(largest_cc)
    if cc_sub.number_of_nodes() > 1:
        try:
            diam = float(nx.diameter(cc_sub))
        except nx.NetworkXError:
            diam = 0.0
    else:
        diam = 0.0

    w_k, w_r, w_d = weights
    kappa_norm = min(kappa / max_components, 1.0)
    diam_norm = min(diam / max_diameter, 1.0)
    cohesion_inv = 1.0 - rho  # high density = easy (inverted)

    delta_top = w_k * kappa_norm + w_r * cohesion_inv + w_d * diam_norm
    return delta_top, kappa, rho, diam


# ---------------------------------------------------------------------------
# D5 — Community Dispersion
# ---------------------------------------------------------------------------

def _get_or_compute_communities(graph: nx.Graph) -> dict[str, int]:
    """Return node→community_id map. Lazily runs Leiden if not cached on graph."""
    # Check if community IDs are already stored as node attributes
    first = next(iter(graph.nodes(data=True)), None)
    if first is not None and "community" in first[1]:
        return {n: d["community"] for n, d in graph.nodes(data=True)}

    try:
        from src.knowledge_graph.scripts.leiden_communities import (
            _check_imports,
            _nx_to_igraph,
            _run_leiden,
        )
        _check_imports()
        ig_graph, node_labels = _nx_to_igraph(graph)
        membership, modularity = _run_leiden(ig_graph, resolution=1.0, seed=42)
        n_comms = len(set(membership))
        logger.debug("Leiden: %d communities (modularity=%.4f)", n_comms, modularity)
        community_map = {node_labels[i]: membership[i] for i in range(len(node_labels))}
        # Cache on the graph so subsequent calls skip recomputation
        for node, comm_id in community_map.items():
            graph.nodes[node]["community"] = comm_id
        return community_map
    except Exception as exc:
        logger.warning("Community detection failed, D5 will be 0: %s", exc)
        return {}


def _compute_d5_community(
    query_nodes: list[str],
    graph: nx.Graph,
    community_map: dict[str, int] | None,
) -> tuple[float, int]:
    """δ_com = normalized entropy of community distribution. Returns (δ_com, n_unique_communities)."""
    if not query_nodes:
        return 0.0, 0

    if community_map is None:
        community_map = _get_or_compute_communities(graph)

    if not community_map:
        return 0.0, 0

    node_communities = [community_map[n] for n in query_nodes if n in community_map]
    if not node_communities:
        return 0.0, 0

    unique_comms = set(node_communities)
    n_unique = len(unique_comms)
    if n_unique <= 1:
        return 0.0, n_unique

    counts = Counter(node_communities)
    total = len(node_communities)
    probs = np.array([v / total for v in counts.values()], dtype=float)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    delta_com = entropy / math.log(n_unique)
    return delta_com, n_unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_difficulty_features(
    query: str,
    graph: nx.Graph,
    canonical_lookup: CanonicalLookup,
    retrieval_scores: dict[int, float] | None = None,
    chunks: list[Chunk] | None = None,
    context_window: int = 8192,
    relevance_threshold: float = 0.3,
    top_k_conf: int = 10,
    community_map: dict[str, int] | None = None,
) -> QueryFeatures:
    """Compute all five difficulty features for *query*."""
    query_nodes = extract_query_nodes(query, graph, canonical_lookup)
    logger.debug("Query nodes: %s", query_nodes)

    delta_cov, total_concepts, matched_concepts = _compute_d1_coverage(
        query, graph, canonical_lookup
    )
    delta_conf = _compute_d2_confidence(retrieval_scores, top_k=top_k_conf)
    delta_cap, relevant_tokens = _compute_d3_capacity(
        retrieval_scores, chunks, context_window, relevance_threshold
    )
    delta_top, kappa, rho, diam = _compute_d4_topology(query_nodes, graph)
    delta_com, n_communities = _compute_d5_community(query_nodes, graph, community_map)

    subgraph = extract_query_subgraph(query_nodes, graph) if query_nodes else nx.Graph()

    return QueryFeatures(
        query_concept_count=total_concepts,
        matched_concept_count=matched_concepts,
        corpus_coverage=round(delta_cov, 6),
        retrieval_confidence=round(delta_conf, 6),
        relevant_token_count=relevant_tokens,
        context_capacity=round(delta_cap, 6),
        component_count=kappa,
        edge_density=round(rho, 6),
        subgraph_diameter=diam,
        topological_complexity=round(delta_top, 6),
        subgraph_node_count=subgraph.number_of_nodes(),
        subgraph_edge_count=subgraph.number_of_edges(),
        community_count=n_communities,
        community_dispersion=round(delta_com, 6),
    )


def compute_difficulty_score(
    features: QueryFeatures,
    weights: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2),
    category_thresholds: tuple[float, float] = (0.33, 0.67),
) -> DifficultyScore:
    """Compute the composite difficulty score D(q) = Σ wᵢ·δᵢ."""
    w1, w2, w3, w4, w5 = weights
    # δ_cap is clamped to 1 for composite scoring (it can exceed 1 by design)
    cap_clamped = min(features.context_capacity, 1.0)
    score = (
        w1 * features.corpus_coverage
        + w2 * features.retrieval_confidence
        + w3 * cap_clamped
        + w4 * features.topological_complexity
        + w5 * features.community_dispersion
    )
    score = round(score, 6)

    easy_max, medium_max = category_thresholds
    if score <= easy_max:
        category = DifficultyCategory.EASY
    elif score <= medium_max:
        category = DifficultyCategory.MEDIUM
    else:
        category = DifficultyCategory.HARD

    return DifficultyScore(
        score=score,
        category=category,
        components=DifficultyComponents(
            corpus_coverage=features.corpus_coverage,
            retrieval_confidence=features.retrieval_confidence,
            context_capacity=features.context_capacity,
            topological_complexity=features.topological_complexity,
            community_dispersion=features.community_dispersion,
        ),
    )


def analyze_query(
    query: str,
    graph: nx.Graph,
    canonical_lookup: CanonicalLookup,
    retrieval_scores: dict[int, float] | None = None,
    chunks: list[Chunk] | None = None,
    context_window: int = 8192,
    relevance_threshold: float = 0.3,
    top_k_conf: int = 10,
    community_map: dict[str, int] | None = None,
    weights: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2),
    category_thresholds: tuple[float, float] = (0.33, 0.67),
) -> QueryAnalysisResult:
    """Run the full difficulty analysis pipeline for *query*."""
    features = compute_difficulty_features(
        query,
        graph,
        canonical_lookup,
        retrieval_scores=retrieval_scores,
        chunks=chunks,
        context_window=context_window,
        relevance_threshold=relevance_threshold,
        top_k_conf=top_k_conf,
        community_map=community_map,
    )
    difficulty = compute_difficulty_score(
        features,
        weights=weights,
        category_thresholds=category_thresholds,
    )
    return QueryAnalysisResult(query=query, features=features, difficulty=difficulty)
