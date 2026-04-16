"""Semantic KG retriever and optional query intent classifier."""

import json
import logging
import re
from typing import Any

import networkx as nx

from src.retriever import Retriever
from src.knowledge_graph.query import CanonicalLookup, extract_query_nodes
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.utils.semantic_prompts import (
    DEFAULT_RELATION_WEIGHTS,
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_USER_TEMPLATE,
    RELATION_TYPES,
    _RELATION_LIST_STR,
)

logger = logging.getLogger(__name__)


class QueryIntentClassifier:
    """Classify a query's intent as a list of relevant relation types.

    Uses a lightweight LLM call to map a query string to 1-3 relation types
    from the semantic KG taxonomy.  Results are cached in-memory so repeated
    queries within a session incur only one LLM call.

    Args:
        api_key: OpenRouter API key.
        llm_model: OpenRouter model identifier.
        retries: Extra LLM retries on failure.
    """

    def __init__(
        self,
        api_key: str,
        llm_model: str = "openai/gpt-4o-mini",
        retries: int = 1,
    ):
        self._client = OpenRouterClient(api_key, retries=retries)
        self.llm_model = llm_model
        self._cache: dict[str, list[str]] = {}

    def classify(self, query: str) -> list[str]:
        """Return a list of relation type strings relevant to *query*.

        Falls back to an empty list (no intent boost applied) on any error.
        """
        query_key = query.strip().lower()
        if query_key in self._cache:
            return self._cache[query_key]

        user = INTENT_CLASSIFICATION_USER_TEMPLATE.format(
            relation_list=_RELATION_LIST_STR,
            query=query,
        )
        try:
            raw = self._client.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": INTENT_CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            data = json.loads(raw)
            intent_relations = [
                r for r in data.get("intent_relations", []) if r in RELATION_TYPES
            ]
            logger.debug("Intent classification for %r: %s", query, intent_relations)
            self._cache[query_key] = intent_relations
            return intent_relations
        except Exception as exc:
            logger.warning("QueryIntentClassifier: failed — %s", exc)
            self._cache[query_key] = []
            return []


class SemanticKGRetriever(Retriever):
    """Knowledge-graph retriever over a typed directed multi-graph.

    Scores chunks by traversing the semantic graph from matched query nodes,
    weighting edges by their relation type and direction.

    **Scoring algorithm:**

    1. Match query n-grams to graph nodes (reuses :func:`extract_query_nodes`).
    2. Hop 0: direct matches contribute ``+1.0`` per chunk_id.
    3. Hops 1..num_hops: BFS over both outgoing (full relation weight) and
       incoming (``backward_factor × relation_weight``) edges.
    4. Per-edge contribution:
       ``decay^hop × rel_weight × (edge_weight / max_edge_weight)``
       where ``rel_weight`` is boosted by ``intent_boost`` if the relation type
       matches the classified query intent.
    5. Scores normalized to [0, 1].

    Args:
        graph: Semantic :class:`~networkx.MultiDiGraph` from :class:`SemanticLinker`.
        kg_chunks: Mapping of ``chunk_id → text`` (from the KG run).
        canonical_lookup: Optional synonym resolver for query terms.
        neighbor_weight: Exponential decay per hop (default 0.5).
        num_hops: BFS depth (default 2).
        relation_weights: Override default per-relation-type weights.  Empty
            dict → use :data:`DEFAULT_RELATION_WEIGHTS`.
        use_intent_classification: If True, classify query intent with an LLM
            and boost matching relation types.
        intent_classifier: Pre-built :class:`QueryIntentClassifier`.  Required
            when ``use_intent_classification=True``.
        intent_boost: Multiplicative boost applied to edges whose relation type
            matches the classified intent (default 1.2).
        backward_factor: Down-weight applied when traversing an edge in reverse
            (incoming direction, default 0.6).
    """

    name = "kg_semantic"

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        kg_chunks: dict[int, str],
        canonical_lookup: CanonicalLookup | None = None,
        neighbor_weight: float = 0.5,
        num_hops: int = 2,
        relation_weights: dict[str, float] | None = None,
        use_intent_classification: bool = False,
        intent_classifier: QueryIntentClassifier | None = None,
        intent_boost: float = 1.2,
        backward_factor: float = 0.6,
    ):
        self.graph = graph
        self.kg_chunks = kg_chunks
        self.canonical_lookup = canonical_lookup
        self.neighbor_weight = neighbor_weight
        self.num_hops = num_hops
        self.relation_weights: dict[str, float] = (
            relation_weights if relation_weights else dict(DEFAULT_RELATION_WEIGHTS)
        )
        self.use_intent_classification = use_intent_classification
        self.intent_classifier = intent_classifier
        self.intent_boost = intent_boost
        self.backward_factor = backward_factor

    def get_scores(self, query: str, pool_size: int, chunks: list) -> dict[int, float]:
        """Return relation-aware BFS scores keyed by chunk index.

        Args:
            query:     Natural-language query string.
            pool_size: Maximum number of chunks to return scores for.
            chunks:    The RAG pipeline's chunk list (used only for length).

        Returns:
            ``Dict[chunk_id, score]`` normalized to [0, 1].
            Returns an empty dict if no query nodes match the graph.
        """
        query_nodes = extract_query_nodes(query, self.graph, self.canonical_lookup)
        logger.debug("SemanticKGRetriever query: %r", query)
        logger.debug("Matched query nodes (%d): %s", len(query_nodes), query_nodes)
        if not query_nodes:
            return {}

        # Classify intent for optional relation-type boosting
        active_relations: set[str] = set()
        if self.use_intent_classification and self.intent_classifier is not None:
            active_relations = set(self.intent_classifier.classify(query))
            logger.debug("Active intent relations: %s", active_relations)

        # Find the maximum edge weight across the entire graph for normalization
        max_edge_weight = max(
            (data.get("weight", 1) for _, _, data in self.graph.edges(data=True)),
            default=1,
        )
        max_edge_weight = max(max_edge_weight, 1)

        scores: dict[int, float] = {}

        # Hop 0: direct matches
        for node in query_nodes:
            for chunk_id in self.graph.nodes[node].get("chunk_ids", []):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0

        # BFS over hops 1..num_hops
        visited: set[str] = set(query_nodes)
        frontier: set[str] = set(query_nodes)

        for hop in range(1, self.num_hops + 1):
            decay = self.neighbor_weight ** hop
            next_frontier: set[str] = set()

            for node in frontier:
                # Outgoing edges: full relation weight
                for _, neighbor, data in self.graph.out_edges(node, data=True):
                    if neighbor in visited:
                        continue
                    next_frontier.add(neighbor)
                    contribution = self._edge_contribution(
                        data, decay, max_edge_weight, active_relations, forward=True
                    )
                    for chunk_id in self.graph.nodes[neighbor].get("chunk_ids", []):
                        scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution

                # Incoming edges: down-weighted by backward_factor
                for neighbor, _, data in self.graph.in_edges(node, data=True):
                    if neighbor in visited:
                        continue
                    next_frontier.add(neighbor)
                    contribution = self._edge_contribution(
                        data, decay, max_edge_weight, active_relations, forward=False
                    )
                    for chunk_id in self.graph.nodes[neighbor].get("chunk_ids", []):
                        scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution

            visited |= next_frontier
            frontier = next_frontier
            logger.debug("Hop %d: %d new node(s).", hop, len(next_frontier))
            if not frontier:
                break

        if not scores:
            return {}

        max_score = max(scores.values())
        if max_score <= 0:
            return {}

        return {cid: s / max_score for cid, s in scores.items()}

    def _edge_contribution(
        self,
        edge_data: dict[str, Any],
        decay: float,
        max_edge_weight: int,
        active_relations: set[str],
        forward: bool,
    ) -> float:
        """Compute the score contribution of a single edge traversal."""
        relation = edge_data.get("relation", "")
        edge_weight = edge_data.get("weight", 1)

        rel_weight = self.relation_weights.get(relation, 0.5)
        if active_relations and relation in active_relations:
            rel_weight = min(rel_weight * self.intent_boost, 1.0)

        direction_factor = 1.0 if forward else self.backward_factor
        return decay * rel_weight * direction_factor * (edge_weight / max_edge_weight)
