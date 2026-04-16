import json
import os

import networkx as nx
import numpy as np

from src.knowledge_graph.persisters import BasePersister
from src.knowledge_graph.models import Chunk, RunMetadata

from src.knowledge_graph.canonicalizer import CanonicalizationResult


class NetworkxJsonPersister(BasePersister):
    """Save the graph in NetworkX node-link JSON format and chunks as a
    separate JSON dictionary.

    The caller is responsible for creating a timestamped run directory and
    passing it as ``output_dir``.  This persister writes fixed filenames into
    that directory so the directory itself carries the run identity:

    * ``graph.json``               — NetworkX node-link serialization (default filename)
    * ``chunks.json``              — ``{ "0": "chunk text …", "1": "…" }``
    * ``run_metadata.json``        — timing + graph statistics (optional)
    * ``synonym_table.json``       — keyword → canonical mapping (if canonicalized)
    * ``canonical_keywords.json``  — sorted list of canonical forms (if canonicalized)
    * ``canonical_embeddings.npy`` — embedding matrix for canonical keywords (if canonicalized)
    * ``semantic_triples.json``    — flat triple list for inspection (if provided)

    Args:
        graph_filename: Override the output filename for the graph JSON.
            Useful when saving a semantic graph alongside a cooccurrence graph
            (e.g. ``"semantic_graph.json"``).
    """

    def __init__(self, graph_filename: str = "graph.json"):
        super().__init__()
        self.graph_filename = graph_filename

    def persist(
        self,
        graph: nx.Graph,
        chunks: list[Chunk],
        output_dir: str,
        run_metadata: RunMetadata | None = None,
        canonicalization_result: CanonicalizationResult | None = None,
        semantic_triples: list[dict] | None = None,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)

        # --- graph file (graph.json or semantic_graph.json) ---
        graph_data = nx.node_link_data(graph)
        with open(os.path.join(output_dir, self.graph_filename), "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        # --- chunks.json ---
        chunk_store = {str(chunk.id): chunk.text for chunk in chunks}
        with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunk_store, f, indent=2, ensure_ascii=False)

        # --- canonicalization artifacts ---
        if canonicalization_result is not None:
            with open(
                os.path.join(output_dir, "synonym_table.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(canonicalization_result.synonym_table, f, indent=2, ensure_ascii=False)

            with open(
                os.path.join(output_dir, "canonical_keywords.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(canonicalization_result.canonical_keywords, f, indent=2, ensure_ascii=False)

            np.save(
                os.path.join(output_dir, "canonical_embeddings.npy"),
                canonicalization_result.canonical_embeddings,
            )

        # --- semantic_triples.json (optional, for inspection/debug) ---
        if semantic_triples is not None:
            with open(
                os.path.join(output_dir, "semantic_triples.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(semantic_triples, f, indent=2, ensure_ascii=False)

        # --- run_metadata.json ---
        if run_metadata:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            # Use weakly_connected_components for directed graphs, connected_components for undirected
            if graph.is_directed():
                comp_list = list(nx.weakly_connected_components(graph))
                avg_degree = (num_edges / num_nodes) if num_nodes > 0 else 0.0
                # average_clustering is not defined for MultiDiGraph; skip gracefully
                try:
                    avg_clustering = nx.average_clustering(graph)
                except Exception:
                    avg_clustering = None
            else:
                comp_list = list(nx.connected_components(graph))
                avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0.0
                avg_clustering = nx.average_clustering(graph)

            largest_comp_size = len(max(comp_list, key=len)) if comp_list else 0

            graph_stats: dict = {
                "nodes": num_nodes,
                "edges": num_edges,
                "density": nx.density(graph),
                "avg_degree": avg_degree,
                "num_connected_components": len(comp_list),
                "largest_component_size": largest_comp_size,
                "max_degree": max(dict(graph.degree()).values(), default=0),
            }
            if avg_clustering is not None:
                graph_stats["avg_clustering"] = avg_clustering

            run_metadata.statistics["graph"] = graph_stats
            with open(
                os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(run_metadata.to_dict(), f, indent=2, ensure_ascii=False)
