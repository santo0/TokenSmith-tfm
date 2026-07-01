import argparse
import logging
import os

import networkx as nx

from src.knowledge_graph.analysis import extract_query_subgraph
from src.knowledge_graph.io import (
    RUNS_DIR,
    load_graph,
    load_canonicalization_data,
    load_keyword_index,
    build_keyword_index,
)
from src.knowledge_graph.query import (
    CanonicalLookup,
    extract_query_nodes,
    extract_query_nodes_embedding,
    extract_query_nodes_hybrid,
)

logger = logging.getLogger(__name__)


def _resolve_embed_model(run_dir: str, cli_override: str | None) -> str:
    if cli_override:
        return cli_override
    config_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(config_path):
        import json
        with open(config_path, encoding="utf-8") as f:
            rc = json.load(f)
        if "embed_model" in rc:
            return rc["embed_model"]
    return "sentence-transformers/all-MiniLM-L6-v2"


def expand_khop(subgraph_nodes: set[str], graph: nx.Graph, k: int) -> dict[str, int]:
    """Return nodes reachable within *k* hops from *subgraph_nodes* not already in the subgraph.

    Returns a mapping node → hop distance (1-indexed).
    """
    khop_nodes: dict[str, int] = {}
    frontier = set(subgraph_nodes)
    visited = set(subgraph_nodes)

    for hop in range(1, k + 1):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    khop_nodes[neighbor] = hop
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    return khop_nodes


def build_visualization_graph(
    query_nodes: list[str],
    subgraph: nx.Graph,
    khop_nodes: dict[str, int],
    full_graph: nx.Graph,
) -> tuple[nx.Graph, dict[str, str]]:
    """Merge the subgraph and k-hop neighbors into a single graph for rendering.

    Returns the combined graph and a node → role mapping:
      'query'   — matched query node
      'path'    — subgraph path/bridge node
      'khop_N'  — k-hop neighbor at distance N
    """
    combined = subgraph.copy()

    for node, hop in khop_nodes.items():
        combined.add_node(node, **full_graph.nodes[node])
        for neighbor in full_graph.neighbors(node):
            if combined.has_node(neighbor):
                combined.add_edge(node, neighbor, **full_graph[node][neighbor])

    roles: dict[str, str] = {}
    query_set = set(query_nodes)
    for node in combined.nodes:
        if node in query_set:
            roles[node] = "query"
        elif node in khop_nodes:
            roles[node] = f"khop_{khop_nodes[node]}"
        else:
            roles[node] = "path"

    return combined, roles


_ROLE_COLORS = {
    "query": "#e74c3c",   # red
    "path": "#3498db",    # blue
}
_KHOP_PALETTE = ["#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]


def _node_color(role: str) -> str:
    if role in _ROLE_COLORS:
        return _ROLE_COLORS[role]
    # khop_1, khop_2, …
    try:
        hop = int(role.split("_")[1])
        return _KHOP_PALETTE[(hop - 1) % len(_KHOP_PALETTE)]
    except (IndexError, ValueError):
        return "#95a5a6"


def visualize(
    combined: nx.Graph,
    roles: dict[str, str],
    query: str,
    output_path: str | None,
    figsize: tuple[int, int] = (16, 12),
) -> None:
    import matplotlib
    if output_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    node_colors = [_node_color(roles.get(n, "path")) for n in combined.nodes]
    edge_weights = [combined[u][v].get("weight", 1) for u, v in combined.edges]
    max_w = max(edge_weights, default=1)
    edge_widths = [0.5 + 3.0 * (w / max_w) for w in edge_weights]

    fig, ax = plt.subplots(figsize=figsize)

    pos = nx.spring_layout(combined, seed=42, k=1.5)
    nx.draw_networkx_edges(combined, pos, width=edge_widths, alpha=0.4, edge_color="#aaaaaa", ax=ax)
    nx.draw_networkx_nodes(combined, pos, node_color=node_colors, node_size=700, alpha=0.9, ax=ax)

    # Only label nodes with short labels to avoid clutter; truncate long ones
    labels = {n: (n if len(n) <= 20 else n[:18] + "…") for n in combined.nodes}
    nx.draw_networkx_labels(combined, pos, labels=labels, font_size=7, ax=ax)

    # Build legend — collect distinct roles
    seen_roles: dict[str, str] = {}
    for node, role in roles.items():
        if role not in seen_roles:
            seen_roles[role] = _node_color(role)
    legend_handles = [
        mpatches.Patch(color=color, label=role.replace("_", " "))
        for role, color in sorted(seen_roles.items())
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    ax.set_title(f'Query subgraph: "{query}"\n{combined.number_of_nodes()} nodes · {combined.number_of_edges()} edges', fontsize=11)
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        logger.info("Saved visualization to %s", output_path)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the query subgraph from a Knowledge Graph run."
    )
    parser.add_argument(
        "--run-dir",
        default=os.path.join(RUNS_DIR, "latest"),
    )
    parser.add_argument("--query", required=True, help="Query string to visualize.")
    parser.add_argument(
        "--khops",
        type=int,
        default=0,
        metavar="K",
        help="Expand visualization by K additional hops beyond the query subgraph (default: 0).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Save the figure to PATH (PNG/PDF/SVG). Displays interactively if omitted.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=int,
        default=[16, 12],
        metavar=("W", "H"),
        help="Figure size in inches (default: 16 12).",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        default=False,
        help="Use embedding-based query node extraction instead of n-gram matching.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        default=False,
        help="Use hybrid extraction: exact match first, then fill remaining slots "
             "with embedding results up to ceil(sqrt(num_query_words)).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        metavar="MODEL",
        help="SentenceTransformer model override for --use-embeddings. "
             "Defaults to the model recorded in config.json.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.4,
        metavar="T",
        help="Cosine similarity threshold for --use-embeddings (default: 0.78).",
    )
    parser.add_argument(
        "--top-k-keywords",
        type=int,
        default=10,
        metavar="K",
        help="FAISS neighbours to retrieve for --use-embeddings (default: 10).",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s %(name)s  %(levelname)s %(message)s")

    graph_path = os.path.join(args.run_dir, "graph.json")
    graph = load_graph(graph_path)
    logger.debug("Loaded graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

    synonym_table, canonical_keywords, canonical_embeddings = load_canonicalization_data(args.run_dir)

    canonical_lookup = None
    if synonym_table is not None:
        canonical_lookup = CanonicalLookup(synonym_table, canonical_keywords, canonical_embeddings)

    if args.use_embeddings or args.hybrid:
        keyword_index = load_keyword_index(args.run_dir)
        if keyword_index is None:
            logger.info("keyword_index.faiss not found — building lazily...")
            if canonical_embeddings is None:
                print("Error: canonicalization data not found; cannot build keyword index.")
                return
            keyword_index = build_keyword_index(canonical_embeddings, args.run_dir)
            print("Keyword index built and saved.")
        embed_model = _resolve_embed_model(args.run_dir, args.embedding_model)
        logger.info("Using embedding model: %s", embed_model)

        if args.hybrid:
            query_nodes = extract_query_nodes_hybrid(
                args.query, graph, keyword_index, canonical_keywords,
                canonical_lookup=canonical_lookup,
                embedding_model=embed_model,
                embedding_top_k=args.top_k_keywords,
                embedding_threshold=args.similarity_threshold,
            )
        else:
            query_nodes = extract_query_nodes_embedding(
                args.query, graph, keyword_index, canonical_keywords,
                embedding_model=embed_model,
                top_k=args.top_k_keywords,
                similarity_threshold=args.similarity_threshold,
            )
    else:
        query_nodes = extract_query_nodes(args.query, graph, canonical_lookup)

    if not query_nodes:
        print("No query nodes matched in the graph. Check your query or run directory.")
        return

    print(f"Matched query nodes ({len(query_nodes)}): {query_nodes}")

    subgraph = extract_query_subgraph(query_nodes, graph)
    logger.debug("Subgraph: %d nodes, %d edges", subgraph.number_of_nodes(), subgraph.number_of_edges())

    khop_nodes: dict[str, int] = {}
    if args.khops > 0:
        khop_nodes = expand_khop(set(subgraph.nodes), graph, args.khops)
        print(f"K-hop expansion (k={args.khops}): {len(khop_nodes)} additional nodes")

    combined, roles = build_visualization_graph(query_nodes, subgraph, khop_nodes, graph)
    print(f"Visualization graph: {combined.number_of_nodes()} nodes, {combined.number_of_edges()} edges")

    visualize(combined, roles, args.query, args.output, tuple(args.figsize))


if __name__ == "__main__":
    main()
