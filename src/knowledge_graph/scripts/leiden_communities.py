"""Hierarchical Leiden community detection for the Knowledge Graph.

Algorithm
---------
Level 0 : Run Leiden on the full graph → partition P0
Level k  : Contract G by P(k-1) (communities → super-nodes, weights summed),
           run Leiden on the contracted graph → Pk.
Repeat until a single community remains or *max_levels* is reached.

Dependencies (not in environment.yml):
    pip install python-igraph leidenalg

Usage
-----
    python -m knowledge_graph.scripts.leiden_communities \\
        --run-dir data/knowledge_graph/runs/latest \\
        --max-levels 4 \\
        --resolution 1.0 \\
        --output communities.json \\
        --visualize communities.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from src.knowledge_graph.build import RUNS_DIR
from src.knowledge_graph.io import load_graph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LevelResult:
    level: int
    num_communities: int
    modularity: float
    # community_id -> sorted list of original node labels
    communities: dict[int, list[str]] = field(default_factory=dict)
    # original node label -> community_id at this level
    node_to_community: dict[str, int] = field(default_factory=dict)


@dataclass
class HierarchyResult:
    num_levels: int
    levels: list[LevelResult]
    # original node -> community IDs across levels [level0, level1, …]
    node_path: dict[str, list[int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _check_imports() -> None:
    missing = []
    try:
        import igraph  # noqa: F401
    except ImportError:
        missing.append("python-igraph")
    try:
        import leidenalg  # noqa: F401
    except ImportError:
        missing.append("leidenalg")
    if missing:
        raise SystemExit(
            f"Missing dependencies: {', '.join(missing)}\n"
            "Install with:  pip install python-igraph leidenalg"
        )


# ---------------------------------------------------------------------------
# NetworkX → igraph conversion
# ---------------------------------------------------------------------------

def _nx_to_igraph(G: nx.Graph) -> tuple:
    """Convert a weighted NetworkX graph to igraph.

    Returns (ig_graph, node_labels) where node_labels[i] is the label of
    igraph vertex i (may be a string or int depending on contraction level).
    """
    import igraph as ig

    nodes = list(G.nodes())
    index = {n: i for i, n in enumerate(nodes)}

    edges = [(index[u], index[v]) for u, v in G.edges()]
    weights = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]

    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_graph.vs["label"] = nodes
    ig_graph.es["weight"] = weights if weights else [1.0] * len(edges)
    return ig_graph, nodes


# ---------------------------------------------------------------------------
# Single Leiden pass
# ---------------------------------------------------------------------------

def _run_leiden(
    ig_graph,
    resolution: float,
    seed: int,
) -> tuple[list[int], float]:
    """Run one Leiden pass.

    Uses RBConfigurationVertexPartition which supports a resolution parameter:
      - resolution > 1 → smaller, more fine-grained communities
      - resolution < 1 → larger, coarser communities
      - resolution = 1 → equivalent to standard modularity maximisation

    Returns (membership, modularity) where membership[i] is the community id
    of igraph vertex i.
    """
    import leidenalg

    weights = ig_graph.es["weight"] if ig_graph.ecount() > 0 else None
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        n_iterations=-1,    # iterate until the partition is stable
        seed=seed,
    )
    return partition.membership, partition.modularity


# ---------------------------------------------------------------------------
# Graph contraction
# ---------------------------------------------------------------------------

def _contract(G: nx.Graph, super_to_comm: dict) -> nx.Graph:
    """Contract *G* by mapping each node (super-node) to its community.

    The contracted graph has one node per community. Inter-community edge
    weights are summed; intra-community edges are dropped.
    """
    agg: dict[tuple, float] = defaultdict(float)
    for u, v, data in G.edges(data=True):
        cu, cv = super_to_comm[u], super_to_comm[v]
        if cu != cv:
            agg[(min(cu, cv), max(cu, cv))] += float(data.get("weight", 1.0))

    CG = nx.Graph()
    CG.add_nodes_from(set(super_to_comm.values()))
    for (cu, cv), w in agg.items():
        CG.add_edge(cu, cv, weight=w)
    return CG


# ---------------------------------------------------------------------------
# Hierarchical Leiden
# ---------------------------------------------------------------------------

def hierarchical_leiden(
    G: nx.Graph,
    max_levels: int = 5,
    resolution: float = 1.0,
    seed: int = 42,
) -> HierarchyResult:
    """Run Hierarchical Leiden on *G*.

    At each level the graph is contracted by the previous partition and Leiden
    is applied again.  Stops early when a single community is found or
    *max_levels* is reached.
    """
    _check_imports()

    original_nodes = list(G.nodes())
    levels: list[LevelResult] = []

    current_G = G
    # Maps current-graph super-node → list of original node labels it contains
    super_to_originals: dict = {n: [n] for n in original_nodes}

    for lvl in range(max_levels):
        logger.info(
            "Level %d — %d nodes, %d edges",
            lvl, current_G.number_of_nodes(), current_G.number_of_edges(),
        )

        ig_graph, ig_labels = _nx_to_igraph(current_G)
        membership, modularity = _run_leiden(ig_graph, resolution, seed + lvl)

        # Map super-node → community id at this level
        super_to_comm: dict = {ig_labels[i]: membership[i] for i in range(len(ig_labels))}

        # Propagate back to original nodes
        node_to_community: dict[str, int] = {}
        for super_node, originals in super_to_originals.items():
            comm = super_to_comm[super_node]
            for orig in originals:
                node_to_community[orig] = comm

        communities: dict[int, list[str]] = defaultdict(list)
        for n, c in node_to_community.items():
            communities[c].append(n)
        communities = {c: sorted(ns) for c, ns in communities.items()}

        num_communities = len(communities)
        levels.append(LevelResult(
            level=lvl,
            num_communities=num_communities,
            modularity=modularity,
            communities=communities,
            node_to_community=node_to_community,
        ))
        logger.info("  → %d communities, modularity=%.4f", num_communities, modularity)

        if num_communities <= 1:
            logger.info("Single community reached — stopping.")
            break

        # Contract graph for the next level
        current_G = _contract(current_G, super_to_comm)

        # Update super_to_originals: new super-node = community id
        new_super: dict = defaultdict(list)
        for super_node, originals in super_to_originals.items():
            new_super[super_to_comm[super_node]].extend(originals)
        super_to_originals = dict(new_super)

        # If the contracted graph has no fewer nodes, further merging is impossible
        if current_G.number_of_nodes() >= num_communities:
            logger.info("No further merging possible — stopping.")
            break

    # Build per-node ancestry path [comm_level0, comm_level1, …]
    node_path: dict[str, list[int]] = {n: [] for n in original_nodes}
    for lvl_result in levels:
        for n in original_nodes:
            node_path[n].append(lvl_result.node_to_community[n])

    return HierarchyResult(num_levels=len(levels), levels=levels, node_path=node_path)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_results(result: HierarchyResult, path: str) -> None:
    payload = {
        "num_levels": result.num_levels,
        "levels": [
            {
                "level": lvl.level,
                "num_communities": lvl.num_communities,
                "modularity": round(lvl.modularity, 6),
                "communities": {
                    str(cid): nodes
                    for cid, nodes in sorted(lvl.communities.items())
                },
                "node_to_community": lvl.node_to_community,
            }
            for lvl in result.levels
        ],
        "node_path": result.node_path,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Community results saved → {path}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(result: HierarchyResult) -> None:
    print(f"\n{'='*62}")
    print(f"  Hierarchical Leiden  —  {result.num_levels} level(s)")
    print(f"{'='*62}")
    for lvl in result.levels:
        sizes = sorted((len(ns) for ns in lvl.communities.values()), reverse=True)
        singletons = sum(1 for s in sizes if s == 1)
        median = sizes[len(sizes) // 2]
        print(
            f"\nLevel {lvl.level}:  {lvl.num_communities} communities  "
            f"modularity={lvl.modularity:.4f}"
        )
        print(
            f"  sizes — max:{sizes[0]}  min:{sizes[-1]}  "
            f"median:{median}  singletons:{singletons}"
        )
        top = sorted(lvl.communities.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
        for cid, nodes in top:
            sample = ", ".join(nodes[:6])
            more = f"  …+{len(nodes)-6}" if len(nodes) > 6 else ""
            print(f"    [{cid:>4}] {len(nodes):>4d} nodes:  {sample}{more}")
    print()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_communities(
    G: nx.Graph,
    level_result: LevelResult,
    output_path: str | None,
    figsize: tuple[int, int] = (18, 14),
    max_nodes: int = 300,
) -> None:
    """Draw the graph coloured by community membership at *level_result*."""
    import matplotlib
    if output_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm

    subgraph = G
    node_to_comm = level_result.node_to_community

    if G.number_of_nodes() > max_nodes:
        logger.warning("Graph has %d nodes — sampling top communities for visualization.",
                       G.number_of_nodes())
        top = sorted(level_result.communities.items(), key=lambda kv: len(kv[1]), reverse=True)
        keep: set[str] = set()
        per = max(1, max_nodes // min(len(top), 15))
        for _, nodes in top[:15]:
            keep.update(nodes[:per])
        subgraph = G.subgraph(keep).copy()
        node_to_comm = {n: level_result.node_to_community[n] for n in subgraph.nodes()}

    community_ids = sorted(set(node_to_comm.values()))
    cmap = cm.get_cmap("tab20", max(len(community_ids), 1))
    comm_color = {cid: cmap(i) for i, cid in enumerate(community_ids)}

    node_colors = [comm_color[node_to_comm[n]] for n in subgraph.nodes()]
    edge_weights = [subgraph[u][v].get("weight", 1) for u, v in subgraph.edges()]
    max_w = max(edge_weights, default=1)
    edge_widths = [0.3 + 2.5 * (w / max_w) for w in edge_weights]

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(subgraph, seed=42, k=1.2)

    nx.draw_networkx_edges(subgraph, pos, width=edge_widths, alpha=0.3,
                           edge_color="#aaaaaa", ax=ax)
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                           node_size=140, alpha=0.88, ax=ax)

    if subgraph.number_of_nodes() <= 80:
        labels = {n: (n if len(n) <= 18 else n[:16] + "…") for n in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=6, ax=ax)

    patches = [
        mpatches.Patch(color=comm_color[cid], label=f"Community {cid}")
        for cid in community_ids[:20]
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=7,
              ncol=2 if len(patches) > 10 else 1)
    ax.set_title(
        f"Leiden communities — Level {level_result.level}\n"
        f"{subgraph.number_of_nodes()} nodes shown · "
        f"{level_result.num_communities} communities · "
        f"modularity={level_result.modularity:.4f}",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Visualization saved → {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical Leiden community detection on the Knowledge Graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        default=os.path.join(RUNS_DIR, "latest"),
        help="KG run directory (or parent runs/ dir with a 'latest' symlink).",
    )
    parser.add_argument(
        "--max-levels", type=int, default=5, metavar="N",
        help="Maximum number of hierarchy levels.",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.5,
        help="Resolution parameter (>1 → finer communities, <1 → coarser).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--largest-component", action="store_true",
        help="Run only on the largest connected component.",
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Save community assignments as JSON to PATH.",
    )
    parser.add_argument(
        "--visualize", default=None, metavar="PATH",
        help="Save a community membership visualization PNG to PATH.",
    )
    parser.add_argument(
        "--vis-level", type=int, default=0,
        help="Hierarchy level to visualize (0 = finest).",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    # Resolve graph path
    run_dir = args.run_dir
    graph_path = os.path.join(run_dir, "graph.json")
    if not os.path.isfile(graph_path):
        latest = os.path.realpath(os.path.join(run_dir, "latest"))
        graph_path = os.path.join(latest, "graph.json")
        run_dir = latest

    logger.info("Loading graph from %s", graph_path)
    G = load_graph(graph_path)
    logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    if args.largest_component:
        lcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc).copy()
        logger.info(
            "Largest component: %d nodes, %d edges",
            G.number_of_nodes(), G.number_of_edges(),
        )

    result = hierarchical_leiden(
        G,
        max_levels=args.max_levels,
        resolution=args.resolution,
        seed=args.seed,
    )

    print_report(result)

    if args.output:
        save_results(result, args.output)

    if args.visualize:
        vis_lvl = min(args.vis_level, result.num_levels - 1)
        visualize_communities(G, result.levels[vis_lvl], args.visualize)


if __name__ == "__main__":
    main()
