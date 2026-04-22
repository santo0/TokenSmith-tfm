import json
import argparse
import os
import logging

from src.knowledge_graph.analysis import analyze_query
from src.knowledge_graph.query import CanonicalLookup
from src.knowledge_graph.io import RUNS_DIR, load_graph, load_canonicalization_data
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze query difficulty against a Knowledge Graph."
    )
    parser.add_argument(
        "--run-dir",
        default=os.path.join(RUNS_DIR, "latest"),
    )
    parser.add_argument("--query", required=True, help="The query string to analyze.")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug information during analysis.")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s %(name)s  %(levelname)s %(message)s")

    graph_path = os.path.join(args.run_dir, "graph.json")

    graph = load_graph(graph_path)

    logger.debug(
        f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    synonym_table, canonical_keywords, canonical_embeddings = load_canonicalization_data(
        args.run_dir)  # Populate canonicalization data for analysis
    canonical_lookup = CanonicalLookup(synonym_table, canonical_keywords,
                                       canonical_embeddings)
    result = analyze_query(args.query, graph, canonical_lookup)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
