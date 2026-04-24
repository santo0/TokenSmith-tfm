import argparse

from src.knowledge_graph.io import load_graph_chunks_and_tree, RUNS_DIR


def analyze_chunk_id_consistency(output_dir: str) -> None:
    graph, chunks, tree = load_graph_chunks_and_tree(output_dir)

    chunk_ids = set(chunks.keys())

    # Collect all chunk_ids referenced by graph nodes
    graph_chunk_ids: set[int] = set()
    for _, data in graph.nodes(data=True):
        for cid in data.get("chunk_ids", []):
            graph_chunk_ids.add(cid)

    # Collect all chunk_ids referenced by tree nodes
    tree_chunk_ids: set[int] = set()
    if tree is not None:
        tree_chunk_ids = set(tree.chunk_to_sections.keys())

    print(f"chunks dict   : {len(chunk_ids):>6} IDs  range=[{min(chunk_ids)}, {max(chunk_ids)}]")
    print(f"graph nodes   : {len(graph_chunk_ids):>6} IDs  range=[{min(graph_chunk_ids) if graph_chunk_ids else 'n/a'}, {max(graph_chunk_ids) if graph_chunk_ids else 'n/a'}]")
    if tree is not None:
        print(f"section tree  : {len(tree_chunk_ids):>6} IDs  range=[{min(tree_chunk_ids) if tree_chunk_ids else 'n/a'}, {max(tree_chunk_ids) if tree_chunk_ids else 'n/a'}]")
    else:
        print("section tree  : not found")

    print()

    # Graph vs chunks
    graph_not_in_chunks = graph_chunk_ids - chunk_ids
    chunks_not_in_graph = chunk_ids - graph_chunk_ids
    print(f"graph refs not in chunks : {len(graph_not_in_chunks)}")
    if graph_not_in_chunks:
        print(f"  sample: {sorted(graph_not_in_chunks)[:10]}")
    print(f"chunks not in graph      : {len(chunks_not_in_graph)}")
    if chunks_not_in_graph:
        print(f"  sample: {sorted(chunks_not_in_graph)[:10]}")

    if tree is not None:
        print()
        tree_not_in_chunks = tree_chunk_ids - chunk_ids
        chunks_not_in_tree = chunk_ids - tree_chunk_ids
        print(f"tree refs not in chunks  : {len(tree_not_in_chunks)}")
        if tree_not_in_chunks:
            print(f"  sample: {sorted(tree_not_in_chunks)[:10]}")
        print(f"chunks not in tree       : {len(chunks_not_in_tree)}")
        if chunks_not_in_tree:
            print(f"  sample: {sorted(chunks_not_in_tree)[:10]}")

        print()
        graph_not_in_tree = graph_chunk_ids - tree_chunk_ids
        tree_not_in_graph = tree_chunk_ids - graph_chunk_ids
        print(f"graph refs not in tree   : {len(graph_not_in_tree)}")
        if graph_not_in_tree:
            print(f"  sample: {sorted(graph_not_in_tree)[:10]}")
        print(f"tree refs not in graph   : {len(tree_not_in_graph)}")
        if tree_not_in_graph:
            print(f"  sample: {sorted(tree_not_in_graph)[:10]}")

    print()
    all_match = (
        not graph_not_in_chunks
        and not chunks_not_in_graph
        and (tree is None or (not tree_not_in_chunks and not chunks_not_in_tree))
    )
    print("All chunk IDs consistent:", all_match)

    # ── Text content check: does each graph node keyword appear in its chunks? ──
    print()
    print("=== Text content check (graph node keyword ∈ chunk text) ===")
    total_refs = 0
    mismatches: list[tuple[str, int, str]] = []  # (keyword, chunk_id, chunk_snippet)
    for node_name, data in graph.nodes(data=True):
        node_lower = node_name.lower()
        for cid in data.get("chunk_ids", []):
            total_refs += 1
            text = chunks.get(cid, "")
            if node_lower not in text.lower():
                snippet = text[:80].replace("\n", " ")
                mismatches.append((node_name, cid, snippet))

    hit_rate = (total_refs - len(mismatches)) / total_refs if total_refs else 0.0
    print(f"Total (keyword, chunk) pairs checked : {total_refs}")
    print(f"Keyword found in chunk text          : {total_refs - len(mismatches)}  ({hit_rate:.1%})")
    print(f"Keyword NOT found in chunk text      : {len(mismatches)}")
    if mismatches:
        print("  Sample mismatches (keyword → chunk_id | chunk snippet):")
        for kw, cid, snippet in mismatches[:10]:
            print(f"    {kw!r:30s} → chunk {cid:5d} | {snippet!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check chunk ID consistency across graph, chunks, and section tree.")
    parser.add_argument("--run-dir", default=RUNS_DIR, help="Path to run dir or runs/ parent (default: latest run)")
    args = parser.parse_args()
    analyze_chunk_id_consistency(args.run_dir)
