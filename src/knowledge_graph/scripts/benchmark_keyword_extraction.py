"""Benchmark the three query-keyword extraction strategies against queries.json.

Compares:
  - classic  : extract_query_nodes (n-gram + canonical lookup)
  - embedding: extract_query_nodes_embedding (FAISS ANN)
  - hybrid   : extract_query_nodes_hybrid (classic fill with embedding)

Metrics per query:
  - recall    : fraction of expected entities found among extracted nodes
  - precision : fraction of extracted nodes that are expected entities
  - f1        : harmonic mean of precision and recall

Matching is done after normalizing expected entities with the same Normalizer
used at build time, then resolving through CanonicalLookup when available.
"""

import argparse
import json
import logging

from src.knowledge_graph.build import RUNS_DIR
from src.knowledge_graph.io import (
    load_canonicalization_data,
    load_graph_and_chunks,
    load_keyword_index,
    resolve_run_dir,
)
from src.knowledge_graph.normalizer import Normalizer
from src.knowledge_graph.query import (
    CanonicalLookup,
    extract_query_nodes,
    extract_query_nodes_embedding,
    extract_query_nodes_hybrid,
)

logger = logging.getLogger(__name__)

_normalizer = Normalizer()

METHODS = ("classic", "embedding", "hybrid")


def _normalize_expected(
    expected: list[str],
    canonical_lookup: CanonicalLookup | None,
) -> set[str]:
    """Normalize and optionally canonicalize expected entity strings."""
    normalized = _normalizer.normalize(expected)
    if canonical_lookup is None:
        return set(normalized)
    return {canonical_lookup.resolve(t) for t in normalized}


def _metrics(extracted: list[str], expected_norm: set[str]) -> dict:
    if not expected_norm and not extracted:
        return {"recall": 1.0, "precision": 1.0, "f1": 1.0, "hits": [], "misses": []}

    hit_set = {e for e in extracted if e in expected_norm}
    recall = len(hit_set) / len(expected_norm) if expected_norm else 0.0
    precision = len(hit_set) / len(extracted) if extracted else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    hits = sorted(hit_set)
    misses = sorted(expected_norm - hit_set)
    extra = sorted(set(extracted) - expected_norm)

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "hits": hits,
        "misses": misses,
        "extra": extra,
    }


def run_benchmark(
    run_dir: str,
    queries: list[dict],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_top_k: int = 20,
    embedding_threshold: float = 0.78,
    hybrid_threshold: float = 0.4,
) -> list[dict]:
    resolved = resolve_run_dir(run_dir)
    graph, _ = load_graph_and_chunks(resolved)

    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = (
        CanonicalLookup(syn_table, can_kw, can_emb) if syn_table is not None else None
    )

    keyword_index = load_keyword_index(resolved)
    if keyword_index is None:
        logger.warning("No keyword_index.faiss found — embedding/hybrid methods will be skipped.")

    results = []
    for q in queries:
        qid = q.get("id", "?")
        query_text = q.get("query", "")
        expected_raw: list[str] = q.get("expected_entities", [])
        expected_norm = _normalize_expected(expected_raw, canonical_lookup)

        extracted: dict[str, list[str]] = {}

        extracted["classic"] = extract_query_nodes(query_text, graph, canonical_lookup)

        if keyword_index is not None and can_kw:
            extracted["embedding"] = extract_query_nodes_embedding(
                query_text,
                graph,
                keyword_index,
                can_kw,
                embedding_model=embedding_model,
                top_k=embedding_top_k,
                similarity_threshold=embedding_threshold,
            )
            extracted["hybrid"] = extract_query_nodes_hybrid(
                query_text,
                graph,
                keyword_index,
                can_kw,
                canonical_lookup=canonical_lookup,
                embedding_model=embedding_model,
                embedding_top_k=embedding_top_k,
                embedding_threshold=hybrid_threshold,
            )
        else:
            extracted["embedding"] = []
            extracted["hybrid"] = []

        method_results = {}
        for method in METHODS:
            method_results[method] = _metrics(extracted[method], expected_norm)
            method_results[method]["extracted"] = extracted[method]

        results.append(
            {
                "id": qid,
                "query": query_text,
                "mode": q.get("mode", ""),
                "expected_norm": sorted(expected_norm),
                "methods": method_results,
            }
        )

    return results


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def print_summary(results: list[dict]) -> None:
    col_q = 32
    col_m = 14  # per method: recall + precision + f1

    header1 = f"{'Query':<{col_q}}"
    header2 = " " * col_q
    for m in METHODS:
        header1 += f"  {m.center(col_m * 3 + 4)}"
        header2 += f"  {'rec':>4}  {'pre':>4}  {'f1':>4}"
    sep = "-" * len(header2)

    print(f"\n{sep}")
    print(header1)
    print(header2)
    print(sep)

    agg: dict[str, dict[str, list[float]]] = {
        m: {"recall": [], "precision": [], "f1": []} for m in METHODS
    }

    for r in results:
        row = f"{r['id'][:col_q]:<{col_q}}"
        for m in METHODS:
            mm = r["methods"].get(m, {})
            rec = mm.get("recall", 0.0)
            pre = mm.get("precision", 0.0)
            f1 = mm.get("f1", 0.0)
            agg[m]["recall"].append(rec)
            agg[m]["precision"].append(pre)
            agg[m]["f1"].append(f1)
            row += f"  {rec:.2f}  {pre:.2f}  {f1:.2f}"
        print(row)

    print(sep)
    avg_row = f"{'AVERAGE':<{col_q}}"
    for m in METHODS:
        avg_row += (
            f"  {_avg(agg[m]['recall']):.2f}"
            f"  {_avg(agg[m]['precision']):.2f}"
            f"  {_avg(agg[m]['f1']):.2f}"
        )
    print(avg_row)
    print(sep)

    # Per-mode breakdown
    modes = sorted({r["mode"] for r in results if r.get("mode")})
    if len(modes) > 1:
        print("\nPer-mode averages (recall / f1):")
        for mode in modes:
            subset = [r for r in results if r.get("mode") == mode]
            print(f"  {mode}:")
            for m in METHODS:
                rec = _avg([r["methods"][m]["recall"] for r in subset])
                f1 = _avg([r["methods"][m]["f1"] for r in subset])
                print(f"    {m:10s}  rec={rec:.2f}  f1={f1:.2f}")


def print_details(results: list[dict]) -> None:
    for r in results:
        print(f"\n[{r['id']}] {r['query']}")
        print(f"  expected : {r['expected_norm']}")
        for m in METHODS:
            mm = r["methods"][m]
            print(
                f"  {m:10s}: extracted={mm['extracted']}  "
                f"rec={mm['recall']:.2f}  pre={mm['precision']:.2f}  f1={mm['f1']:.2f}"
            )
            if mm["misses"]:
                print(f"             misses={mm['misses']}")
            if mm["extra"]:
                print(f"             extra={mm['extra']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark query-keyword extraction methods against queries.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        default=RUNS_DIR,
        help="KG run directory or runs/ parent with 'latest' symlink",
    )
    parser.add_argument(
        "--queries",
        default="queries.json",
        help="Path to queries.json",
    )
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model (must match the one used to build the index)",
    )
    parser.add_argument(
        "--embedding-top-k",
        type=int,
        default=20,
        help="FAISS neighbours for embedding/hybrid retrieval",
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.78,
        help="Cosine similarity threshold for embedding-only method",
    )
    parser.add_argument(
        "--hybrid-threshold",
        type=float,
        default=0.4,
        help="Cosine similarity threshold for the hybrid fill step",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write full results to this JSON file",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print per-query extracted nodes, hits, misses",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    with open(args.queries, encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("queries", data) if isinstance(data, dict) else data
    print(f"Loaded {len(queries)} queries from {args.queries}")

    results = run_benchmark(
        run_dir=args.run_dir,
        queries=queries,
        embedding_model=args.embed_model,
        embedding_top_k=args.embedding_top_k,
        embedding_threshold=args.embedding_threshold,
        hybrid_threshold=args.hybrid_threshold,
    )

    print_summary(results)

    if args.details:
        print_details(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
