"""A3 — Query node seeding strategies comparison.

For each labeled benchmark query, compares three strategies and two ablations:

  S1  full         — extract_query_nodes: n-gram decomposition + canonical lookup
                     (synonym table + embedding fallback) + greedy non-overlapping
                     selection. The proposed best strategy.

  S2  emb_query    — embed the whole query as a single vector, ANN over the
                     canonical keyword FAISS index.

  S3  emb_ngram    — embed each query n-gram independently via CanonicalLookup,
                     accept matches above a similarity threshold. No overlap
                     constraint.

  A1  exact_only   — identical to S1 but canonical_lookup=None. Isolates the
                     contribution of CanonicalLookup (synonym + embedding
                     resolution). F1 delta vs S1 measures the value of canonical
                     resolution alone.

  A2  no_dedup     — identical to S1 scoring but without the greedy
                     non-overlapping constraint. All n-grams that resolve to a
                     graph node are included regardless of span overlap. Isolates
                     the contribution of position-aware deduplication.

Evaluation: precision, recall, F1 against the benchmark's gold ``keywords`` field
(after normalization via the same Normalizer used in the graph).

Usage:
    python -m src.knowledge_graph.scripts.experiment_a3_seed_keywords \\
        --run-dir data/knowledge_graph/runs/latest \\
        --benchmarks tests/benchmarks.yaml \\
        --output results_a3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _prf(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    tp = len(predicted & gold)
    p = tp / len(predicted)
    r = tp / len(gold)
    return p, r, _f1(p, r)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A3: Compare seeding strategies for KG query node extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model (must match KG build)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Top-k for whole-query embedding extraction (S2)",
    )
    parser.add_argument(
        "--sim-threshold", type=float, default=0.40,
        help="Cosine similarity threshold for whole-query embedding match (S2)",
    )
    parser.add_argument(
        "--ngram-threshold", type=float, default=0.85,
        help="Cosine similarity threshold for per-ngram canonical resolve (S3)",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    from src.knowledge_graph.io import (
        load_graph_and_chunks,
        load_canonicalization_data,
        load_keyword_index,
        resolve_run_dir,
    )
    from src.knowledge_graph.query import (
        CanonicalLookup,
        TERM_BLACKLIST,
        extract_query_nodes,
        extract_query_nodes_embedding,
    )
    from src.knowledge_graph.ngrams import KW_PATTERN, extract_ngrams_with_spans
    from src.knowledge_graph.normalizer import Normalizer
    from src.knowledge_graph.scripts.eval_utils import load_benchmarks, print_table

    root = Path(__file__).parent.parent.parent.parent
    run_dir_path = Path(args.run_dir)
    if not run_dir_path.is_absolute():
        run_dir_path = root / run_dir_path

    print(f"Loading KG from {run_dir_path}...")
    graph, chunks = load_graph_and_chunks(str(run_dir_path))
    resolved = resolve_run_dir(str(run_dir_path))
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = CanonicalLookup(syn_table, can_kw, can_emb) if syn_table else None
    keyword_index = load_keyword_index(resolved)

    normalizer = Normalizer()
    benchmarks = load_benchmarks(args.benchmarks)
    labeled = [b for b in benchmarks if b.get("keywords")]

    print(f"Evaluating on {len(labeled)} benchmarks with gold keywords...")

    # S3: per-ngram embedding — embeds each ngram, no overlap constraint
    def _ngram_embed_nodes(query: str) -> set[str]:
        if canonical_lookup is None:
            return set()
        nodes: set[str] = set()
        for ngram, _ in extract_ngrams_with_spans(query, KW_PATTERN):
            if ngram.lower() in TERM_BLACKLIST:
                continue
            normalized = normalizer.normalize([ngram])
            if not normalized:
                continue
            canonical, score = canonical_lookup.resolve_with_score(normalized[0])
            if score >= args.ngram_threshold and graph.has_node(canonical):
                nodes.add(canonical)
        return nodes

    # A2: same resolution as S1 but without greedy non-overlapping constraint
    def _no_dedup_nodes(query: str) -> set[str]:
        nodes: set[str] = set()
        for ngram, _ in extract_ngrams_with_spans(query, KW_PATTERN):
            if ngram.lower() in TERM_BLACKLIST:
                continue
            normalized = normalizer.normalize([ngram])
            if not normalized:
                continue
            norm = normalized[0]
            if graph.has_node(norm):
                nodes.add(norm)
            elif canonical_lookup is not None:
                canonical, score = canonical_lookup.resolve_with_score(norm)
                if score > 0 and graph.has_node(canonical):
                    nodes.add(canonical)
        return nodes

    rows = []
    for bm in labeled:
        query = bm["question"]
        gold_raw: list[str] = bm["keywords"]
        gold_norm = set(normalizer.normalize(gold_raw))

        # S1: full strategy — n-gram + canonical lookup + greedy overlap constraint
        s1_nodes = set(extract_query_nodes(query, graph, canonical_lookup))
        s1_norm = set(normalizer.normalize(list(s1_nodes)))

        # S2: whole-query embedding ANN
        s2_nodes: set[str] = set()
        if keyword_index is not None and can_kw:
            s2_nodes = set(extract_query_nodes_embedding(
                query, graph, keyword_index, can_kw,
                embedding_model=args.embed_model,
                top_k=args.top_k,
                similarity_threshold=args.sim_threshold,
            ))
        s2_norm = set(normalizer.normalize(list(s2_nodes)))

        # S3: per-ngram embedding, no overlap constraint
        s3_nodes = _ngram_embed_nodes(query)
        s3_norm = set(normalizer.normalize(list(s3_nodes)))

        # A1: exact-only — no canonical lookup
        a1_nodes = set(extract_query_nodes(query, graph, None))
        a1_norm = set(normalizer.normalize(list(a1_nodes)))

        # A2: full resolution but no overlap deduplication
        a2_nodes = _no_dedup_nodes(query)
        a2_norm = set(normalizer.normalize(list(a2_nodes)))

        p_s1, r_s1, f_s1 = _prf(s1_norm, gold_norm)
        p_s2, r_s2, f_s2 = _prf(s2_norm, gold_norm)
        p_s3, r_s3, f_s3 = _prf(s3_norm, gold_norm)
        p_a1, r_a1, f_a1 = _prf(a1_norm, gold_norm)
        p_a2, r_a2, f_a2 = _prf(a2_norm, gold_norm)

        if args.verbose:
            print(f"\n[{bm['id']}] {query}")
            print(f"  Gold:          {sorted(gold_norm)}")
            print(f"  S1 full:       {sorted(s1_norm)}")
            print(f"  S2 emb_query:  {sorted(s2_norm)}")
            print(f"  S3 emb_ngram:  {sorted(s3_norm)}")
            print(f"  A1 exact_only: {sorted(a1_norm)}")
            print(f"  A2 no_dedup:   {sorted(a2_norm)}")
            print(f"  S1  P={p_s1:.2f} R={r_s1:.2f} F1={f_s1:.2f}")
            print(f"  S2  P={p_s2:.2f} R={r_s2:.2f} F1={f_s2:.2f}")
            print(f"  S3  P={p_s3:.2f} R={r_s3:.2f} F1={f_s3:.2f}")
            print(f"  A1  P={p_a1:.2f} R={r_a1:.2f} F1={f_a1:.2f}")
            print(f"  A2  P={p_a2:.2f} R={r_a2:.2f} F1={f_a2:.2f}")

        rows.append({
            "id": bm["id"],
            "gold_n": len(gold_norm),
            "s1_P": round(p_s1, 3), "s1_R": round(r_s1, 3), "s1_F1": round(f_s1, 3),
            "s2_P": round(p_s2, 3), "s2_R": round(r_s2, 3), "s2_F1": round(f_s2, 3),
            "s3_P": round(p_s3, 3), "s3_R": round(r_s3, 3), "s3_F1": round(f_s3, 3),
            "a1_P": round(p_a1, 3), "a1_R": round(r_a1, 3), "a1_F1": round(f_a1, 3),
            "a2_P": round(p_a2, 3), "a2_R": round(r_a2, 3), "a2_F1": round(f_a2, 3),
        })

    # Macro-averages
    n = len(rows)
    metric_keys = (
        "s1_P", "s1_R", "s1_F1",
        "s2_P", "s2_R", "s2_F1",
        "s3_P", "s3_R", "s3_F1",
        "a1_P", "a1_R", "a1_F1",
        "a2_P", "a2_R", "a2_F1",
    )
    macro = {k: round(sum(r[k] for r in rows) / n, 4) for k in metric_keys}

    W = 70
    print(f"\n{'=' * W}")
    print("A3 Results — seed keyword extraction P/R/F1 vs gold keywords")
    print(f"{'=' * W}")
    print_table(rows, [
        "id", "gold_n",
        "s1_P", "s1_R", "s1_F1",
        "s2_P", "s2_R", "s2_F1",
        "s3_P", "s3_R", "s3_F1",
        "a1_P", "a1_R", "a1_F1",
        "a2_P", "a2_R", "a2_F1",
    ])
    print()

    rows_summary = [
        ("S1  full        ", "s1"),
        ("S2  emb_query   ", "s2"),
        ("S3  emb_ngram   ", "s3"),
        ("A1  exact_only  ", "a1"),
        ("A2  no_dedup    ", "a2"),
    ]
    for label, key in rows_summary:
        print(
            f"  {label}  "
            f"P={macro[f'{key}_P']:.3f}  "
            f"R={macro[f'{key}_R']:.3f}  "
            f"F1={macro[f'{key}_F1']:.3f}"
        )

    best_key, best_f1 = max(
        ((key, macro[f"{key}_F1"]) for _, key in rows_summary),
        key=lambda x: x[1],
    )
    best_label = next(label.strip() for label, key in rows_summary if key == best_key)
    print(f"\n  Best macro-F1: {best_label}  ({best_f1:.4f})")

    # Component deltas
    lookup_gain = round(macro["s1_F1"] - macro["a1_F1"], 4)
    dedup_gain = round(macro["s1_F1"] - macro["a2_F1"], 4)
    print(f"  CanonicalLookup gain (S1 - A1): {lookup_gain:+.4f}")
    print(f"  Overlap dedup gain  (S1 - A2): {dedup_gain:+.4f}")
    print(f"{'=' * W}")

    result = {
        "n_benchmarks": n,
        "macro": macro,
        "per_query": rows,
        "component_deltas": {
            "canonical_lookup_f1_gain": lookup_gain,
            "overlap_dedup_f1_gain": dedup_gain,
        },
        "config": {
            "embed_model": args.embed_model,
            "top_k": args.top_k,
            "sim_threshold": args.sim_threshold,
            "ngram_threshold": args.ngram_threshold,
        },
    }

    if args.output:
        out = Path(args.output)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {out}")


if __name__ == "__main__":
    load_dotenv()
    main()
