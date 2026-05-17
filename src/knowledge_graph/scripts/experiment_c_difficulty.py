"""C1–C7 — Difficulty estimation: topology/centrality features vs quality.

Runs the full difficulty feature extraction pipeline on all labeled benchmarks,
correlates individual features with retrieval quality (LLM-as-judge), and fits
a logistic regression classifier (LOOCV, n=small → pilot mode).

Hypotheses covered:
  C1: Subgraph disconnection (component count) correlates with difficulty.
  C2: Subgraph diameter predicts reasoning depth.
  C3: Low concept centrality correlates with poor recall@k.
  C4: Gold chunk count exceeding context budget is a volume-based signal.
  C5: Near-zero recall@k_max flags out-of-scope queries.
  C6: Composite topological signal outperforms single features (LOOCV LogReg).
  C7: Community features improve prediction over topology-only.

Usage:
    python -m src.knowledge_graph.scripts.experiment_c_difficulty \\
        --run-dir data/knowledge_graph/runs/latest \\
        --artifacts-dir index/sections \\
        --embed-model <model-path> \\
        --output results_c.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv


def _spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Compute Spearman ρ and p-value (requires scipy)."""
    try:
        from scipy.stats import spearmanr
        if len(xs) < 3:
            return float("nan"), float("nan")
        result = spearmanr(xs, ys)
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        return float("nan"), float("nan")


def _loocv_auc(X: list[list[float]], y: list[int]) -> float:
    """Leave-one-out cross-validation AUC for logistic regression."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        Xarr = np.array(X, dtype=float)
        yarr = np.array(y, dtype=int)
        if len(set(yarr)) < 2:
            return float("nan")

        scaler = StandardScaler()
        loo = LeaveOneOut()
        probs: list[float] = []
        for train_idx, test_idx in loo.split(Xarr):
            X_train, X_test = Xarr[train_idx], Xarr[test_idx]
            y_train = yarr[train_idx]
            if len(set(y_train)) < 2:
                probs.append(0.5)
                continue
            Xs_train = scaler.fit_transform(X_train)
            Xs_test = scaler.transform(X_test)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(max_iter=1000)
                clf.fit(Xs_train, y_train)
            prob = clf.predict_proba(Xs_test)[0]
            pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
            probs.append(prob[pos_idx])

        try:
            return float(roc_auc_score(yarr, probs))
        except Exception:
            return float("nan")
    except ImportError:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C1-C7: Difficulty feature correlation and classifier evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument("--artifacts-dir", default=None,
                        help="RAG artifacts dir (FAISS). If omitted, KG-only retrieval used.")
    parser.add_argument("--index-prefix", default="textbook_index")
    parser.add_argument("--embed-model", default=None,
                        help="Embedding model for FAISS. Required when --artifacts-dir set.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-k-max", type=int, default=20, help="k for C5 out-of-scope detection")
    parser.add_argument("--difficulty-threshold", type=float, default=0.5,
                        help="LLM judge score below this → hard=1 for C6/C7 labels")
    parser.add_argument("--context-window", type=int, default=8192)
    parser.add_argument("--llm-model", default="google/gemini-3-flash-preview")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    use_llm = not args.no_llm and bool(api_key)

    from src.knowledge_graph.io import (
        load_graph_and_chunks, load_canonicalization_data, resolve_run_dir
    )
    from src.knowledge_graph.query import CanonicalLookup, KGNodeRetriever
    from src.knowledge_graph.analysis import (
        compute_difficulty_features, extract_query_subgraph
    )
    from src.knowledge_graph.scripts.eval_utils import (
        load_labeled_benchmarks, recall_at_k, scores_to_ranked_ids,
        retrieved_tuples, llm_judge, print_table,
    )

    root = Path(__file__).parent.parent.parent.parent
    run_path = Path(args.run_dir)
    if not run_path.is_absolute():
        run_path = root / run_path

    print(f"Loading KG from {run_path}...")
    graph, kg_chunks = load_graph_and_chunks(str(run_path))
    resolved = resolve_run_dir(str(run_path))
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = CanonicalLookup(syn_table, can_kw, can_emb) if syn_table else None

    # Optional FAISS retriever
    faiss_ret = None
    raw_chunks: list[str] = []
    if args.artifacts_dir and args.embed_model:
        try:
            from src.retriever import FAISSRetriever, load_artifacts
            apath = Path(args.artifacts_dir)
            if not apath.is_absolute():
                apath = root / apath
            faiss_idx, _, raw_chunks, _, metadata = load_artifacts(str(apath), args.index_prefix)
            chunk_id_map = [m["chunk_id"] for m in metadata]
            faiss_ret = FAISSRetriever(faiss_idx, args.embed_model, chunk_id_map=chunk_id_map)
            print(f"FAISS enabled ({len(raw_chunks)} chunks).")
        except Exception as e:
            print(f"FAISS not available: {e}")

    kg_ret = KGNodeRetriever(
        graph, kg_chunks, canonical_lookup=canonical_lookup
    )

    llm_client = None
    if use_llm:
        from src.knowledge_graph.openrouter_client import OpenRouterClient
        llm_client = OpenRouterClient(api_key, retries=2)
        print(f"LLM judge enabled: {args.llm_model}")

    benchmarks = load_labeled_benchmarks(args.benchmarks)
    print(f"Evaluating {len(benchmarks)} labeled benchmarks...")

    import networkx as nx

    # Pre-compute community map once (Leiden, cached on graph)
    community_map: dict[str, int] | None = None
    try:
        from src.knowledge_graph.analysis import _get_or_compute_communities
        community_map = _get_or_compute_communities(graph)
        print(f"Community map: {len(set(community_map.values()))} communities.")
    except Exception as e:
        print(f"Community detection failed: {e} — C7 community features will be 0.")

    per_query: list[dict] = []
    for bm in benchmarks:
        query = bm["question"]
        gold: list[int] = bm["ideal_retrieved_chunks"]

        # Retrieve (combined faiss+kg or kg-only)
        kg_scores = kg_ret.get_scores(query, args.top_k_max, list(kg_chunks.values()))
        if faiss_ret:
            faiss_scores = faiss_ret.get_scores(query, args.top_k_max, raw_chunks)
            from src.ranking.ranker import EnsembleRanker
            ranker = EnsembleRanker("rrf", {"faiss": 0.5, "kg_node": 0.5})
            ranked_ids, _ = ranker.rank({"faiss": faiss_scores, "kg_node": kg_scores})
            ranked_ids = ranked_ids[:args.top_k_max]
        else:
            ranked_ids = scores_to_ranked_ids(kg_scores, args.top_k_max)

        recall_k = recall_at_k(ranked_ids, gold, args.top_k)
        recall_kmax = recall_at_k(ranked_ids, gold, args.top_k_max)

        # LLM judge
        judge_score: float | None = None
        if llm_client:
            tuples = [(cid, kg_chunks.get(cid, "")) for cid in ranked_ids[:args.top_k] if cid in kg_chunks]
            try:
                judge_score = llm_judge(llm_client, args.llm_model, query, tuples)
            except Exception as e:
                print(f"  [{bm['id']}] LLM judge failed: {e}")

        # Difficulty features
        features = compute_difficulty_features(
            query, graph, canonical_lookup,
            community_map=community_map,
            context_window=args.context_window,
        )

        # C3: betweenness centrality of query nodes in subgraph
        from src.knowledge_graph.query import extract_query_nodes
        query_nodes = extract_query_nodes(query, graph, canonical_lookup)
        mean_betweenness = 0.0
        if query_nodes:
            subgraph = extract_query_subgraph(query_nodes, graph)
            if subgraph.number_of_nodes() > 1:
                try:
                    bc = nx.betweenness_centrality(subgraph)
                    mean_betweenness = sum(bc.get(n, 0.0) for n in query_nodes) / len(query_nodes)
                except Exception:
                    pass

        # C4: gold chunk token count vs context window
        gold_token_count = sum(len(kg_chunks.get(cid, "").split()) for cid in gold)
        over_budget = int(gold_token_count > args.context_window * 0.75)

        # C7: community span (how many distinct communities the query subgraph touches)
        community_span = 0
        if community_map and query_nodes:
            comms = {community_map[n] for n in query_nodes if n in community_map}
            community_span = len(comms)

        row = {
            "id": bm["id"],
            "n_gold": len(gold),
            "gold_tokens": gold_token_count,
            "over_budget": over_budget,
            # Topology features (D4)
            "components": features.component_count,
            "diameter": features.subgraph_diameter,
            "density": features.edge_density,
            # Coverage (D1)
            "coverage_delta": features.corpus_coverage,
            # Community (D5)
            "community_dispersion": features.community_dispersion,
            "community_span": community_span,
            # Centrality (C3)
            "mean_betweenness": round(mean_betweenness, 6),
            # Retrieval quality
            f"recall@{args.top_k}": round(recall_k, 3),
            f"recall@{args.top_k_max}": round(recall_kmax, 3),
            "judge_score": round(judge_score, 3) if judge_score is not None else None,
            "hard_label": int((judge_score or 0.0) < args.difficulty_threshold) if judge_score is not None else None,
        }

        if args.verbose:
            print(f"\n[{bm['id']}] {query}")
            print(f"  components={row['components']}  diameter={row['diameter']}  "
                  f"density={row['density']:.2f}  community_span={row['community_span']}")
            print(f"  recall@{args.top_k}={row[f'recall@{args.top_k}']}  "
                  f"judge={row['judge_score']}")

        per_query.append(row)

    n = len(per_query)
    print(f"\n{'=' * 70}")
    print(f"C1–C7 Difficulty Results (n={n} queries)")
    print(f"{'=' * 70}")
    print_table(per_query, [
        "id", "components", "diameter", "mean_betweenness",
        f"recall@{args.top_k}", "judge_score", "hard_label"
    ])

    # -----------------------------------------------------------------------
    # Correlation analysis (C1–C3)
    # -----------------------------------------------------------------------
    valid = [r for r in per_query if r["judge_score"] is not None]
    vn = len(valid)
    print(f"\nCorrelation analysis (n={vn} queries with LLM judge scores):")
    note = " (n < 5: exploratory only)" if vn < 5 else ""
    print(f"  Note: n={vn}{note}")

    if vn >= 3:
        for feature, feat_col, quality_col, label in [
            ("C1", "components",        "judge_score", "Spearman(components, 1-judge)"),
            ("C2", "diameter",          "judge_score", "Spearman(diameter, 1-judge)"),
            ("C3", "mean_betweenness",  f"recall@{args.top_k}", "Spearman(centrality, recall)"),
        ]:
            xs = [r[feat_col] for r in valid]
            ys_raw = [r[quality_col] for r in valid if r[quality_col] is not None]
            xs_filt = [r[feat_col] for r in valid if r[quality_col] is not None]
            if feature == "C3":
                ys = ys_raw
            else:
                ys = [1.0 - y for y in ys_raw]
            rho, pval = _spearman(xs_filt, ys)
            print(f"  {feature}: {label:40s}  ρ={rho:.3f}  p={pval:.3f}")

    # C4: over-budget analysis
    over = [r for r in valid if r["over_budget"]]
    under = [r for r in valid if not r["over_budget"]]
    if over and under:
        m_over = sum(r["judge_score"] for r in over) / len(over)
        m_under = sum(r["judge_score"] for r in under) / len(under)
        print(f"\nC4: Gold chunks over context budget ({len(over)} queries)  "
              f"judge={m_over:.3f} vs under-budget ({len(under)})  judge={m_under:.3f}")

    # C5: near-zero recall@kmax → out-of-scope
    zero_recall = [r for r in per_query if r[f"recall@{args.top_k_max}"] == 0.0]
    print(f"\nC5: recall@{args.top_k_max}=0 → possible out-of-scope: "
          f"{len(zero_recall)}/{n} queries")
    for r in zero_recall:
        print(f"  [{r['id']}]")

    # -----------------------------------------------------------------------
    # C6/C7: Logistic regression pilot (LOOCV AUC)
    # -----------------------------------------------------------------------
    labeled_rows = [r for r in per_query if r["hard_label"] is not None]
    ln = len(labeled_rows)
    print(f"\nC6/C7 Classifier pilot (LOOCV, n={ln} labeled queries):")
    if ln < 3:
        print("  Too few labeled queries for meaningful classification.")
    else:
        feature_sets = {
            "C6_full": ["components", "diameter", "density", "coverage_delta", "community_dispersion"],
            "C6_components": ["components"],
            "C6_diameter": ["diameter"],
            "C6_density": ["density"],
            "C6_coverage": ["coverage_delta"],
            "C7_with_community": ["components", "diameter", "density", "coverage_delta",
                                   "community_dispersion", "community_span"],
        }
        y = [r["hard_label"] for r in labeled_rows]
        auc_results: dict[str, float] = {}
        for set_name, feats in feature_sets.items():
            X = [[r[f] for f in feats] for r in labeled_rows]
            auc = _loocv_auc(X, y)
            auc_results[set_name] = auc
            auc_str = f"{auc:.3f}" if not math.isnan(auc) else "N/A"
            print(f"  {set_name:30s}  LOOCV AUC={auc_str}")

        if not math.isnan(auc_results.get("C6_full", float("nan"))) and \
           not math.isnan(auc_results.get("C7_with_community", float("nan"))):
            delta = auc_results["C7_with_community"] - auc_results["C6_full"]
            print(f"\n  C7 community features Δ AUC = {delta:+.3f} vs C6 baseline")

    print(f"\n  *** n={n} is a proof-of-concept. Results need a larger labeled dataset. ***")
    print(f"{'=' * 70}")

    result = {
        "n_benchmarks": n,
        "top_k": args.top_k,
        "top_k_max": args.top_k_max,
        "difficulty_threshold": args.difficulty_threshold,
        "per_query": per_query,
        "c5_out_of_scope": [r["id"] for r in zero_recall],
    }

    if args.output:
        out = Path(args.output)
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=lambda o: None if math.isnan(float(o)) else float(o))
        print(f"Results written to {out}")


if __name__ == "__main__":
    load_dotenv()
    main()
