"""GC1–GC3 — Global centrality of query nodes predicts retrieval quality.

The original difficulty experiment (C1–C7) examined query *subgraph* topology and
found no signal: all 32 benchmark queries produce structurally identical subgraphs
(1 connected component, diameter 0–3, near-zero local betweenness). The features
carry no variance because they measure the same small, dense neighbourhood.

This experiment shifts the lens from the local subgraph to the query nodes'
position inside the *full* knowledge graph. The intuition is that a concept
node sitting at the periphery of the KG — low PageRank, seldom traversed on
shortest paths, few co-occurrence neighbours — is harder for the retriever to
reach through graph walks than a well-connected hub concept.

Hypotheses:
  GC1: Low mean PageRank of query nodes → harder retrieval.
       PageRank approximates how often a random walk visits a node; peripheral
       nodes accumulate less traffic and offer fewer indirect retrieval paths.
  GC2: Low global betweenness centrality of query nodes → harder retrieval.
       Betweenness counts how often a node lies on shortest paths between other
       nodes. Low betweenness means the concept is bypassed by the graph's routing
       structure, so multi-hop retrieval is less likely to pass through it.
  GC3: Low degree of query nodes → harder retrieval.
       Degree directly controls how many neighbours a node exposes; the KG
       retriever expands from query nodes via BFS, so fewer neighbours means
       a smaller retrieval frontier.

Expected correlations (Spearman ρ with judge_score / recall@k):
  All three centrality measures should show positive ρ
  (higher centrality → easier query → higher quality).

Usage:
    python -m src.knowledge_graph.scripts.experiment_c_global_centrality \\
        --run-dir data/knowledge_graph/runs/latest \\
        --output results_c_global_centrality.json
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
    """Compute Spearman ρ and two-tailed p-value (requires scipy)."""
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
        description="GC1-GC3: Global KG centrality of query nodes vs. retrieval quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--difficulty-threshold", type=float, default=0.5,
                        help="LLM judge score below this → hard=1 for classifier labels")
    parser.add_argument("--llm-model", default="google/gemini-3-flash-preview")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--unanswerable-benchmarks", default=None,
                        help="Path to benchmarks_unanswerable.yaml; produces "
                             "centrality features with no retrieval eval.")
    parser.add_argument("--unanswerable-output", default=None,
                        help="Output path for unanswerable-query centrality results.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    use_llm = not args.no_llm and bool(api_key)

    from src.knowledge_graph.io import (
        load_graph_and_chunks, load_canonicalization_data, resolve_run_dir
    )
    from src.knowledge_graph.query import CanonicalLookup, KGNodeRetriever, extract_query_nodes
    from src.knowledge_graph.scripts.eval_utils import (
        load_benchmarks, load_labeled_benchmarks, recall_at_k,
        scores_to_ranked_ids, llm_judge, print_table,
    )
    import networkx as nx

    root = Path(__file__).parent.parent.parent.parent
    run_path = Path(args.run_dir)
    if not run_path.is_absolute():
        run_path = root / run_path

    print(f"Loading KG from {run_path}...")
    graph, kg_chunks = load_graph_and_chunks(str(run_path))
    resolved = resolve_run_dir(str(run_path))
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = CanonicalLookup(syn_table, can_kw, can_emb) if syn_table else None

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    print(f"Graph: {n_nodes} nodes, {n_edges} edges")

    # ------------------------------------------------------------------
    # Pre-compute global centrality measures (once, before the query loop)
    # ------------------------------------------------------------------
    print("Computing PageRank (alpha=0.85)...")
    pagerank: dict[str, float] = nx.pagerank(graph)

    betweenness_k = min(500, n_nodes)
    print(f"Computing approximate betweenness centrality (k={betweenness_k}, seed=42)...")
    betweenness: dict[str, float] = nx.betweenness_centrality(graph, k=betweenness_k, seed=42)
    # Degree is O(1) per node; no pre-computation needed.

    kg_ret = KGNodeRetriever(graph, kg_chunks, canonical_lookup=canonical_lookup)

    llm_client = None
    if use_llm:
        from src.knowledge_graph.openrouter_client import OpenRouterClient
        llm_client = OpenRouterClient(api_key, retries=2)
        print(f"LLM judge enabled: {args.llm_model}")

    benchmarks = load_labeled_benchmarks(args.benchmarks)
    print(f"Evaluating {len(benchmarks)} labeled benchmarks...")

    per_query: list[dict] = []
    for bm in benchmarks:
        query = bm["question"]
        gold: list[int] = bm["ideal_retrieved_chunks"]

        # Retrieve (KG-only)
        kg_scores = kg_ret.get_scores(query, args.top_k, list(kg_chunks.values()))
        ranked_ids = scores_to_ranked_ids(kg_scores, args.top_k)
        recall_k = recall_at_k(ranked_ids, gold, args.top_k)

        # LLM judge
        judge_score: float | None = None
        if llm_client:
            tuples = [(cid, kg_chunks.get(cid, "")) for cid in ranked_ids if cid in kg_chunks]
            try:
                judge_score = llm_judge(llm_client, args.llm_model, query, tuples)
            except Exception as e:
                print(f"  [{bm['id']}] LLM judge failed: {e}")

        # Extract query nodes and compute centrality aggregates
        query_nodes = extract_query_nodes(query, graph, canonical_lookup)

        if query_nodes:
            pr_vals  = [pagerank.get(n, 0.0)    for n in query_nodes]
            bc_vals  = [betweenness.get(n, 0.0) for n in query_nodes]
            deg_vals = [graph.degree(n)          for n in query_nodes]
            mean_pr  = sum(pr_vals)  / len(pr_vals)
            min_pr   = min(pr_vals)
            mean_bc  = sum(bc_vals)  / len(bc_vals)
            min_bc   = min(bc_vals)
            mean_deg = sum(deg_vals) / len(deg_vals)
            min_deg  = int(min(deg_vals))
        else:
            if args.verbose:
                print(f"  [{bm['id']}] No query nodes matched — centrality features set to 0")
            mean_pr = min_pr = mean_bc = min_bc = mean_deg = 0.0
            min_deg = 0

        row = {
            "id": bm["id"],
            "n_query_nodes": len(query_nodes),
            # GC1: PageRank (scale ~1e-4 to 1e-2 on a 21k-node graph)
            "mean_pagerank": round(mean_pr, 8),
            "min_pagerank":  round(min_pr, 8),
            # GC2: Betweenness (approximate, k=500)
            "mean_betweenness": round(mean_bc, 6),
            "min_betweenness":  round(min_bc, 6),
            # GC3: Degree
            "mean_degree": round(mean_deg, 3),
            "min_degree":  min_deg,
            # Retrieval quality
            f"recall@{args.top_k}": round(recall_k, 3),
            "judge_score": round(judge_score, 3) if judge_score is not None else None,
            "hard_label": (
                int((judge_score or 0.0) < args.difficulty_threshold)
                if judge_score is not None else None
            ),
        }

        if args.verbose:
            print(f"\n[{bm['id']}] {query}")
            print(f"  query_nodes={query_nodes}")
            print(f"  mean_pr={row['mean_pagerank']:.2e}  mean_bc={row['mean_betweenness']:.4f}"
                  f"  mean_deg={row['mean_degree']:.1f}")
            print(f"  recall@{args.top_k}={row[f'recall@{args.top_k}']}  judge={row['judge_score']}")

        per_query.append(row)

    n = len(per_query)
    print(f"\n{'=' * 70}")
    print(f"GC1–GC3 Global Centrality Results (n={n} queries)")
    print(f"{'=' * 70}")
    print_table(per_query, [
        "id", "n_query_nodes", "mean_pagerank", "mean_betweenness",
        "mean_degree", f"recall@{args.top_k}", "judge_score", "hard_label"
    ])

    # ------------------------------------------------------------------
    # Correlation analysis (GC1–GC3)
    # ------------------------------------------------------------------
    valid = [r for r in per_query if r["judge_score"] is not None]
    vn = len(valid)
    print(f"\nCorrelation analysis (n={vn} queries with LLM judge scores):")
    note = " (n < 5: exploratory only)" if vn < 5 else ""
    print(f"  Note: n={vn}{note}")

    corr_specs = [
        ("GC1", "mean_pagerank",    "judge_score",          "Spearman(mean_PR,  judge_score)  [expect ρ>0]"),
        ("GC1", "min_pagerank",     "judge_score",          "Spearman(min_PR,   judge_score)  [expect ρ>0]"),
        ("GC2", "mean_betweenness", "judge_score",          "Spearman(mean_BC,  judge_score)  [expect ρ>0]"),
        ("GC3", "mean_degree",      "judge_score",          "Spearman(mean_deg, judge_score)  [expect ρ>0]"),
        ("GC1", "mean_pagerank",    f"recall@{args.top_k}", "Spearman(mean_PR,  recall@k)     [expect ρ>0]"),
        ("GC2", "mean_betweenness", f"recall@{args.top_k}", "Spearman(mean_BC,  recall@k)     [expect ρ>0]"),
        ("GC3", "mean_degree",      f"recall@{args.top_k}", "Spearman(mean_deg, recall@k)     [expect ρ>0]"),
    ]

    corr_results: dict[str, float] = {}
    if vn >= 3:
        for hyp, feat, quality_col, label in corr_specs:
            if quality_col == "judge_score":
                pool = valid
            else:
                pool = per_query
            pool = [r for r in pool if r.get(quality_col) is not None and r.get(feat) is not None]
            if len(pool) < 3:
                print(f"  {hyp}: {label}  (n={len(pool)} < 3, skipped)")
                continue
            rho, pval = _spearman(
                [r[feat] for r in pool],
                [r[quality_col] for r in pool],
            )
            corr_results[f"{feat}_vs_{quality_col}"] = rho
            pval_str = f"{pval:.3f}" if not math.isnan(pval) else "N/A"
            print(f"  {hyp}: {label:50s}  ρ={rho:.3f}  p={pval_str}")
    else:
        print("  Too few queries with judge scores for correlation analysis.")

    # ------------------------------------------------------------------
    # GC6/GC7-style classifier (LOOCV AUC) on hard_label
    # ------------------------------------------------------------------
    labeled = [r for r in per_query if r["hard_label"] is not None]
    ln = len(labeled)
    print(f"\nClassifier pilot (LOOCV, n={ln} labeled queries):")
    if ln < 3:
        print("  Too few labeled queries for meaningful classification.")
    else:
        feature_sets = {
            "GC_all3":          ["mean_pagerank", "mean_betweenness", "mean_degree"],
            "GC1_pagerank":     ["mean_pagerank"],
            "GC2_betweenness":  ["mean_betweenness"],
            "GC3_degree":       ["mean_degree"],
        }
        y = [r["hard_label"] for r in labeled]
        for set_name, feats in feature_sets.items():
            X = [[r[f] for f in feats] for r in labeled]
            auc = _loocv_auc(X, y)
            auc_str = f"{auc:.3f}" if not math.isnan(auc) else "N/A"
            print(f"  {set_name:20s}  LOOCV AUC={auc_str}")

    print(f"\n  *** n={n} is a proof-of-concept. Results need a larger labeled dataset. ***")
    print(f"{'=' * 70}")

    result = {
        "n_benchmarks": n,
        "top_k": args.top_k,
        "difficulty_threshold": args.difficulty_threshold,
        "graph_stats": {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "betweenness_k": betweenness_k,
        },
        "per_query": per_query,
    }

    if args.output:
        out = Path(args.output)
        with open(out, "w") as f:
            json.dump(result, f, indent=2,
                      default=lambda o: None if math.isnan(float(o)) else float(o))
        print(f"Results written to {out}")

    # ------------------------------------------------------------------
    # Optional: compute centrality for unanswerable benchmarks
    # ------------------------------------------------------------------
    if args.unanswerable_benchmarks and args.unanswerable_output:
        unans_bms = load_benchmarks(args.unanswerable_benchmarks)
        print(f"\nComputing centrality for {len(unans_bms)} unanswerable benchmarks...")

        unans_rows: list[dict] = []
        for bm in unans_bms:
            query = bm["question"]
            query_nodes = extract_query_nodes(query, graph, canonical_lookup)

            if query_nodes:
                pr_vals  = [pagerank.get(n, 0.0)    for n in query_nodes]
                bc_vals  = [betweenness.get(n, 0.0) for n in query_nodes]
                deg_vals = [graph.degree(n)          for n in query_nodes]
                mean_pr  = sum(pr_vals)  / len(pr_vals)
                min_pr   = min(pr_vals)
                mean_bc  = sum(bc_vals)  / len(bc_vals)
                min_bc   = min(bc_vals)
                mean_deg = sum(deg_vals) / len(deg_vals)
                min_deg  = int(min(deg_vals))
            else:
                if args.verbose:
                    print(f"  [{bm['id']}] No query nodes matched — centrality set to 0")
                mean_pr = min_pr = mean_bc = min_bc = mean_deg = 0.0
                min_deg = 0

            row = {
                "id": bm["id"],
                "n_query_nodes": len(query_nodes),
                "mean_pagerank":     round(mean_pr,  8),
                "min_pagerank":      round(min_pr,   8),
                "mean_betweenness":  round(mean_bc,  6),
                "min_betweenness":   round(min_bc,   6),
                "mean_degree":       round(mean_deg, 3),
                "min_degree":        min_deg,
            }
            if args.verbose:
                print(f"  [{bm['id']}] nodes={len(query_nodes)}"
                      f"  PR={row['mean_pagerank']:.2e}"
                      f"  BC={row['mean_betweenness']:.4f}"
                      f"  deg={row['mean_degree']:.1f}")
            unans_rows.append(row)

        unans_result = {
            "n_benchmarks": len(unans_rows),
            "graph_stats": result["graph_stats"],
            "per_query": unans_rows,
        }
        unans_out = Path(args.unanswerable_output)
        with open(unans_out, "w") as f:
            json.dump(unans_result, f, indent=2,
                      default=lambda o: None if math.isnan(float(o)) else float(o))
        print(f"Unanswerable results written to {unans_out}")


if __name__ == "__main__":
    load_dotenv()
    main()
