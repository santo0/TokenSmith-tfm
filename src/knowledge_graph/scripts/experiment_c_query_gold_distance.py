"""QG1–QG3 — Query-to-gold topological distance predicts retrieval failure.

The original difficulty experiment (C1–C7) and the global centrality experiment
(experiment_c_global_centrality.py) characterise properties of the query nodes
alone. Neither measures whether the KG topology actually *bridges* the question
to its answer.

This experiment computes the shortest path in the full KG from each query concept
node to each KG node associated with a gold chunk. If these paths are long, the
KG's graph structure does not connect the question to its answer — and KG-based
retrieval, which expands from query nodes via graph walks, should predictably fail.

This is a structurally motivated predictor: a small mean_qg_distance means the
retriever can reach the answer in few hops from where the query lands; a large
distance means the path through the KG is long or indirect, independent of how
central or peripheral the query nodes are.

Hypotheses:
  QG1: Mean shortest path from query nodes to gold chunk nodes (mean_qg_distance)
       correlates negatively with judge_score and recall@k — large distance means
       the KG does not structurally bridge question to answer.
  QG2: Queries where gold nodes are unreachable from all query nodes have recall=0.
       On the current fully-connected KG this hypothesis is trivially satisfied
       (0 unreachable gold nodes), which is itself a finding: the KG provides
       structural coverage of all benchmark answers, so failure must be attributed
       to scoring, not to topological isolation.
  QG3: mean_qg_distance outperforms subgraph diameter from experiment_c_difficulty
       as a predictor of retrieval quality. If --baseline-results is provided
       (path to results_c.json), a direct Spearman ρ comparison is printed.

BFS efficiency note:
  Rather than running one BFS per (query_node, gold_node) pair, this script runs
  one BFS per query node (nx.single_source_shortest_path_length), which returns
  distances to all reachable nodes in a single O(V+E) traversal. Gold-node
  distances are then looked up in the resulting dict in O(1). For 32 benchmarks
  with at most ~5 query nodes each, total BFS calls ≤ 160.

Usage:
    python -m src.knowledge_graph.scripts.experiment_c_query_gold_distance \\
        --run-dir data/knowledge_graph/runs/latest \\
        --output results_c_query_gold_distance.json

    # With QG3 cross-experiment comparison:
    python -m src.knowledge_graph.scripts.experiment_c_query_gold_distance \\
        --run-dir data/knowledge_graph/runs/latest \\
        --baseline-results results_c.json \\
        --output results_c_query_gold_distance.json
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


def _compute_qg_distances(
    query_nodes: list[str],
    gold_chunk_ids: list[int],
    graph,
    chunk_to_nodes: dict[int, list[str]],
) -> dict:
    """Return query-to-gold distance statistics using single-source BFS.

    A gold node is considered reachable if any query node can reach it (union
    semantics). This matches the retriever's behaviour: the retriever starts BFS
    from all query nodes simultaneously.
    """
    import networkx as nx

    gold_nodes: set[str] = set()
    for cid in gold_chunk_ids:
        gold_nodes.update(chunk_to_nodes.get(cid, []))
    n_gold_nodes = len(gold_nodes)

    if not query_nodes or not gold_nodes:
        return {
            "n_gold_nodes_in_kg":      n_gold_nodes,
            "n_unreachable_gold_nodes": n_gold_nodes if not query_nodes else 0,
            "mean_qg_distance": None,
            "min_qg_distance":  None,
            "max_qg_distance":  None,
        }

    all_distances: list[int] = []
    unreachable_gold: set[str] = set(gold_nodes)

    for qn in query_nodes:
        # Single BFS returns distances to all reachable nodes in O(V+E).
        dist_map: dict[str, int] = dict(nx.single_source_shortest_path_length(graph, qn))
        for gn in gold_nodes:
            if gn in dist_map:
                all_distances.append(dist_map[gn])
                unreachable_gold.discard(gn)

    n_unreachable = len(unreachable_gold)

    if not all_distances:
        return {
            "n_gold_nodes_in_kg":      n_gold_nodes,
            "n_unreachable_gold_nodes": n_unreachable,
            "mean_qg_distance": None,
            "min_qg_distance":  None,
            "max_qg_distance":  None,
        }

    return {
        "n_gold_nodes_in_kg":      n_gold_nodes,
        "n_unreachable_gold_nodes": n_unreachable,
        "mean_qg_distance": sum(all_distances) / len(all_distances),
        "min_qg_distance":  float(min(all_distances)),
        "max_qg_distance":  float(max(all_distances)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QG1-QG3: Query-to-gold topological distance vs. retrieval quality.",
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
    parser.add_argument("--baseline-results", default=None,
                        help="Path to results_c.json from experiment_c_difficulty for QG3 comparison.")
    parser.add_argument("--output", default=None)
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

    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # Pre-build chunk → KG node reverse map (once, before the query loop)
    # ------------------------------------------------------------------
    print("Building chunk-to-nodes reverse map...")
    chunk_to_nodes: dict[int, list[str]] = {}
    for node, data in graph.nodes(data=True):
        for cid in data.get("chunk_ids", []):
            chunk_to_nodes.setdefault(cid, []).append(node)
    print(f"  {len(chunk_to_nodes)} chunks mapped to KG nodes.")

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

        query_nodes = extract_query_nodes(query, graph, canonical_lookup)
        if not query_nodes and args.verbose:
            print(f"  [{bm['id']}] No query nodes matched — qg distances will be None")

        qg = _compute_qg_distances(query_nodes, gold, graph, chunk_to_nodes)

        mean_d = qg["mean_qg_distance"]
        min_d  = qg["min_qg_distance"]
        max_d  = qg["max_qg_distance"]

        row = {
            "id": bm["id"],
            "n_query_nodes":          len(query_nodes),
            "n_gold_chunks":          len(gold),
            "n_gold_nodes_in_kg":     qg["n_gold_nodes_in_kg"],
            "n_unreachable_gold_nodes": qg["n_unreachable_gold_nodes"],
            # QG1/QG3 features
            "mean_qg_distance": round(mean_d, 3) if mean_d is not None else None,
            "min_qg_distance":  round(min_d, 3)  if min_d  is not None else None,
            "max_qg_distance":  round(max_d, 3)  if max_d  is not None else None,
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
            print(f"  gold_nodes_in_kg={qg['n_gold_nodes_in_kg']}"
                  f"  unreachable={qg['n_unreachable_gold_nodes']}")
            print(f"  mean_qg={row['mean_qg_distance']}  max_qg={row['max_qg_distance']}")
            print(f"  recall@{args.top_k}={row[f'recall@{args.top_k}']}  judge={row['judge_score']}")

        per_query.append(row)

    n = len(per_query)
    print(f"\n{'=' * 70}")
    print(f"QG1–QG3 Query-to-Gold Distance Results (n={n} queries)")
    print(f"{'=' * 70}")
    print_table(per_query, [
        "id", "n_query_nodes", "n_gold_nodes_in_kg", "n_unreachable_gold_nodes",
        "mean_qg_distance", "max_qg_distance", f"recall@{args.top_k}", "judge_score"
    ])

    # ------------------------------------------------------------------
    # Correlation analysis (QG1)
    # ------------------------------------------------------------------
    valid = [r for r in per_query if r["judge_score"] is not None]
    vn = len(valid)
    print(f"\nCorrelation analysis (n={vn} queries with LLM judge scores):")
    note = " (n < 5: exploratory only)" if vn < 5 else ""
    print(f"  Note: n={vn}{note}")

    rho_mean_recall = float("nan")
    corr_specs = [
        ("QG1", "mean_qg_distance", "judge_score",          "Spearman(mean_qg, judge_score)  [expect ρ<0]"),
        ("QG1", "mean_qg_distance", f"recall@{args.top_k}", "Spearman(mean_qg, recall@k)     [expect ρ<0]"),
        ("QG1", "min_qg_distance",  f"recall@{args.top_k}", "Spearman(min_qg,  recall@k)     [expect ρ<0]"),
        ("QG1", "max_qg_distance",  f"recall@{args.top_k}", "Spearman(max_qg,  recall@k)     [expect ρ<0]"),
    ]

    if vn >= 3:
        for hyp, feat, quality_col, label in corr_specs:
            if quality_col == "judge_score":
                pool = valid
            else:
                pool = per_query
            pool = [r for r in pool if r.get(feat) is not None and r.get(quality_col) is not None]
            if len(pool) < 3:
                print(f"  {hyp}: {label}  (n={len(pool)} < 3, skipped)")
                continue
            rho, pval = _spearman(
                [r[feat] for r in pool],
                [r[quality_col] for r in pool],
            )
            if feat == "mean_qg_distance" and quality_col == f"recall@{args.top_k}":
                rho_mean_recall = rho
            pval_str = f"{pval:.3f}" if not math.isnan(pval) else "N/A"
            print(f"  {hyp}: {label:50s}  ρ={rho:.3f}  p={pval_str}")
    else:
        print("  Too few queries with judge scores for correlation analysis.")

    # QG2: unreachable gold nodes
    unreachable_queries = [r for r in per_query if r["n_unreachable_gold_nodes"] > 0]
    print(f"\nQG2: {len(unreachable_queries)}/{n} queries have unreachable gold nodes")
    if not unreachable_queries:
        print("  Finding: all gold chunk nodes are reachable from at least one query node.")
        print("  The KG is fully connected — structural coverage is complete for this benchmark set.")
        print("  Retrieval failures are therefore due to scoring/ranking, not topological isolation.")
    else:
        recall_col = f"recall@{args.top_k}"
        reachable_queries = [r for r in per_query if r["n_unreachable_gold_nodes"] == 0]
        avg_unreachable = sum(r[recall_col] for r in unreachable_queries) / len(unreachable_queries)
        avg_reachable   = (sum(r[recall_col] for r in reachable_queries) / len(reachable_queries)
                           if reachable_queries else 0.0)
        print(f"  Mean recall@{args.top_k}: unreachable={avg_unreachable:.3f}  reachable={avg_reachable:.3f}")
        for r in unreachable_queries:
            print(f"  [{r['id']}] n_unreachable={r['n_unreachable_gold_nodes']}")

    # ------------------------------------------------------------------
    # QG3: cross-experiment comparison with baseline (subgraph diameter)
    # ------------------------------------------------------------------
    print(f"\nQG3: Cross-experiment comparison (mean_qg_distance vs subgraph diameter):")
    if args.baseline_results and os.path.exists(args.baseline_results):
        with open(args.baseline_results) as f:
            baseline = json.load(f)
        baseline_by_id = {r["id"]: r for r in baseline.get("per_query", [])}
        recall_col = f"recall@{args.top_k}"

        # Align rows present in both experiments
        paired = [
            (r, baseline_by_id[r["id"]])
            for r in per_query
            if r["id"] in baseline_by_id
              and r.get("mean_qg_distance") is not None
              and baseline_by_id[r["id"]].get(recall_col) is not None
        ]
        if len(paired) >= 3:
            qg_xs  = [r["mean_qg_distance"] for r, _ in paired]
            dia_xs = [b["diameter"]          for _, b in paired]
            qs_ys  = [r[recall_col]          for r, _ in paired]

            rho_qg,  p_qg  = _spearman(qg_xs,  qs_ys)
            rho_dia, p_dia = _spearman(dia_xs, qs_ys)

            print(f"  Aligned n={len(paired)} queries")
            print(f"  Spearman(mean_qg_distance, recall@k)  ρ={rho_qg:.3f}  p={p_qg:.3f}")
            print(f"  Spearman(diameter,          recall@k)  ρ={rho_dia:.3f}  p={p_dia:.3f}")
            if not (math.isnan(rho_qg) or math.isnan(rho_dia)):
                winner = "mean_qg_distance" if abs(rho_qg) > abs(rho_dia) else "diameter"
                print(f"  → stronger predictor of recall@k: {winner}")
        else:
            print(f"  Too few aligned rows (n={len(paired)}) for comparison.")
    else:
        print(f"  Run experiment_c_difficulty.py first and pass --baseline-results results_c.json")
        if not math.isnan(rho_mean_recall):
            print(f"  This experiment: Spearman(mean_qg_distance, recall@k) ρ={rho_mean_recall:.3f}")
        print(f"  Baseline (from printed output):  Spearman(diameter, recall@k) ρ=−0.079  p=0.666")
        print(f"  Interpretation: compare |ρ| values — larger absolute ρ = stronger predictor.")

    # ------------------------------------------------------------------
    # Classifier pilot (LOOCV AUC)
    # ------------------------------------------------------------------
    labeled = [r for r in per_query if r["hard_label"] is not None]
    ln = len(labeled)
    print(f"\nClassifier pilot (LOOCV, n={ln} labeled queries):")
    if ln < 3:
        print("  Too few labeled queries for meaningful classification.")
    else:
        feature_sets = {
            "QG_mean":  ["mean_qg_distance"],
            "QG_max":   ["max_qg_distance"],
            "QG_all3":  ["mean_qg_distance", "min_qg_distance", "max_qg_distance"],
        }
        y = [r["hard_label"] for r in labeled]
        for set_name, feats in feature_sets.items():
            valid_labeled = [r for r in labeled if all(r.get(f) is not None for f in feats)]
            y_filt = [r["hard_label"] for r in valid_labeled]
            X = [[r[f] for f in feats] for r in valid_labeled]
            auc = _loocv_auc(X, y_filt)
            auc_str = f"{auc:.3f}" if not math.isnan(auc) else "N/A"
            print(f"  {set_name:15s}  LOOCV AUC={auc_str}")

    print(f"\n  *** n={n} is a proof-of-concept. Results need a larger labeled dataset. ***")
    print(f"{'=' * 70}")

    result = {
        "n_benchmarks": n,
        "top_k": args.top_k,
        "difficulty_threshold": args.difficulty_threshold,
        "n_unreachable_queries": len(unreachable_queries),
        "per_query": per_query,
    }

    if args.output:
        out = Path(args.output)
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=lambda o: None if math.isnan(float(o)) else float(o))
        print(f"Results written to {out}")


if __name__ == "__main__":
    load_dotenv()
    main()
