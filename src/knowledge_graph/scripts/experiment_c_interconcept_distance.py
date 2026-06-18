"""IP1–IP3 — Inter-concept shortest path distance predicts retrieval difficulty.

The original difficulty experiment (C1–C7) measured topology *within* the query
subgraph and found no signal: subgraphs are structurally near-identical across
all 32 benchmarks. A key limitation is that the subgraph is built from query
nodes only, so its diameter reflects the local neighbourhood, not how far apart
the query concepts are from one another in the full knowledge graph.

This experiment measures the pairwise shortest path between every pair of query
concept nodes in the *full* KG. The rationale: when a query involves two concepts
that are topologically distant (many hops apart), the system must traverse a
longer reasoning path to connect them, making retrieval harder.

Hypotheses:
  IP1: Mean pairwise shortest path between query concept nodes correlates
       negatively with judge_score and recall@k — distant concepts signal a
       harder query because the graph provides no short bridge between topics.
  IP2: Queries where at least one concept pair is disconnected (no path)
       show worse recall than fully connected queries. On the current KG the
       graph is fully connected, so this hypothesis will be trivially satisfied
       (0 disconnected pairs across all benchmarks) — this is itself a finding:
       the KG covers all topic pairs in the benchmark set without structural gaps.
  IP3: Max inter-concept distance is a stronger predictor than mean distance,
       because the hardest concept pair (worst-case reasoning depth) drives
       retrieval failure more than the average.

Expected correlations (Spearman ρ):
  IP1: negative ρ between mean_distance and judge_score / recall@k.
  IP3: |ρ(max_distance)| > |ρ(mean_distance)|.

Usage:
    python -m src.knowledge_graph.scripts.experiment_c_interconcept_distance \\
        --run-dir data/knowledge_graph/runs/latest \\
        --output results_c_interconcept_distance.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from itertools import combinations
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


def _compute_interconcept_distances(
    query_nodes: list[str],
    graph,
) -> tuple[float, float, float, int, int]:
    """Return (mean_dist, max_dist, min_dist, n_disconnected, n_pairs).

    For a single-node query there are no pairs; returns (0, 0, 0, 0, 0)
    so that single-concept queries appear as the trivially-focused (easiest)
    case in correlation analysis rather than as missing data.

    Uses nx.shortest_path_length for each pair. The guard via nx.has_path is
    retained for robustness even though the current graph is fully connected.
    """
    import networkx as nx

    pairs = list(combinations(query_nodes, 2))
    n_pairs = len(pairs)
    if n_pairs == 0:
        return 0.0, 0.0, 0.0, 0, 0

    distances: list[int] = []
    n_disconnected = 0
    for u, v in pairs:
        # Graph is currently fully connected; guard retained for robustness.
        if nx.has_path(graph, u, v):
            distances.append(nx.shortest_path_length(graph, u, v))
        else:
            n_disconnected += 1

    if not distances:
        # All pairs disconnected — distances undefined.
        return float("inf"), float("inf"), float("inf"), n_disconnected, n_pairs

    mean_d = sum(distances) / len(distances)
    return mean_d, float(max(distances)), float(min(distances)), n_disconnected, n_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IP1-IP3: Inter-concept shortest path distance vs. retrieval quality.",
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
                             "inter-concept distances with no retrieval eval.")
    parser.add_argument("--unanswerable-output", default=None,
                        help="Output path for unanswerable-query distance results.")
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
            print(f"  [{bm['id']}] No query nodes matched — distances set to 0")

        mean_d, max_d, min_d, n_disconnected, n_pairs = _compute_interconcept_distances(
            query_nodes, graph
        )

        # Represent all-disconnected case as None in the output row (excluded from correlations).
        mean_d_out = round(mean_d, 3) if not math.isinf(mean_d) else None
        max_d_out  = round(max_d, 3)  if not math.isinf(max_d)  else None
        min_d_out  = round(min_d, 3)  if not math.isinf(min_d)  else None

        row = {
            "id": bm["id"],
            "n_query_nodes":       len(query_nodes),
            "n_concept_pairs":     n_pairs,
            "n_disconnected_pairs": n_disconnected,
            # IP1/IP3 features
            "mean_distance": mean_d_out,
            "max_distance":  max_d_out,
            "min_distance":  min_d_out,
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
            print(f"  query_nodes={query_nodes}  n_pairs={n_pairs}")
            print(f"  mean_dist={mean_d_out}  max_dist={max_d_out}  disconnected={n_disconnected}")
            print(f"  recall@{args.top_k}={row[f'recall@{args.top_k}']}  judge={row['judge_score']}")

        per_query.append(row)

    n = len(per_query)
    print(f"\n{'=' * 70}")
    print(f"IP1–IP3 Inter-concept Distance Results (n={n} queries)")
    print(f"{'=' * 70}")
    print_table(per_query, [
        "id", "n_query_nodes", "n_concept_pairs", "mean_distance",
        "max_distance", f"recall@{args.top_k}", "judge_score", "hard_label"
    ])

    # ------------------------------------------------------------------
    # Correlation analysis (IP1, IP3)
    # ------------------------------------------------------------------
    valid = [r for r in per_query if r["judge_score"] is not None]
    vn = len(valid)
    print(f"\nCorrelation analysis (n={vn} queries with LLM judge scores):")
    note = " (n < 5: exploratory only)" if vn < 5 else ""
    print(f"  Note: n={vn}{note}")

    rho_mean_judge = float("nan")
    rho_max_judge  = float("nan")

    corr_specs = [
        ("IP1", "mean_distance", "judge_score",          "Spearman(mean_dist, judge_score)  [expect ρ<0]"),
        ("IP3", "max_distance",  "judge_score",          "Spearman(max_dist,  judge_score)  [expect ρ<0]"),
        ("IP1", "mean_distance", f"recall@{args.top_k}", "Spearman(mean_dist, recall@k)     [expect ρ<0]"),
        ("IP3", "max_distance",  f"recall@{args.top_k}", "Spearman(max_dist,  recall@k)     [expect ρ<0]"),
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
            if feat == "mean_distance" and quality_col == "judge_score":
                rho_mean_judge = rho
            if feat == "max_distance" and quality_col == "judge_score":
                rho_max_judge = rho
            pval_str = f"{pval:.3f}" if not math.isnan(pval) else "N/A"
            print(f"  {hyp}: {label:50s}  ρ={rho:.3f}  p={pval_str}")
    else:
        print("  Too few queries with judge scores for correlation analysis.")

    # IP3 head-to-head comparison
    if not (math.isnan(rho_mean_judge) or math.isnan(rho_max_judge)):
        winner = "max_distance" if abs(rho_max_judge) > abs(rho_mean_judge) else "mean_distance"
        print(f"\n  IP3: max_dist ρ={rho_max_judge:.3f} vs mean_dist ρ={rho_mean_judge:.3f}"
              f"  → stronger predictor: {winner}")

    # IP2: disconnected concept pairs
    disconnected_queries = [r for r in per_query if r["n_disconnected_pairs"] > 0]
    connected_queries    = [r for r in per_query if r["n_disconnected_pairs"] == 0]
    print(f"\nIP2: {len(disconnected_queries)}/{n} queries have disconnected concept pairs")
    if not disconnected_queries:
        print("  Finding: all query concept pairs are connected in the current KG.")
        print("  The graph has no topological gaps between benchmark topic pairs.")
    else:
        recall_col = f"recall@{args.top_k}"
        avg_disc = sum(r[recall_col] for r in disconnected_queries) / len(disconnected_queries)
        avg_conn = sum(r[recall_col] for r in connected_queries) / len(connected_queries) if connected_queries else 0.0
        print(f"  Mean recall@{args.top_k}: disconnected={avg_disc:.3f}  connected={avg_conn:.3f}")
        for r in disconnected_queries:
            print(f"  [{r['id']}] n_disconnected_pairs={r['n_disconnected_pairs']}")

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
            "IP_mean_dist":  ["mean_distance"],
            "IP_max_dist":   ["max_distance"],
            "IP_combined":   ["mean_distance", "max_distance"],
        }
        y = [r["hard_label"] for r in labeled]
        ip3_max_stronger = False
        auc_mean = auc_max = float("nan")
        for set_name, feats in feature_sets.items():
            valid_labeled = [r for r in labeled if all(r.get(f) is not None for f in feats)]
            y_filt = [r["hard_label"] for r in valid_labeled]
            X = [[r[f] for f in feats] for r in valid_labeled]
            auc = _loocv_auc(X, y_filt)
            if set_name == "IP_mean_dist":
                auc_mean = auc
            elif set_name == "IP_max_dist":
                auc_max = auc
            auc_str = f"{auc:.3f}" if not math.isnan(auc) else "N/A"
            print(f"  {set_name:20s}  LOOCV AUC={auc_str}")
        if not (math.isnan(auc_mean) or math.isnan(auc_max)):
            ip3_max_stronger = auc_max > auc_mean
            print(f"\n  IP3 (AUC): max_dist={auc_max:.3f} vs mean_dist={auc_mean:.3f}"
                  f"  → max stronger: {ip3_max_stronger}")

    print(f"\n  *** n={n} is a proof-of-concept. Results need a larger labeled dataset. ***")
    print(f"{'=' * 70}")

    result = {
        "n_benchmarks": n,
        "top_k": args.top_k,
        "difficulty_threshold": args.difficulty_threshold,
        "n_disconnected_queries": len(disconnected_queries),
        "n_fully_connected_queries": len(connected_queries),
        "per_query": per_query,
    }

    if args.output:
        out = Path(args.output)
        with open(out, "w") as f:
            json.dump(result, f, indent=2,
                      default=lambda o: None if math.isnan(float(o)) else float(o))
        print(f"Results written to {out}")

    # ------------------------------------------------------------------
    # Optional: compute inter-concept distances for unanswerable benchmarks
    # ------------------------------------------------------------------
    if args.unanswerable_benchmarks and args.unanswerable_output:
        unans_bms = load_benchmarks(args.unanswerable_benchmarks)
        print(f"\nComputing inter-concept distances for "
              f"{len(unans_bms)} unanswerable benchmarks...")

        unans_rows: list[dict] = []
        for bm in unans_bms:
            query = bm["question"]
            query_nodes = extract_query_nodes(query, graph, canonical_lookup)
            if not query_nodes and args.verbose:
                print(f"  [{bm['id']}] No query nodes matched — distances set to 0")

            mean_d, max_d, min_d, n_disc, n_pairs = _compute_interconcept_distances(
                query_nodes, graph
            )
            mean_d_out = round(mean_d, 3) if not math.isinf(mean_d) else None
            max_d_out  = round(max_d, 3)  if not math.isinf(max_d)  else None
            min_d_out  = round(min_d, 3)  if not math.isinf(min_d)  else None

            row = {
                "id": bm["id"],
                "n_query_nodes":        len(query_nodes),
                "n_concept_pairs":      n_pairs,
                "n_disconnected_pairs": n_disc,
                "mean_distance": mean_d_out,
                "max_distance":  max_d_out,
                "min_distance":  min_d_out,
            }
            if args.verbose:
                print(f"  [{bm['id']}] nodes={len(query_nodes)}"
                      f"  pairs={n_pairs}"
                      f"  mean={mean_d_out}  max={max_d_out}")
            unans_rows.append(row)

        unans_result = {
            "n_benchmarks": len(unans_rows),
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
