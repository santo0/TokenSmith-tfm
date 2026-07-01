"""experiment_c_series.py — Series C unified analysis (5.8.2 / 5.8.3 / 5.8.4).

Two input modes:

  JSON mode (pre-computed files):
    python -m src.knowledge_graph.scripts.experiment_c_series \\
        --topology        results_c.json \\
        --topology-unans  results_c_unanswerable.json \\
        --gc              results_c_global_centrality.json \\
        --gc-unans        results_c_global_centrality_unanswerable.json \\
        --ip              results_c_interconcept_distance.json \\
        --ip-unans        results_c_interconcept_distance_unanswerable.json \\
        --benchmarks      tests/benchmarks.yaml \\
        --b2              results_b2.json \\
        --output          results_c_series.json

  Run-dir mode (compute features from a KG run directory):
    python -m src.knowledge_graph.scripts.experiment_c_series \\
        --run-dir         data/knowledge_graph/runs/latest \\
        --benchmarks      tests/benchmarks.yaml \\
        --benchmarks-unans tests/benchmarks_unanswerable.yaml \\
        --b2              results_b2.json \\
        --output          results_c_series.json

  Canonicalization comparison (both run dirs):
    python -m src.knowledge_graph.scripts.experiment_c_series \\
        --run-dir         data/knowledge_graph/runs/latest \\
        --raw-run-dir     data/knowledge_graph/runs/2026-06-17_11-58-04 \\
        --benchmarks      tests/benchmarks.yaml \\
        --benchmarks-unans tests/benchmarks_unanswerable.yaml \\
        --b2              results_b2.json \\
        --output          results_c_series.json \\
        --raw-output      results_c_series_raw.json

Feature set (10 features, max_distance dropped):
  topology  — diameter, density, coverage_delta, community_span,
               community_dispersion, local_mean_betweenness
  ip        — mean_distance
  gc        — mean_pagerank, gc_mean_betweenness, mean_degree

Statistical pass: Mann-Whitney U × 3 pairs, Holm per feature,
Cohen's d = (mean_a − mean_b) / pooled_sd.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Feature constants
# ---------------------------------------------------------------------------

# Internal key names used throughout; source JSON keys may differ (see loaders).
FEATURES_TOPOLOGY = [
    "diameter",
    "density",
    "coverage_delta",
    "community_span",
    "community_dispersion",
    "local_mean_betweenness",
]

FEATURE_IP = ["mean_distance"]

FEATURES_GC = ["mean_pagerank", "gc_mean_betweenness", "mean_degree"]

ALL_FEATURES = FEATURES_TOPOLOGY + FEATURE_IP + FEATURES_GC

FEATURE_FAMILY: dict[str, str] = {
    **{f: "topology" for f in FEATURES_TOPOLOGY},
    **{f: "ip"       for f in FEATURE_IP},
    **{f: "gc"       for f in FEATURES_GC},
}

PAIRS = [
    ("easy", "hard"),
    ("easy", "unanswerable"),
    ("hard", "unanswerable"),
]


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _mannwhitney(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U; returns (U_stat, p_value)."""
    from scipy.stats import mannwhitneyu
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def _cohens_d(a: list[float], b: list[float]) -> float:
    """d = (mean_a − mean_b) / pooled_sd."""
    import numpy as np
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    pooled = math.sqrt(
        ((na - 1) * float(np.var(a, ddof=1)) + (nb - 1) * float(np.var(b, ddof=1)))
        / (na + nb - 2)
    )
    if pooled == 0:
        return float("nan")
    return (float(np.mean(a)) - float(np.mean(b))) / pooled


def _holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction; returns corrected p in original order."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    corrected = [0.0] * n
    prev = 0.0
    for rank, i in enumerate(order):
        c = min(1.0, p_values[i] * (n - rank))
        corrected[i] = max(c, prev)
        prev = corrected[i]
    return corrected


def _descriptive(vals: list[float]) -> dict[str, float]:
    import numpy as np
    if not vals:
        return {}
    arr = np.array(vals, dtype=float)
    return {
        "n":      len(vals),
        "mean":   float(np.mean(arr)),
        "sd":     float(np.std(arr, ddof=1)) if len(vals) > 1 else 0.0,
        "median": float(np.median(arr)),
    }


def _spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    from scipy.stats import spearmanr
    if len(xs) < 3:
        return float("nan"), float("nan")
    result = spearmanr(xs, ys)
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        root = Path(__file__).parent.parent.parent.parent
        p = root / path
    return p


def _load_json_per_query(path: str) -> list[dict]:
    with open(_resolve(path)) as f:
        return json.load(f)["per_query"]


def _load_benchmark_labels(path: str) -> dict[str, str]:
    """Return {query_id: 'easy'|'hard'} for benchmarks with a difficulty field."""
    import yaml
    with open(_resolve(path)) as f:
        data = yaml.safe_load(f)
    result: dict[str, str] = {}
    for entry in data.get("benchmarks", []):
        diff = entry.get("difficulty")
        if diff in ("easy", "hard"):
            result[entry["id"]] = diff
    return result


def _build_feature_rows(
    topology_ans:  list[dict],
    topology_unans: list[dict],
    gc_ans:        list[dict],
    gc_unans:      list[dict],
    ip_ans:        list[dict],
    ip_unans:      list[dict],
    labels:        dict[str, str],
) -> dict[str, dict]:
    """Build {query_id: {feature: value, ..., 'group': label}} for all 40 queries."""
    rows: dict[str, dict] = {}

    def _upsert(qid: str, updates: dict) -> None:
        if qid not in rows:
            rows[qid] = {"id": qid}
        rows[qid].update(updates)

    for r in topology_ans:
        qid = r["id"]
        _upsert(qid, {
            "diameter":              r.get("diameter"),
            "density":               r.get("density"),
            "coverage_delta":        r.get("coverage_delta"),
            "community_span":        r.get("community_span"),
            "community_dispersion":  r.get("community_dispersion"),
            "local_mean_betweenness": r.get("mean_betweenness"),
        })

    for r in topology_unans:
        qid = r["id"]
        _upsert(qid, {
            "diameter":              r.get("diameter"),
            "density":               r.get("density"),
            "coverage_delta":        r.get("coverage_delta"),
            "community_span":        r.get("community_span"),
            "community_dispersion":  r.get("community_dispersion"),
            "local_mean_betweenness": r.get("mean_betweenness"),
        })

    for r in gc_ans + gc_unans:
        qid = r["id"]
        _upsert(qid, {
            "mean_pagerank":       r.get("mean_pagerank"),
            "gc_mean_betweenness": r.get("mean_betweenness"),
            "mean_degree":         r.get("mean_degree"),
        })

    for r in ip_ans + ip_unans:
        qid = r["id"]
        _upsert(qid, {"mean_distance": r.get("mean_distance")})

    # Assign group labels
    for qid in rows:
        if qid in labels:
            rows[qid]["group"] = labels[qid]
        else:
            rows[qid]["group"] = "unanswerable"

    return rows


def _group_vals(rows: dict[str, dict], group: str, feature: str) -> list[float]:
    return [
        float(r[feature])
        for r in rows.values()
        if r.get("group") == group and r.get(feature) is not None
    ]


# ---------------------------------------------------------------------------
# Pairwise analysis pass (one run over all features)
# ---------------------------------------------------------------------------

def _pairwise_pass(rows: dict[str, dict]) -> list[dict]:
    """Run MWU + Holm + Cohen's d for every feature × pair combination.

    Returns a flat list of result dicts, one per (feature, group_a, group_b).
    Holm correction is applied within each feature's three-contrast family.
    Sign convention: d = mean(group_a) − mean(group_b) / pooled_sd.
    """
    results: list[dict] = []
    for feat in ALL_FEATURES:
        group_vals = {g: _group_vals(rows, g, feat) for g in ("easy", "hard", "unanswerable")}
        raw_ps: list[float] = []
        pair_stats: list[dict] = []
        for ga, gb in PAIRS:
            U, p_raw = _mannwhitney(group_vals[ga], group_vals[gb])
            d = _cohens_d(group_vals[ga], group_vals[gb])
            raw_ps.append(p_raw if not math.isnan(p_raw) else 1.0)
            pair_stats.append({
                "feature":  feat,
                "family":   FEATURE_FAMILY[feat],
                "group_a":  ga,
                "group_b":  gb,
                "n_a":      len(group_vals[ga]),
                "n_b":      len(group_vals[gb]),
                "mean_a":   float(sum(group_vals[ga]) / len(group_vals[ga])) if group_vals[ga] else float("nan"),
                "mean_b":   float(sum(group_vals[gb]) / len(group_vals[gb])) if group_vals[gb] else float("nan"),
                "U":        U,
                "p_raw":    p_raw,
                "d":        d,
            })
        corrected = _holm_correct(raw_ps)
        for stat, p_holm in zip(pair_stats, corrected):
            stat["p_holm"] = p_holm
            stat["survives_holm"] = (not math.isnan(p_holm)) and p_holm < 0.05
            results.append(stat)

    return results


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v: Any, dec: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{dec}f}"


def _sig(p: float) -> str:
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return "ns"


def _sec(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def _pw_row(row: dict, show_feature: bool = True) -> None:
    feat  = row["feature"] if show_feature else " " * len(row["feature"])
    pair  = f"{row['group_a']} / {row['group_b']}"
    surv  = "YES" if row["survives_holm"] else "no"
    print(f"  {feat:<28} {pair:<22} U={_fmt(row['U'],1):>8}  "
          f"p_raw={_fmt(row['p_raw'],4):>7}  p_holm={_fmt(row['p_holm'],4):>7}  "
          f"d={_fmt(row['d'],3):>7}  holm={surv}")


# ---------------------------------------------------------------------------
# Section 5.8.2 — unanswerable contrasts
# ---------------------------------------------------------------------------

def _report_582(pw: list[dict], rows: dict[str, dict]) -> None:
    _sec("5.8.2  Detecting unanswerable queries")
    print("  Contrasts involving the unanswerable group (from the unified pairwise pass).\n")

    # Group means table
    header = f"  {'Feature':<28} {'easy mean':>10} {'hard mean':>10} {'unans mean':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for feat in ALL_FEATURES:
        e = _descriptive(_group_vals(rows, "easy",         feat))
        h = _descriptive(_group_vals(rows, "hard",         feat))
        u = _descriptive(_group_vals(rows, "unanswerable", feat))
        fam = f"[{FEATURE_FAMILY[feat]}]"
        print(f"  {feat:<22} {fam:<6} "
              f"{_fmt(e.get('mean')):>10} "
              f"{_fmt(h.get('mean')):>10} "
              f"{_fmt(u.get('mean')):>11}")

    print()
    print(f"  {'Feature':<28} {'Contrast':<22} {'U':>8}  {'p_raw':>7}  {'p_holm':>7}  "
          f"{'d':>7}  holm")
    print("  " + "-" * 88)

    unans_rows_pw = [r for r in pw if r["group_b"] == "unanswerable"]
    last = None
    for row in unans_rows_pw:
        show = row["feature"] != last
        _pw_row(row, show_feature=show)
        last = row["feature"]


# ---------------------------------------------------------------------------
# Section 5.8.3 — easy/hard contrast + mean_distance details + circularity
# ---------------------------------------------------------------------------

def _report_583(pw: list[dict], rows: dict[str, dict],
                b2_recall: dict[str, float] | None) -> dict:
    _sec("5.8.3  Grading difficulty within the answerable set")
    print("  Easy/hard contrast for all ten features.\n")

    print(f"  {'Feature':<28} {'Contrast':<22} {'U':>8}  {'p_raw':>7}  {'p_holm':>7}  "
          f"{'d':>7}  holm")
    print("  " + "-" * 88)

    eh_rows = [r for r in pw if r["group_a"] == "easy" and r["group_b"] == "hard"]
    for row in eh_rows:
        _pw_row(row)

    # mean_distance three-group table
    print("\n  mean_distance three-group means (SD):")
    for grp in ("easy", "hard", "unanswerable"):
        d = _descriptive(_group_vals(rows, grp, "mean_distance"))
        print(f"    {grp:<14} n={d.get('n','?')}  mean={_fmt(d.get('mean'))}"
              f"  sd={_fmt(d.get('sd'))}")

    # Circularity check
    circularity: dict = {}
    if b2_recall:
        ans_ids = [qid for qid, r in rows.items() if r.get("group") in ("easy", "hard")]
        paired = [
            (rows[qid]["mean_distance"], b2_recall[qid])
            for qid in ans_ids
            if rows[qid].get("mean_distance") is not None and qid in b2_recall
        ]
        n_circ = len(paired)
        if paired:
            xs = [p[0] for p in paired]
            ys = [p[1] for p in paired]
            rho, p_rho = _spearman(xs, ys)
            print(f"\n  Circularity check — Spearman(mean_distance, FAISS dense_R@10):")
            print(f"    n={n_circ}  rho={_fmt(rho)}  p={_fmt(p_rho)}")
            if not math.isnan(rho):
                interp = ("negative as expected: higher distance -> lower recall"
                          if rho < -0.1
                          else ("near-zero: topology distance does not predict "
                                "measured retrieval quality" if abs(rho) < 0.1
                                else "positive — unexpected direction, inspect"))
                print(f"    Interpretation: {interp}")
            circularity = {"n": n_circ, "rho": rho, "p": p_rho, "pairs": paired}
        else:
            print("\n  Circularity check: no overlapping IDs between IP and B2 results.")
    else:
        print("\n  Circularity check skipped (no --b2 file provided).")

    return circularity


# ---------------------------------------------------------------------------
# Section 5.8.4 — GC null confirmation
# ---------------------------------------------------------------------------

def _report_584(pw: list[dict], rows: dict[str, dict]) -> None:
    _sec("5.8.4  What carries no signal (GC features)")

    gc_pw = [r for r in pw if r["family"] == "gc"]
    survivors = [r for r in gc_pw if r["survives_holm"]]
    print(f"  GC features: {FEATURES_GC}")
    print(f"  Total GC contrasts: {len(gc_pw)}  |  Holm survivors: {len(survivors)}\n")

    print(f"  {'Feature':<28} {'Contrast':<22} {'U':>8}  {'p_raw':>7}  {'p_holm':>7}  "
          f"{'d':>7}  holm")
    print("  " + "-" * 88)
    last = None
    for row in gc_pw:
        show = row["feature"] != last
        _pw_row(row, show_feature=show)
        last = row["feature"]

    print("\n  Descriptive means (easy / hard / unanswerable):")
    print(f"  {'Feature':<28} {'easy':>10} {'hard':>10} {'unanswerable':>14}")
    print("  " + "-" * 64)
    for feat in FEATURES_GC:
        e = _descriptive(_group_vals(rows, "easy",         feat))
        h = _descriptive(_group_vals(rows, "hard",         feat))
        u = _descriptive(_group_vals(rows, "unanswerable", feat))
        print(f"  {feat:<28} {_fmt(e.get('mean')):>10} "
              f"{_fmt(h.get('mean')):>10} {_fmt(u.get('mean')):>14}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _summary_table(pw: list[dict]) -> list[dict]:
    """One row per feature: family, best surviving contrast, its d, Holm survival."""
    _sec("Summary table — all ten features")
    print(f"  {'Feature':<28} {'Family':<10} {'Best contrast':<22} "
          f"{'d':>7}  {'Holm'}  {'p_holm':>7}")
    print("  " + "-" * 80)

    summary_rows: list[dict] = []
    for feat in ALL_FEATURES:
        feat_rows = [r for r in pw if r["feature"] == feat]
        survivors = [r for r in feat_rows if r["survives_holm"]]
        if survivors:
            best = min(survivors, key=lambda r: r["p_holm"])
        else:
            best = min(feat_rows, key=lambda r: r["p_holm"] if not math.isnan(r["p_holm"]) else 1.0)
        any_holm = bool(survivors)
        contrast = f"{best['group_a']} / {best['group_b']}"
        print(f"  {feat:<28} {FEATURE_FAMILY[feat]:<10} {contrast:<22} "
              f"{_fmt(best['d']):>7}  {'YES' if any_holm else 'no':<5}  "
              f"{_fmt(best['p_holm']):>7}")
        summary_rows.append({
            "feature":        feat,
            "family":         FEATURE_FAMILY[feat],
            "best_contrast":  contrast,
            "best_d":         best["d"],
            "best_p_holm":    best["p_holm"],
            "survives_holm":  any_holm,
        })

    print()
    print("  Synthesis:")
    ip_topo_survivors = [r for r in summary_rows
                         if r["survives_holm"] and r["family"] in ("ip", "topology")]
    gc_survivors = [r for r in summary_rows if r["survives_holm"] and r["family"] == "gc"]
    md_row = next((r for r in summary_rows if r["feature"] == "mean_distance"), None)
    md_eh = next((r for r in pw if r["feature"] == "mean_distance"
                  and r["group_a"] == "easy" and r["group_b"] == "hard"), None)

    print(f"  IP/topology features surviving Holm on ≥1 unanswerable contrast: "
          f"{[r['feature'] for r in ip_topo_survivors]}")
    if md_row and md_eh and md_eh["survives_holm"]:
        print(f"  mean_distance additionally survives easy/hard "
              f"(d={_fmt(md_eh['d'])}, p_holm={_fmt(md_eh['p_holm'])}).")
    print(f"  GC features surviving Holm: {[r['feature'] for r in gc_survivors]} "
          f"({'null confirmed' if not gc_survivors else 'unexpected — inspect'}).")

    return summary_rows


# ---------------------------------------------------------------------------
# Scatter plot (circularity check)
# ---------------------------------------------------------------------------

def _plot_scatter(pairs: list[tuple[float, float]],
                  rows: dict[str, dict],
                  b2_recall: dict[str, float],
                  rho: float, p_rho: float,
                  out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colours = {"easy": "#27ae60", "hard": "#e67e22"}
        markers = {"easy": "o", "hard": "s"}

        fig, ax = plt.subplots(figsize=(6, 5))
        for grp in ("easy", "hard"):
            grp_ids = [qid for qid, r in rows.items() if r.get("group") == grp]
            xs = [rows[qid]["mean_distance"] for qid in grp_ids
                  if rows[qid].get("mean_distance") is not None and qid in b2_recall]
            ys = [b2_recall[qid] for qid in grp_ids
                  if rows[qid].get("mean_distance") is not None and qid in b2_recall]
            ax.scatter(xs, ys, c=colours[grp], marker=markers[grp],
                       label=grp.capitalize(), s=60, alpha=0.85,
                       edgecolors="white", linewidths=0.6, zorder=3)

        rho_str = _fmt(rho) if not math.isnan(rho) else "n/a"
        p_str   = _fmt(p_rho) if not math.isnan(p_rho) else "n/a"
        ax.set_xlabel("mean_distance (inter-concept shortest path)", fontsize=10)
        ax.set_ylabel("FAISS dense recall@10", fontsize=10)
        ax.set_title(f"Circularity check\nSpearman ρ = {rho_str}, p = {p_str}",
                     fontsize=10)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Scatter saved to {out_path}")
    except ImportError:
        print("matplotlib not available — scatter not saved.")


# ---------------------------------------------------------------------------
# Graph-based feature computation (run-dir mode)
# ---------------------------------------------------------------------------

def _load_graph_from_run_dir(run_dir: str):
    """Load graph and optional canonical lookup from a KG run directory."""
    from src.knowledge_graph.io import (
        load_graph_and_chunks, load_canonicalization_data, resolve_run_dir,
    )
    from src.knowledge_graph.query import CanonicalLookup

    root = Path(__file__).parent.parent.parent.parent
    run_path = Path(run_dir) if Path(run_dir).is_absolute() else root / run_dir
    print(f"  Loading graph from {run_path}...")
    graph, _ = load_graph_and_chunks(str(run_path))
    resolved = resolve_run_dir(str(run_path))
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = CanonicalLookup(syn_table, can_kw, can_emb) if syn_table else None
    print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
          + (" (canonicalized)" if canonical_lookup else " (raw)"))
    return graph, canonical_lookup


def _compute_all_features_for_queries(
    graph,
    canonical_lookup,
    ans_benchmarks: list[dict],
    unans_benchmarks: list[dict],
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Compute all 10 features for every query in one pass.

    Returns (answerable_rows, unanswerable_rows). Each row contains all
    topology + GC + IP feature keys in the same internal format as
    _build_feature_rows().
    """
    import networkx as nx
    from itertools import combinations

    from src.knowledge_graph.analysis import (
        compute_difficulty_features, extract_query_subgraph,
        _get_or_compute_communities,
    )
    from src.knowledge_graph.query import extract_query_nodes

    # Pre-compute global centrality measures once
    n_nodes = graph.number_of_nodes()
    print("  Computing PageRank...")
    pagerank: dict = nx.pagerank(graph)
    k_bc = min(500, n_nodes)
    print(f"  Computing betweenness centrality (k={k_bc})...")
    betweenness: dict = nx.betweenness_centrality(graph, k=k_bc, seed=42)

    # Community map for community_span
    community_map: dict | None = None
    try:
        community_map = _get_or_compute_communities(graph)
        print(f"  Communities: {len(set(community_map.values()))}")
    except Exception as e:
        print(f"  Community detection failed: {e} — community_span set to 0")

    def _row_for_query(bm: dict) -> dict:
        query = bm["question"]
        query_nodes = extract_query_nodes(query, graph, canonical_lookup)

        # ── Topology features ──────────────────────────────────────────────
        feats = compute_difficulty_features(
            query, graph, canonical_lookup, community_map=community_map
        )
        # local_mean_betweenness: betweenness within the query subgraph
        local_bc = 0.0
        if query_nodes:
            sg = extract_query_subgraph(query_nodes, graph)
            if sg.number_of_nodes() > 1:
                try:
                    bc = nx.betweenness_centrality(sg)
                    local_bc = sum(bc.get(n, 0.0) for n in query_nodes) / len(query_nodes)
                except Exception:
                    pass
        # community_span
        community_span = 0
        if community_map and query_nodes:
            community_span = len({community_map[n] for n in query_nodes
                                   if n in community_map})

        # ── GC features ───────────────────────────────────────────────────
        if query_nodes:
            pr_vals  = [pagerank.get(n, 0.0)    for n in query_nodes]
            bc_vals  = [betweenness.get(n, 0.0) for n in query_nodes]
            deg_vals = [graph.degree(n)          for n in query_nodes]
            mean_pr  = sum(pr_vals)  / len(pr_vals)
            mean_bc  = sum(bc_vals)  / len(bc_vals)
            mean_deg = sum(deg_vals) / len(deg_vals)
        else:
            mean_pr = mean_bc = mean_deg = 0.0

        # ── IP feature ────────────────────────────────────────────────────
        pairs = list(combinations(query_nodes, 2)) if query_nodes else []
        distances: list[int] = []
        for u, v in pairs:
            if nx.has_path(graph, u, v):
                distances.append(nx.shortest_path_length(graph, u, v))
        mean_dist: float | None = (
            round(sum(distances) / len(distances), 3) if distances else
            (0.0 if not pairs else None)
        )

        if verbose:
            print(f"    [{bm['id']}] nodes={len(query_nodes)}  "
                  f"diameter={feats.subgraph_diameter}  "
                  f"mean_dist={mean_dist}  mean_deg={mean_deg:.0f}")

        return {
            "id":                    bm["id"],
            # topology
            "diameter":              feats.subgraph_diameter,
            "density":               feats.edge_density,
            "coverage_delta":        feats.corpus_coverage,
            "community_span":        community_span,
            "community_dispersion":  feats.community_dispersion,
            "local_mean_betweenness": round(local_bc, 6),
            # ip
            "mean_distance":         mean_dist,
            # gc
            "mean_pagerank":         round(mean_pr, 8),
            "gc_mean_betweenness":   round(mean_bc, 6),
            "mean_degree":           round(mean_deg, 3),
        }

    print(f"  Computing features for {len(ans_benchmarks)} answerable queries...")
    ans_rows = [_row_for_query(bm) for bm in ans_benchmarks]
    print(f"  Computing features for {len(unans_benchmarks)} unanswerable queries...")
    unans_rows = [_row_for_query(bm) for bm in unans_benchmarks]
    return ans_rows, unans_rows


def _load_yaml_benchmarks(path: str) -> list[dict]:
    import yaml
    with open(_resolve(path)) as f:
        data = yaml.safe_load(f)
    return data.get("benchmarks", [])


def _rows_from_run_dir(
    run_dir: str,
    benchmarks_path: str,
    benchmarks_unans_path: str,
    labels: dict[str, str],
    verbose: bool = False,
) -> dict[str, dict]:
    """Load graph, compute features, return a flat rows dict with group labels.

    In run-dir mode all features are already in canonical key names
    (local_mean_betweenness, gc_mean_betweenness) so we bypass
    _build_feature_rows and assign group labels directly.
    """
    graph, canonical_lookup = _load_graph_from_run_dir(run_dir)
    ans_bms   = [bm for bm in _load_yaml_benchmarks(benchmarks_path)
                 if bm["id"] in labels]
    unans_bms = _load_yaml_benchmarks(benchmarks_unans_path)

    ans_feat_rows, unans_feat_rows = _compute_all_features_for_queries(
        graph, canonical_lookup, ans_bms, unans_bms, verbose=verbose
    )

    rows: dict[str, dict] = {}
    for r in ans_feat_rows:
        rows[r["id"]] = {**r, "group": labels.get(r["id"], "unknown")}
    for r in unans_feat_rows:
        rows[r["id"]] = {**r, "group": "unanswerable"}
    return rows


# ---------------------------------------------------------------------------
# Canonicalization comparison
# ---------------------------------------------------------------------------

def _print_comparison(
    pw_can: list[dict],
    pw_raw: list[dict],
    label_can: str = "canonical",
    label_raw: str = "raw",
) -> list[dict]:
    """Print a side-by-side comparison of d and Holm survival for both graphs."""
    _sec(f"Canonicalization comparison: {label_can} vs {label_raw}")

    raw_by_key = {(r["feature"], r["group_a"], r["group_b"]): r for r in pw_raw}

    header = (f"  {'Feature':<28} {'Contrast':<22} "
              f"{'d_can':>7}  {'holm_can':>9}  "
              f"{'d_raw':>7}  {'holm_raw':>9}  "
              f"{'Δd':>7}  change")
    print(header)
    print("  " + "-" * (len(header) - 2))

    comparison_rows: list[dict] = []
    for row_can in pw_can:
        key = (row_can["feature"], row_can["group_a"], row_can["group_b"])
        row_raw = raw_by_key.get(key)
        if row_raw is None:
            continue
        d_can  = row_can["d"]
        d_raw  = row_raw["d"]
        delta  = (d_raw - d_can) if not (math.isnan(d_can) or math.isnan(d_raw)) else float("nan")
        s_can  = "YES" if row_can["survives_holm"] else "no"
        s_raw  = "YES" if row_raw["survives_holm"] else "no"

        # Highlight changes in Holm survival
        if row_can["survives_holm"] and not row_raw["survives_holm"]:
            change = "lost"
        elif not row_can["survives_holm"] and row_raw["survives_holm"]:
            change = "gained"
        elif abs(delta) < 0.05 if not math.isnan(delta) else True:
            change = "stable"
        else:
            change = "shifted"

        pair = f"{row_can['group_a']} / {row_can['group_b']}"
        print(f"  {row_can['feature']:<28} {pair:<22} "
              f"{_fmt(d_can):>7}  {s_can:>9}  "
              f"{_fmt(d_raw):>7}  {s_raw:>9}  "
              f"{_fmt(delta, 3):>7}  {change}")

        comparison_rows.append({
            "feature":      row_can["feature"],
            "family":       row_can["family"],
            "group_a":      row_can["group_a"],
            "group_b":      row_can["group_b"],
            "d_canonical":  d_can,
            "d_raw":        d_raw,
            "delta_d":      delta,
            "holm_canonical": row_can["survives_holm"],
            "holm_raw":       row_raw["survives_holm"],
            "change":       change,
        })

    # Quick summary
    lost    = [r for r in comparison_rows if r["change"] == "lost"]
    gained  = [r for r in comparison_rows if r["change"] == "gained"]
    stable  = [r for r in comparison_rows if r["change"] == "stable"]
    shifted = [r for r in comparison_rows if r["change"] == "shifted"]
    print(f"\n  Holm survivors lost in raw:    "
          f"{[(r['feature'], r['group_a']+'/'+r['group_b']) for r in lost]}")
    print(f"  Holm survivors gained in raw:  "
          f"{[(r['feature'], r['group_a']+'/'+r['group_b']) for r in gained]}")
    print(f"  Stable (|Δd|<0.05, same Holm): {len(stable)} contrasts")
    print(f"  Shifted (d changed, same Holm): {len(shifted)} contrasts")

    return comparison_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_analysis(
    rows: dict[str, dict],
    b2_recall: dict[str, float] | None,
    scatter_path: str | None,
    label: str = "",
) -> tuple[list[dict], list[dict], dict]:
    """Run full pairwise pass + report sections. Returns (pw, summary, circularity)."""
    labels = {qid: r["group"] for qid, r in rows.items()}
    n_easy  = sum(1 for v in labels.values() if v == "easy")
    n_hard  = sum(1 for v in labels.values() if v == "hard")
    n_unans = sum(1 for v in labels.values() if v == "unanswerable")

    prefix = f" [{label}]" if label else ""
    print(f"\nGroups{prefix}: easy n={n_easy}, hard n={n_hard}, unanswerable n={n_unans}")
    print(f"Features: {len(ALL_FEATURES)} total "
          f"({len(FEATURES_TOPOLOGY)} topology, {len(FEATURE_IP)} IP, {len(FEATURES_GC)} GC)")

    pw = _pairwise_pass(rows)
    _report_582(pw, rows)
    circularity = _report_583(pw, rows, b2_recall)
    _report_584(pw, rows)
    summary = _summary_table(pw)

    if scatter_path and circularity.get("pairs") and b2_recall:
        _plot_scatter(
            circularity["pairs"], rows, b2_recall,
            circularity.get("rho", float("nan")),
            circularity.get("p", float("nan")),
            scatter_path,
        )
    return pw, summary, circularity


def _build_result_json(
    rows: dict[str, dict],
    pw: list[dict],
    summary: list[dict],
    circularity: dict,
) -> dict:
    def _clean(v: Any) -> Any:
        return None if isinstance(v, float) and math.isnan(v) else v

    labels = {qid: r["group"] for qid, r in rows.items()}
    return {
        "groups": {
            "easy":         sum(1 for v in labels.values() if v == "easy"),
            "hard":         sum(1 for v in labels.values() if v == "hard"),
            "unanswerable": sum(1 for v in labels.values() if v == "unanswerable"),
        },
        "features": ALL_FEATURES,
        "pairwise": [{k: _clean(v) for k, v in r.items()} for r in pw],
        "summary":  [{k: _clean(v) for k, v in r.items()} for r in summary],
        "circularity": {
            "n":   circularity.get("n"),
            "rho": _clean(circularity.get("rho", float("nan"))),
            "p":   _clean(circularity.get("p",   float("nan"))),
        } if circularity else None,
        "descriptive": {
            feat: {
                grp: _descriptive(_group_vals(rows, grp, feat))
                for grp in ("easy", "hard", "unanswerable")
            }
            for feat in ALL_FEATURES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Series C unified analysis (5.8.2–5.8.4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── JSON-file mode ───────────────────────────────────────────────────────
    parser.add_argument("--topology",       default=None,
                        help="Pre-computed topology features (answerable).")
    parser.add_argument("--topology-unans", default=None)
    parser.add_argument("--gc",             default=None)
    parser.add_argument("--gc-unans",       default=None)
    parser.add_argument("--ip",             default=None)
    parser.add_argument("--ip-unans",       default=None)
    # ── Run-dir mode ─────────────────────────────────────────────────────────
    parser.add_argument("--run-dir",        default=None,
                        help="Canonical KG run directory (computes features inline).")
    parser.add_argument("--raw-run-dir",    default=None,
                        help="Raw (non-canonicalized) KG run directory for comparison.")
    parser.add_argument("--benchmarks-unans", default="tests/benchmarks_unanswerable.yaml",
                        help="Unanswerable benchmark file (run-dir mode).")
    # ── Shared ───────────────────────────────────────────────────────────────
    parser.add_argument("--benchmarks",     default="tests/benchmarks.yaml")
    parser.add_argument("--b2",             default="results_b2.json",
                        help="Series B FAISS baseline for circularity check (dense_R@10).")
    parser.add_argument("--output",         default=None,
                        help="JSON output for canonical / single-run results.")
    parser.add_argument("--raw-output",     default=None,
                        help="JSON output for raw-graph results (comparison mode).")
    parser.add_argument("--comparison-output", default=None,
                        help="JSON output for the comparison table.")
    parser.add_argument("--scatter",        default=None,
                        help="PNG path for circularity scatter (canonical graph).")
    parser.add_argument("-v", "--verbose",  action="store_true")
    args = parser.parse_args()

    # ── Load benchmark labels (shared by both modes) ─────────────────────────
    labels = _load_benchmark_labels(args.benchmarks)

    # ── Load B2 FAISS recall ──────────────────────────────────────────────────
    b2_recall: dict[str, float] | None = None
    if args.b2:
        try:
            b2_rows = _load_json_per_query(args.b2)
            b2_recall = {r["id"]: float(r["dense_R@10"]) for r in b2_rows
                         if r.get("dense_R@10") is not None}
            print(f"B2 FAISS baseline: {len(b2_recall)} queries with dense_R@10.")
        except Exception as e:
            print(f"Warning: could not load B2 ({e}); circularity check skipped.")

    # ── Build feature rows ────────────────────────────────────────────────────
    # Priority: run-dir mode > JSON mode (with defaults as fallback).
    use_run_dir = bool(args.run_dir or args.raw_run_dir)

    if use_run_dir and args.run_dir:
        print("\n[Canonical graph]")
        rows_can = _rows_from_run_dir(
            args.run_dir, args.benchmarks, args.benchmarks_unans,
            labels, verbose=args.verbose,
        )
    elif not use_run_dir:
        # JSON mode — fall back to defaults when individual args are None
        topo_path       = args.topology       or "results_c.json"
        topo_unans_path = args.topology_unans or "results_c_unanswerable.json"
        gc_path         = args.gc             or "results_c_global_centrality.json"
        gc_unans_path   = args.gc_unans       or "results_c_global_centrality_unanswerable.json"
        ip_path         = args.ip             or "results_c_interconcept_distance.json"
        ip_unans_path   = args.ip_unans       or "results_c_interconcept_distance_unanswerable.json"

        rows_can = _build_feature_rows(
            topology_ans=_load_json_per_query(topo_path),
            topology_unans=_load_json_per_query(topo_unans_path),
            gc_ans=_load_json_per_query(gc_path),
            gc_unans=_load_json_per_query(gc_unans_path),
            ip_ans=_load_json_per_query(ip_path),
            ip_unans=_load_json_per_query(ip_unans_path),
            labels=labels,
        )
    else:
        # raw-run-dir given without run-dir: treat raw as the single graph
        rows_can = None  # type: ignore[assignment]

    # ── Run canonical analysis ────────────────────────────────────────────────
    pw_can = summary_can = circularity_can = None
    if rows_can is not None:
        lbl = "canonical" if args.raw_run_dir else ""
        pw_can, summary_can, circularity_can = _run_analysis(
            rows_can, b2_recall, args.scatter, label=lbl
        )
        if args.output:
            out_path = _resolve(args.output)
            with open(out_path, "w") as f:
                json.dump(_build_result_json(rows_can, pw_can, summary_can, circularity_can),
                          f, indent=2)
            print(f"\nCanonical results written to {out_path}")

    # ── Run raw analysis (comparison mode) ───────────────────────────────────
    if args.raw_run_dir:
        print("\n[Raw graph]")
        rows_raw = _rows_from_run_dir(
            args.raw_run_dir, args.benchmarks, args.benchmarks_unans,
            labels, verbose=args.verbose,
        )
        pw_raw, summary_raw, circularity_raw = _run_analysis(
            rows_raw, b2_recall, scatter_path=None, label="raw"
        )
        if args.raw_output:
            out_path = _resolve(args.raw_output)
            with open(out_path, "w") as f:
                json.dump(_build_result_json(rows_raw, pw_raw, summary_raw, circularity_raw),
                          f, indent=2)
            print(f"\nRaw results written to {out_path}")

        # Comparison (requires canonical pw to exist)
        if pw_can is not None:
            comparison = _print_comparison(pw_can, pw_raw)
            if args.comparison_output:
                out_path = _resolve(args.comparison_output)
                def _clean(v: Any) -> Any:
                    return None if isinstance(v, float) and math.isnan(v) else v
                with open(out_path, "w") as f:
                    json.dump(
                        [{k: _clean(v) for k, v in r.items()} for r in comparison],
                        f, indent=2,
                    )
                print(f"Comparison written to {out_path}")
        elif rows_can is None:
            # Only raw-run-dir supplied: run as standalone
            pw_raw, summary_raw, circularity_raw = _run_analysis(
                rows_raw, b2_recall, scatter_path=args.scatter, label="raw"
            )
            if args.output:
                out_path = _resolve(args.output)
                with open(out_path, "w") as f:
                    json.dump(_build_result_json(rows_raw, pw_raw, summary_raw, circularity_raw),
                              f, indent=2)
                print(f"\nRaw results written to {out_path}")


if __name__ == "__main__":
    main()
