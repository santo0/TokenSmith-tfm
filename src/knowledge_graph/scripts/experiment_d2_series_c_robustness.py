"""D2 — Series C robustness: does mean_distance hold up on the raw (uncanonicalized) graph?

Recomputes the two headline Series C features -- mean_distance (IP1) and
n_query_nodes -- on both the canonical and raw graphs and compares their
ability to separate easy from hard queries.

Ground truth: the 'difficulty' field in benchmarks.yaml (human-assigned,
graph-agnostic). This sidesteps the circularity of using LLM judge scores
derived from canonical-system retrieval as labels.

Outputs:
  - Side-by-side group means (easy / hard) for both graphs
  - Cohen's d (easy vs hard) for mean_distance on both graphs
  - LOOCV AUC for mean_distance classifier on both graphs
  - Spearman rho of mean_distance vs binary difficulty label

A narrow result gap means the difficulty signal is robust to fragmentation
(strengthens RQ1 confidence). A large gap means canonicalization is a
precondition for the signal (strengthens the canonicalization argument,
with the provenance caveat noted in the write-up).

Usage:
    # Step 1: run the IP experiment on the raw graph first:
    #   python -m src.knowledge_graph.scripts.experiment_c_interconcept_distance \\
    #       --run-dir data/knowledge_graph/runs/<raw-timestamp> \\
    #       --no-llm \\
    #       --output results_c_ip_raw.json

    # Step 2: compare
    python -m src.knowledge_graph.scripts.experiment_d2_series_c_robustness \\
        --canonical-ip results_c_interconcept_distance.json \\
        --raw-ip results_c_ip_raw.json \\
        --benchmarks tests/benchmarks.yaml \\
        --output results_d2.json
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _cohen_d(a: list[float], b: list[float]) -> float:
    """Cohen's d: (mean_b - mean_a) / pooled_std. Returns nan if n < 2."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    mean_a = sum(a) / na
    mean_b = sum(b) / nb
    var_a = sum((x - mean_a) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1)
    pooled_std = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std == 0:
        return float("nan")
    return (mean_b - mean_a) / pooled_std


def _spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    try:
        from scipy.stats import spearmanr
        if len(xs) < 3:
            return float("nan"), float("nan")
        r = spearmanr(xs, ys)
        return float(r.statistic), float(r.pvalue)
    except ImportError:
        return float("nan"), float("nan")


def _kruskal(groups: list[list[float]]) -> tuple[float, float]:
    try:
        from scipy.stats import kruskal
        non_empty = [g for g in groups if len(g) >= 1]
        if len(non_empty) < 2:
            return float("nan"), float("nan")
        r = kruskal(*non_empty)
        return float(r.statistic), float(r.pvalue)
    except ImportError:
        return float("nan"), float("nan")


def _loocv_auc(X: list[list[float]], y: list[int]) -> float:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        Xarr = np.array(X, dtype=float)
        yarr = np.array(y, dtype=int)
        if len(set(yarr)) < 2 or len(yarr) < 3:
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

        return float(roc_auc_score(yarr, probs))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def _analyse(
    rows: list[dict],
    id_to_label: dict[str, int],
    feature: str,
) -> dict:
    """Return statistics for one feature on one graph's per-query rows."""
    easy, hard = [], []
    skipped_no_label = 0
    skipped_none_val = 0

    for r in rows:
        label = id_to_label.get(r["id"])
        if label is None:
            skipped_no_label += 1
            continue
        val = r.get(feature)
        if val is None:
            skipped_none_val += 1
            continue
        (easy if label == 0 else hard).append(float(val))

    n_easy, n_hard = len(easy), len(hard)
    mean_easy = sum(easy) / n_easy if n_easy else float("nan")
    mean_hard = sum(hard) / n_hard if n_hard else float("nan")
    d = _cohen_d(easy, hard)

    kw_h, kw_p = _kruskal([easy, hard])

    all_vals = easy + hard
    all_labels = [0] * len(easy) + [1] * len(hard)
    rho, rho_p = _spearman(all_vals, [float(l) for l in all_labels])
    auc = _loocv_auc([[v] for v in all_vals], all_labels)

    return {
        "n_easy": n_easy,
        "n_hard": n_hard,
        "mean_easy": round(mean_easy, 4) if not math.isnan(mean_easy) else None,
        "mean_hard": round(mean_hard, 4) if not math.isnan(mean_hard) else None,
        "cohen_d": round(d, 4) if not math.isnan(d) else None,
        "kruskal_H": round(kw_h, 4) if not math.isnan(kw_h) else None,
        "kruskal_p": round(kw_p, 4) if not math.isnan(kw_p) else None,
        "spearman_rho": round(rho, 4) if not math.isnan(rho) else None,
        "spearman_p": round(rho_p, 4) if not math.isnan(rho_p) else None,
        "loocv_auc": round(auc, 4) if not math.isnan(auc) else None,
        "skipped_no_label": skipped_no_label,
        "skipped_none_val": skipped_none_val,
    }


def _print_side_by_side(feature: str, canon: dict, raw: dict) -> None:
    print(f"\n  Feature: {feature}")
    print(f"  {'Metric':<22} {'Canonical':>12} {'Raw':>12}")
    print(f"  {'-'*22} {'-'*12} {'-'*12}")

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    rows = [
        ("n_easy",       canon["n_easy"],       raw["n_easy"]),
        ("n_hard",       canon["n_hard"],       raw["n_hard"]),
        ("mean_easy",    canon["mean_easy"],    raw["mean_easy"]),
        ("mean_hard",    canon["mean_hard"],    raw["mean_hard"]),
        ("cohen_d",      canon["cohen_d"],      raw["cohen_d"]),
        ("kruskal_H",    canon["kruskal_H"],    raw["kruskal_H"]),
        ("kruskal_p",    canon["kruskal_p"],    raw["kruskal_p"]),
        ("spearman_rho", canon["spearman_rho"], raw["spearman_rho"]),
        ("spearman_p",   canon["spearman_p"],   raw["spearman_p"]),
        ("loocv_auc",    canon["loocv_auc"],    raw["loocv_auc"]),
    ]
    for label, c_val, r_val in rows:
        c_str = str(c_val) if isinstance(c_val, int) else _fmt(c_val)
        r_str = str(r_val) if isinstance(r_val, int) else _fmt(r_val)
        print(f"  {label:<22} {c_str:>12} {r_str:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D2: Series C robustness — mean_distance on raw vs canonical graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--canonical-ip",
        default="results_c_interconcept_distance.json",
        help="IP results JSON from the canonicalized graph",
    )
    parser.add_argument(
        "--raw-ip",
        required=True,
        help="IP results JSON from the raw (no-canon) graph",
    )
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument("--output", default="results_d2.json")
    args = parser.parse_args()

    import yaml

    root = Path(__file__).parent.parent.parent.parent

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else root / path

    # Load inputs
    with open(_resolve(args.benchmarks)) as f:
        bm_data = yaml.safe_load(f)
    benchmarks = bm_data["benchmarks"]

    with open(_resolve(args.canonical_ip)) as f:
        canon_data = json.load(f)
    with open(_resolve(args.raw_ip)) as f:
        raw_data = json.load(f)

    canon_rows: list[dict] = canon_data["per_query"]
    raw_rows: list[dict] = raw_data["per_query"]

    # Build label map from YAML (human-assigned, graph-agnostic)
    # easy=0, hard=1; unknown excluded
    id_to_label: dict[str, int] = {}
    for bm in benchmarks:
        d = bm.get("difficulty")
        if d == "easy":
            id_to_label[bm["id"]] = 0
        elif d == "hard":
            id_to_label[bm["id"]] = 1
        # unknown → excluded

    n_easy_labels = sum(1 for v in id_to_label.values() if v == 0)
    n_hard_labels = sum(1 for v in id_to_label.values() if v == 1)

    # Build per-query node-count table (shows how many seeds each graph matched)
    canon_id_map = {r["id"]: r for r in canon_rows}
    raw_id_map = {r["id"]: r for r in raw_rows}

    SEP = "=" * 64
    print(f"\n{SEP}")
    print("D2 — Series C Robustness: mean_distance on raw vs canonical graph")
    print(SEP)
    print(f"\nLabels from benchmarks.yaml (human-assigned, graph-agnostic):")
    print(f"  easy: {n_easy_labels}   hard: {n_hard_labels}   "
          f"excluded (unknown): {len(benchmarks) - n_easy_labels - n_hard_labels}")

    # ── Query node coverage comparison ────────────────────────────────────────
    print(f"\nQuery node coverage (n matched nodes per benchmark):")
    print(f"  {'id':<35} {'yaml_diff':>9} {'canon_nodes':>11} {'raw_nodes':>9} {'delta':>6}")
    print(f"  {'-'*35} {'-'*9} {'-'*11} {'-'*9} {'-'*6}")
    id_to_diff_str = {bm["id"]: bm.get("difficulty", "unknown") for bm in benchmarks}
    all_ids = sorted(set(canon_id_map) | set(raw_id_map))
    coverage_drops = 0
    for bid in all_ids:
        c_nodes = canon_id_map[bid]["n_query_nodes"] if bid in canon_id_map else "—"
        r_nodes = raw_id_map[bid]["n_query_nodes"] if bid in raw_id_map else "—"
        diff_str = id_to_diff_str.get(bid, "unknown")
        if isinstance(c_nodes, int) and isinstance(r_nodes, int):
            delta = r_nodes - c_nodes
            delta_str = f"{delta:+d}"
            if delta < 0:
                coverage_drops += 1
        else:
            delta_str = "N/A"
        print(f"  {bid:<35} {diff_str:>9} {str(c_nodes):>11} {str(r_nodes):>9} {delta_str:>6}")
    print(f"\n  Benchmarks where raw matched fewer nodes: {coverage_drops}")

    # ── Feature comparison ────────────────────────────────────────────────────
    FEATURES = ["mean_distance", "max_distance", "n_query_nodes"]
    canon_stats: dict[str, dict] = {}
    raw_stats: dict[str, dict] = {}
    for feat in FEATURES:
        canon_stats[feat] = _analyse(canon_rows, id_to_label, feat)
        raw_stats[feat] = _analyse(raw_rows, id_to_label, feat)

    print(f"\n{SEP}")
    print("Feature separation (easy vs hard, YAML labels)")
    print(SEP)
    for feat in FEATURES:
        _print_side_by_side(feat, canon_stats[feat], raw_stats[feat])

    # ── Interpretation note ───────────────────────────────────────────────────
    c_d = canon_stats["mean_distance"]["cohen_d"]
    r_d = raw_stats["mean_distance"]["cohen_d"]
    print(f"\n{SEP}")
    print("Interpretation")
    print(SEP)
    if c_d is not None and r_d is not None:
        delta_d = c_d - r_d
        if abs(delta_d) < 0.2:
            print(f"  mean_distance cohen_d: canonical={c_d:.3f}  raw={r_d:.3f}  delta={delta_d:+.3f}")
            print("  → Separation is robust to canonicalization. The difficulty signal")
            print("    does not depend on the merge step. (Report as robustness finding.)")
        elif c_d > r_d:
            print(f"  mean_distance cohen_d: canonical={c_d:.3f}  raw={r_d:.3f}  delta={delta_d:+.3f}")
            print("  → Canonical graph separates easy/hard more clearly than raw.")
            print("    Fragmentation blurs the distance signal. Note the label-provenance")
            print("    caveat in the write-up (labels from canonical baseline retrieval).")
        else:
            print(f"  mean_distance cohen_d: canonical={c_d:.3f}  raw={r_d:.3f}  delta={delta_d:+.3f}")
            print("  → Raw graph separates easy/hard more clearly. Unexpected; investigate")
            print("    whether coverage drops in canonical are driving the result.")
    print(SEP)

    # ── Save ─────────────────────────────────────────────────────────────────
    result = {
        "label_source": "benchmarks.yaml difficulty field (human-assigned)",
        "n_easy_labels": n_easy_labels,
        "n_hard_labels": n_hard_labels,
        "features": FEATURES,
        "canonical": canon_stats,
        "raw": raw_stats,
        "coverage": {
            bid: {
                "yaml_difficulty": id_to_diff_str.get(bid),
                "canon_nodes": canon_id_map[bid]["n_query_nodes"] if bid in canon_id_map else None,
                "raw_nodes": raw_id_map[bid]["n_query_nodes"] if bid in raw_id_map else None,
            }
            for bid in all_ids
        },
    }
    out = _resolve(args.output)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
