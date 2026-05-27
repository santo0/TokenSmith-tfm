"""experiment_c_comparison.py

Compare KG topology features across three query classes: easy, hard, unanswerable.

Analyses performed
------------------
  1. Descriptive statistics per feature per group (mean ± SD, median, IQR).
  2. Kruskal-Wallis H test per feature — overall non-parametric one-way test.
  3. Pairwise Mann-Whitney U with Holm correction — three pairs:
       easy vs hard, easy vs unanswerable, hard vs unanswerable.
     Rank-biserial r as effect size for each pair.
  4. Cohen's d for each pairwise comparison (parametric reference).
  5. Logistic regression with LOOCV:
       - Binary:      answerable (easy+hard=1) vs unanswerable (0) — AUC.
       - Multinomial: easy / hard / unanswerable — macro OvR AUC.
  6. If judge_score is present, the same pairwise tests are applied to it.

Usage
-----
    python -m src.knowledge_graph.scripts.experiment_c_comparison \\
        --answerable   results_c_chp.json \\
        --unanswerable results_c_chp_unanswerable.json \\
        --benchmarks   tests/benchmarks.yaml \\
        --output       results_c_comparison.json

    # also write an HTML report:
    python -m src.knowledge_graph.scripts.experiment_c_comparison \\
        --answerable   results_c_chp.json \\
        --unanswerable results_c_chp_unanswerable.json \\
        --benchmarks   tests/benchmarks.yaml \\
        --output       results_c_comparison.json \\
        --html         results_c_comparison.html
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _mannwhitney(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Return (U_stat, p_value, rank_biserial_r). Two-sided alternative."""
    from scipy.stats import mannwhitneyu
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan"), float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = mannwhitneyu(a, b, alternative="two-sided")
    U = float(result.statistic)
    p = float(result.pvalue)
    r = 2 * U / (n_a * n_b) - 1
    return U, p, r


def _kruskal_wallis(*groups: list[float]) -> tuple[float, float]:
    """Return (H_stat, p_value) for k ≥ 2 groups."""
    from scipy.stats import kruskal
    valid = [g for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H, p = kruskal(*valid)
    return float(H), float(p)


def _holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction. Returns corrected p-values in original order."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    corrected = [0.0] * n
    prev = 0.0
    for rank, i in enumerate(order):
        c = min(1.0, p_values[i] * (n - rank))
        corrected[i] = max(c, prev)
        prev = corrected[i]
    return corrected


def _cohens_d(a: list[float], b: list[float]) -> float:
    import numpy as np
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    pooled_sd = math.sqrt(
        ((na - 1) * float(np.var(a, ddof=1)) + (nb - 1) * float(np.var(b, ddof=1)))
        / (na + nb - 2)
    )
    if pooled_sd == 0:
        return float("nan")
    return (float(np.mean(a)) - float(np.mean(b))) / pooled_sd


def _loocv_auc_binary(X: list[list[float]], y: list[int]) -> float:
    """Binary LOOCV AUC (answerable=1, unanswerable=0)."""
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
            X_tr, X_te = Xarr[train_idx], Xarr[test_idx]
            y_tr = yarr[train_idx]
            if len(set(y_tr)) < 2:
                probs.append(0.5)
                continue
            Xs_tr = scaler.fit_transform(X_tr)
            Xs_te = scaler.transform(X_te)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(max_iter=2000)
                clf.fit(Xs_tr, y_tr)
            prob = clf.predict_proba(Xs_te)[0]
            pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
            probs.append(prob[pos_idx])

        return float(roc_auc_score(yarr, probs))
    except Exception:
        return float("nan")


def _loocv_auc_multiclass(X: list[list[float]], y: list[int]) -> float:
    """Multinomial LOOCV AUC — macro OvR (easy=2, hard=1, unanswerable=0)."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler, label_binarize

        Xarr = np.array(X, dtype=float)
        yarr = np.array(y, dtype=int)
        classes = sorted(set(yarr))
        if len(classes) < 2:
            return float("nan")

        scaler = StandardScaler()
        loo = LeaveOneOut()
        all_probs: list[list[float]] = []

        for train_idx, test_idx in loo.split(Xarr):
            X_tr, X_te = Xarr[train_idx], Xarr[test_idx]
            y_tr = yarr[train_idx]
            if len(set(y_tr)) < 2:
                all_probs.append([1 / len(classes)] * len(classes))
                continue
            Xs_tr = scaler.fit_transform(X_tr)
            Xs_te = scaler.transform(X_te)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(multi_class="multinomial",
                                         solver="lbfgs", max_iter=2000)
                clf.fit(Xs_tr, y_tr)
            raw_prob = clf.predict_proba(Xs_te)[0]
            aligned = []
            for c in classes:
                if c in clf.classes_:
                    aligned.append(raw_prob[list(clf.classes_).index(c)])
                else:
                    aligned.append(0.0)
            all_probs.append(aligned)

        Y_bin = label_binarize(yarr, classes=classes)
        if Y_bin.shape[1] == 1:
            return float("nan")
        return float(roc_auc_score(
            Y_bin, np.array(all_probs), multi_class="ovr", average="macro"
        ))
    except Exception:
        return float("nan")


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
        "q1":     float(np.percentile(arr, 25)),
        "q3":     float(np.percentile(arr, 75)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
    }


def _effect_label(r: float) -> str:
    ar = abs(r)
    if math.isnan(ar):
        return "n/a"
    if ar < 0.1:
        return "negligible"
    if ar < 0.3:
        return "small"
    if ar < 0.5:
        return "medium"
    return "large"


def _sig_stars(p: float) -> str:
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


def _ascii_bar(r: float, width: int = 20) -> str:
    if math.isnan(r):
        return " " * width
    filled = round(abs(r) * width)
    return ("█" * filled).ljust(width)


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

TOPOLOGY_FEATURES = [
    "diameter",
    "density",
    "coverage_delta",
    "community_dispersion",
    "community_span",
    "mean_betweenness",
]

PAIRS = [
    ("easy", "hard"),
    ("easy", "unanswerable"),
    ("hard", "unanswerable"),
]


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.is_absolute():
        root = Path(__file__).parent.parent.parent.parent
        p = root / path
    with open(p) as f:
        data = json.load(f)
    return data["per_query"]


def _load_difficulties(benchmarks_path: str) -> dict[str, str]:
    """Return {query_id: 'easy'|'hard'} from benchmarks.yaml difficulty field."""
    import yaml
    p = Path(benchmarks_path)
    if not p.is_absolute():
        root = Path(__file__).parent.parent.parent.parent
        p = root / benchmarks_path
    with open(p) as f:
        data = yaml.safe_load(f)
    result = {}
    for entry in data.get("benchmarks", []):
        diff = entry.get("difficulty")
        if diff in ("easy", "hard"):
            result[entry["id"]] = diff
    return result


def _split_by_difficulty(
    rows: list[dict], difficulties: dict[str, str]
) -> tuple[list[dict], list[dict]]:
    """Split answerable rows into (easy_rows, hard_rows)."""
    easy, hard = [], []
    for row in rows:
        qid = row.get("id") or row.get("query_id") or row.get("query")
        diff = difficulties.get(qid)
        if diff == "easy":
            easy.append(row)
        elif diff == "hard":
            hard.append(row)
    return easy, hard


def _extract(rows: list[dict], field: str) -> list[float]:
    return [float(r[field]) for r in rows if r.get(field) is not None]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v: Any, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{decimals}f}"


def _print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# Print sections
# ---------------------------------------------------------------------------

def _print_descriptive(groups: dict[str, list[dict]], features: list[str]) -> None:
    _print_section("1. Descriptive statistics")
    header = (f"{'Feature':<26} {'Group':<14} {'n':>3}  {'mean':>7}  "
              f"{'±sd':>7}  {'median':>7}  {'IQR':>13}  {'[min, max]':>16}")
    print(header)
    print("-" * len(header))
    for feat in features:
        for label, rows in groups.items():
            vals = _extract(rows, feat)
            d = _descriptive(vals)
            if not d:
                print(f"  {feat:<24} {label:<14} {'0':>3}")
                continue
            iqr = f"[{_fmt(d['q1'])}, {_fmt(d['q3'])}]"
            minmax = f"[{_fmt(d['min'])}, {_fmt(d['max'])}]"
            print(f"  {feat:<24} {label:<14} {d['n']:>3}  {_fmt(d['mean']):>7}  "
                  f"{_fmt(d['sd']):>7}  {_fmt(d['median']):>7}  "
                  f"{iqr:>13}  {minmax:>16}")
        print()


def _print_kruskal(kw_results: list[dict]) -> None:
    _print_section("2. Kruskal-Wallis H test (overall, 3 groups)")
    print("  Null hypothesis: all three groups share the same distribution.\n")
    header = f"  {'Feature':<26} {'H':>8}  {'p':>8}  {'sig':>4}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in kw_results:
        print(f"  {row['feature']:<26} {_fmt(row['H_stat'], 2):>8}  "
              f"{_fmt(row['p_value'], 4):>8}  {_sig_stars(row['p_value']):>4}")


def _print_pairwise(pw_results: list[dict]) -> None:
    _print_section("3. Pairwise Mann-Whitney U + Holm correction")
    print("  Significance uses Holm-corrected p per feature (3 comparisons each).")
    print("  *** p<0.001  ** p<0.01  * p<0.05  . p<0.10  ns\n")
    header = (f"  {'Feature':<26} {'Pair':<28} {'U':>8}  "
              f"{'p_raw':>8}  {'p_holm':>8}  {'sig':>4}  "
              f"{'r (rb)':>8}  {'|r|':>20}  {'effect':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    last_feat = None
    for row in pw_results:
        feat = row["feature"] if row["feature"] != last_feat else ""
        last_feat = row["feature"]
        pair_label = f"{row['group_a']} vs {row['group_b']}"
        bar = _ascii_bar(row["rank_biserial_r"])
        print(f"  {feat:<26} {pair_label:<28} {_fmt(row['U_stat'], 1):>8}  "
              f"{_fmt(row['p_raw'], 4):>8}  {_fmt(row['p_holm'], 4):>8}  "
              f"{_sig_stars(row['p_holm']):>4}  "
              f"{_fmt(row['rank_biserial_r']):>8}  {bar:>20}  "
              f"{_effect_label(row['rank_biserial_r']):>10}")
    print()


def _print_loocv(loocv_results: dict[str, float]) -> None:
    _print_section("4. Logistic regression — LOOCV AUC")
    print("  Binary:      answerable (easy+hard=1) vs unanswerable (0).")
    print("  Multinomial: easy / hard / unanswerable — macro OvR AUC.\n")
    for name, auc in loocv_results.items():
        bar = _ascii_bar(max(0.0, auc - 0.5) * 2 if not math.isnan(auc) else float("nan"))
        print(f"  {name:<36} AUC = {_fmt(auc)}  {bar}")


def _print_judge(groups: dict[str, list[float]]) -> None:
    if not any(groups.values()):
        return
    _print_section("5. LLM judge score comparison (judge_score)")
    print("  Answerable groups: high score = relevant content retrieved.")
    print("  Unanswerable: high score = plausible hallucination.\n")
    for label, vals in groups.items():
        d = _descriptive(vals)
        if d:
            print(f"  {label:<14} n={d['n']}  mean={_fmt(d['mean'])}  "
                  f"median={_fmt(d['median'])}  sd={_fmt(d['sd'])}")
    print()
    pairs_vals = [(a, b) for a, b in PAIRS if groups.get(a) and groups.get(b)]
    for ga, gb in pairs_vals:
        U, p, r = _mannwhitney(groups[ga], groups[gb])
        print(f"  {ga} vs {gb:<16} "
              f"U={_fmt(U,1)}  p={_fmt(p,4)} {_sig_stars(p)}  "
              f"r={_fmt(r)}  effect={_effect_label(r)}")


# ---------------------------------------------------------------------------
# Matplotlib / seaborn plots
# ---------------------------------------------------------------------------

_COLOURS = {
    "easy":         "#27ae60",
    "hard":         "#e67e22",
    "unanswerable": "#c0392b",
}
_PLOT_FEATURES = ["density", "coverage_delta", "community_span"]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    import pandas as pd
    _HAS_PLOT = True
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "figure.dpi":         150,
    })
except ImportError:
    _HAS_PLOT = False


def _plot_boxplot_grid(groups: dict[str, list[dict]],
                       pw_results: list[dict], out_path: str) -> None:
    """Box plots for 3 top features, all three groups side-by-side."""
    if not _HAS_PLOT:
        return

    # p_holm per (feature, pair) for annotation
    p_by = {(r["feature"], r["group_a"], r["group_b"]): r["p_holm"]
            for r in pw_results}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Feature distributions: easy / hard / unanswerable",
                 fontsize=13, fontweight="bold", y=1.02)

    group_order = ["easy", "hard", "unanswerable"]
    for ax, feat in zip(axes, _PLOT_FEATURES):
        records = []
        for label in group_order:
            for v in _extract(groups[label], feat):
                records.append({"value": v, "group": label.capitalize()})
        df = pd.DataFrame(records)
        palette = {k.capitalize(): v for k, v in _COLOURS.items()}
        sns.boxplot(
            data=df, x="group", y="value", hue="group",
            palette=palette, order=[g.capitalize() for g in group_order],
            width=0.5, linewidth=1.2, fliersize=4, ax=ax, legend=False,
        )
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(feat, fontsize=9)

        # annotate smallest Holm-corrected p involving this feature
        relevant = [r for r in pw_results if r["feature"] == feat]
        if relevant:
            best = min(relevant, key=lambda r: r["p_holm"])
            p = best["p_holm"]
            stars = _sig_stars(p)
            colour = "#c0392b" if stars not in ("ns", "") else "#555"
            ax.annotate(
                f"best pair p={p:.4f} {stars}",
                xy=(0.5, 1.01), xycoords="axes fraction",
                ha="center", fontsize=8.5, color=colour,
            )

    legend_patches = [mpatches.Patch(color=c, label=k.capitalize())
                      for k, c in _COLOURS.items()]
    fig.legend(handles=legend_patches, loc="upper right",
               fontsize=8.5, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pairwise_effects(pw_results: list[dict], out_path: str) -> None:
    """Grouped horizontal bar chart of rank-biserial r for all pairs × features."""
    if not _HAS_PLOT:
        return

    pair_labels = [f"{a} vs {b}" for a, b in PAIRS]
    pair_colours = ["#4a90d9", "#8e44ad", "#c0392b"]

    features = TOPOLOGY_FEATURES
    y_labels = []
    y_vals = []
    y_colours = []

    # interleave: for each feature, one bar per pair (with gaps between features)
    spacing = 0.35
    tick_positions = []
    tick_labels_list = []
    y = 0.0
    for feat in features:
        for pi, (ga, gb) in enumerate(PAIRS):
            row = next((r for r in pw_results
                        if r["feature"] == feat
                        and r["group_a"] == ga and r["group_b"] == gb), None)
            rv = row["rank_biserial_r"] if row and not math.isnan(row["rank_biserial_r"]) else 0.0
            col = pair_colours[pi]
            if row and (not math.isnan(row["p_holm"])) and row["p_holm"] >= 0.10:
                col = "#aaaaaa"
            y_labels.append(f"{feat}  [{ga[:4]} vs {gb[:4]}]")
            y_vals.append(rv)
            y_colours.append(col)
            tick_positions.append(y)
            tick_labels_list.append(f"{feat}  [{ga[:4]} vs {gb[:4]}]")
            y += spacing
        y += spacing * 0.5  # gap between features

    fig, ax = plt.subplots(figsize=(10, max(6, len(y_vals) * 0.38)))
    ax.barh(tick_positions, y_vals, color=y_colours, height=spacing * 0.75, zorder=3)

    for pos, rv, row_colour in zip(tick_positions, y_vals, y_colours):
        label = f"  {rv:+.3f}" if rv >= 0 else f"{rv:+.3f}  "
        ha = "left" if rv >= 0 else "right"
        ax.text(rv, pos, label, ha=ha, va="center", fontsize=7.5)

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels_list, fontsize=8)
    ax.axvline(0,    color="black", lw=1.0, zorder=4)
    for x in (0.3, -0.3, 0.5, -0.5):
        ax.axvline(x, color="#aaa", lw=0.7, linestyle="--", alpha=0.7, zorder=2)
    ax.set_xlim(-1.15, 1.15)
    ax.set_xlabel("Rank-biserial r (effect size)", fontsize=10)
    ax.set_title("Pairwise effect sizes — Holm-corrected (grey = ns)",
                 fontsize=11, fontweight="bold")

    legend_patches = [mpatches.Patch(color=c, label=l)
                      for c, l in zip(pair_colours, pair_labels)]
    legend_patches.append(mpatches.Patch(color="#aaaaaa", label="ns (p_holm ≥ 0.10)"))
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loocv_auc(loocv_results: dict, out_path: str) -> None:
    """Horizontal bar chart of LOOCV AUC per feature / feature set."""
    if not _HAS_PLOT:
        return

    def _auc_colour(auc) -> str:
        if auc is None or math.isnan(float(auc if auc is not None else float("nan"))):
            return "#aaaaaa"
        if auc >= 0.9:
            return "#27ae60"
        if auc >= 0.7:
            return "#4a90d9"
        if auc >= 0.6:
            return "#e67e22"
        return "#aaaaaa"

    names = list(loocv_results.keys())
    aucs  = [v if v is not None else float("nan") for v in loocv_results.values()]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
    ax.barh(
        names,
        [max(0.0, a) if not math.isnan(a) else 0.0 for a in aucs],
        color=[_auc_colour(a) for a in aucs],
        height=0.55, zorder=3,
    )
    ax.axvline(0.5, color="#999", lw=1.0, linestyle="--", zorder=4, label="chance (0.5)")
    ax.axvline(0.7, color="#4a90d9", lw=1.0, linestyle="--", zorder=4, label="good (0.7)")

    for bar, auc in zip(ax.patches, aucs):
        label = f"  {auc:.3f}" if not math.isnan(auc) else "  n/a"
        ax.text(max(0.0, auc if not math.isnan(auc) else 0.0),
                bar.get_y() + bar.get_height() / 2,
                label, ha="left", va="center", fontsize=8.5)

    ax.set_xlim(0.0, 1.12)
    ax.set_xlabel("LOOCV AUC", fontsize=10)
    ax.set_title("LOOCV AUC — binary & multinomial classification",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(groups: dict[str, list[dict]], out_path: str) -> None:
    """Scatter plot density vs coverage_delta, coloured by group."""
    if not _HAS_PLOT:
        return
    markers = {"easy": "o", "hard": "s", "unanswerable": "D"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, rows in groups.items():
        xs = _extract(rows, "density")
        ys = _extract(rows, "coverage_delta")
        ax.scatter(xs, ys,
                   color=_COLOURS[label],
                   label=f"{label.capitalize()} (n={len(xs)})",
                   s=70, alpha=0.85, edgecolors="white", linewidths=0.8,
                   marker=markers[label], zorder=3)
    ax.set_xlabel("density", fontsize=11)
    ax.set_ylabel("coverage_delta", fontsize=11)
    ax.set_title("Group separation: density vs coverage_delta",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _html_table(caption: str, headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>\n"
        for row in rows
    )
    return f"<h3>{caption}</h3><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>\n"


def _build_html(groups: dict[str, list[dict]],
                kw_results: list[dict],
                pw_results: list[dict],
                loocv_results: dict[str, float],
                judge_groups: dict[str, list[float]],
                fig_paths: dict | None = None) -> str:
    css = """
<style>
  body { font-family: Arial, sans-serif; max-width: 1200px; margin: 2em auto; color: #222; }
  h2 { border-bottom: 2px solid #4a90d9; padding-bottom: .3em; }
  h3 { color: #4a90d9; margin-top: 1.5em; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 1em; font-size: .9em; }
  th { background: #4a90d9; color: #fff; padding: 6px 10px; text-align: left; }
  td { padding: 5px 10px; border-bottom: 1px solid #ddd; }
  tr:nth-child(even) { background: #f5f8ff; }
  .sig { color: #c0392b; font-weight: bold; }
  .large { color: #27ae60; font-weight: bold; }
  .medium { color: #e67e22; }
  .small { color: #7f8c8d; }
  .bar { display: inline-block; background: #4a90d9; height: 12px; vertical-align: middle; }
  .note { font-size: .85em; color: #555; font-style: italic; margin: .5em 0 1em; }
  .chart-img { max-width: 100%; margin: 0.5em 0 2em; display: block;
               border: 1px solid #e0e0e0; border-radius: 4px; }
  .chart-caption { font-size: .82em; color: #555; font-style: italic;
                   margin: -.5em 0 1.5em; }
</style>"""

    ns = {k: len(v) for k, v in groups.items()}
    body = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<title>Easy / Hard / Unanswerable — Experiment C Comparison</title>
{css}
</head>
<body>
<h2>Query Difficulty Comparison: Easy / Hard / Unanswerable</h2>
<p class="note">
  Easy n={ns.get('easy','?')} &nbsp;|&nbsp;
  Hard n={ns.get('hard','?')} &nbsp;|&nbsp;
  Unanswerable n={ns.get('unanswerable','?')} &nbsp;|&nbsp;
  Features: KG topology (diameter, density, coverage, community, betweenness)
</p>
"""

    # Section 1 — descriptive
    desc_headers = ["Feature", "Group", "n", "Mean", "±SD", "Median", "Q1–Q3", "Min", "Max"]
    desc_rows = []
    for feat in TOPOLOGY_FEATURES:
        for label, rows in groups.items():
            d = _descriptive(_extract(rows, feat))
            if not d:
                continue
            desc_rows.append([
                feat, label, str(d["n"]),
                _fmt(d["mean"]), _fmt(d["sd"]), _fmt(d["median"]),
                f"{_fmt(d['q1'])} – {_fmt(d['q3'])}",
                _fmt(d["min"]), _fmt(d["max"]),
            ])
    body += _html_table("1. Descriptive Statistics", desc_headers, desc_rows)

    # Section 2 — Kruskal-Wallis
    kw_headers = ["Feature", "H", "p-value", "Sig"]
    kw_table = []
    for row in kw_results:
        sig = _sig_stars(row["p_value"])
        sig_cls = "sig" if sig in ("*", "**", "***") else ""
        kw_table.append([
            row["feature"],
            _fmt(row["H_stat"], 2),
            f'<span class="{sig_cls}">{_fmt(row["p_value"], 4)}</span>',
            f'<span class="{sig_cls}">{sig}</span>',
        ])
    body += _html_table("2. Kruskal-Wallis H Test (overall)", kw_headers, kw_table)

    # Section 3 — pairwise
    pw_headers = ["Feature", "Pair", "U", "p (raw)", "p (Holm)", "Sig",
                  "Rank-biserial r", "Effect"]
    pw_table = []
    for row in pw_results:
        sig = _sig_stars(row["p_holm"])
        r = row["rank_biserial_r"]
        effect = _effect_label(r)
        bar_w = round(abs(r) * 120) if not math.isnan(r) else 0
        bar_html = f'<span class="bar" style="width:{bar_w}px"></span>'
        sig_cls = "sig" if sig in ("*", "**", "***") else ""
        eff_cls = effect if effect in ("large", "medium", "small") else ""
        pw_table.append([
            row["feature"],
            f"{row['group_a']} vs {row['group_b']}",
            _fmt(row["U_stat"], 1),
            _fmt(row["p_raw"], 4),
            f'<span class="{sig_cls}">{_fmt(row["p_holm"], 4)}</span>',
            f'<span class="{sig_cls}">{sig}</span>',
            f'{_fmt(r)} {bar_html}',
            f'<span class="{eff_cls}">{effect}</span>',
        ])
    body += '<p class="note">Holm correction applied per feature across 3 pairs.</p>'
    body += _html_table("3. Pairwise Mann-Whitney U + Holm Correction",
                        pw_headers, pw_table)

    if fig_paths and fig_paths.get("fig1"):
        body += f'<img src="{fig_paths["fig1"]}" class="chart-img">\n'
        body += '<p class="chart-caption">Figure 1 — Box plots for three features across all groups.</p>\n'
    if fig_paths and fig_paths.get("fig2"):
        body += f'<img src="{fig_paths["fig2"]}" class="chart-img">\n'
        body += '<p class="chart-caption">Figure 2 — Pairwise rank-biserial r. Grey bars are not significant (p_holm ≥ 0.10).</p>\n'

    # Section 4 — LOOCV
    loocv_headers = ["Feature set", "LOOCV AUC", "Power"]
    loocv_table = []
    for name, auc in loocv_results.items():
        power = "good" if auc >= 0.7 else ("moderate" if auc >= 0.6 else "weak")
        bar_w = round(max(0.0, auc - 0.5) * 240)
        bar_html = f'<span class="bar" style="width:{bar_w}px"></span>'
        loocv_table.append([name, f"{_fmt(auc)} {bar_html}", power])
    body += '<p class="note">AUC=0.5 → chance level.</p>'
    body += _html_table("4. Logistic Regression — LOOCV AUC", loocv_headers, loocv_table)

    if fig_paths and fig_paths.get("fig3"):
        body += f'<img src="{fig_paths["fig3"]}" class="chart-img">\n'
        body += '<p class="chart-caption">Figure 3 — LOOCV AUC: binary (ans vs unans) and multinomial (3-class macro).</p>\n'
    if fig_paths and fig_paths.get("fig4"):
        body += f'<img src="{fig_paths["fig4"]}" class="chart-img">\n'
        body += '<p class="chart-caption">Figure 4 — Scatter density vs coverage_delta; circles=easy, squares=hard, diamonds=unanswerable.</p>\n'

    # Section 5 — judge
    if any(judge_groups.values()):
        judge_headers = ["Group", "n", "Mean", "Median", "SD"]
        judge_table = []
        for label, vals in judge_groups.items():
            d = _descriptive(vals)
            if d:
                judge_table.append([label, str(d["n"]), _fmt(d["mean"]),
                                     _fmt(d["median"]), _fmt(d["sd"])])
        body += _html_table("5. LLM Judge Score Comparison", judge_headers, judge_table)

    body += "</body>\n</html>"
    return body


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare topology features: easy / hard / unanswerable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--answerable",   default="results_c_chp.json")
    parser.add_argument("--unanswerable", default="results_c_chp_unanswerable.json")
    parser.add_argument("--benchmarks",   default="tests/benchmarks.yaml",
                        help="benchmarks.yaml with difficulty field to split answerable rows")
    parser.add_argument("--output",       default=None)
    parser.add_argument("--html",         default=None)
    args = parser.parse_args()

    ans_rows   = _load(args.answerable)
    unans_rows = _load(args.unanswerable)

    difficulties = _load_difficulties(args.benchmarks)
    easy_rows, hard_rows = _split_by_difficulty(ans_rows, difficulties)

    groups: dict[str, list[dict]] = {
        "easy":         easy_rows,
        "hard":         hard_rows,
        "unanswerable": unans_rows,
    }
    print(f"\nLoaded {len(easy_rows)} easy, {len(hard_rows)} hard, "
          f"{len(unans_rows)} unanswerable queries.")
    if len(easy_rows) + len(hard_rows) < len(ans_rows):
        n_unlabelled = len(ans_rows) - len(easy_rows) - len(hard_rows)
        print(f"  Warning: {n_unlabelled} answerable rows had no difficulty label "
              f"and were excluded.")

    # ── 1. Descriptive ─────────────────────────────────────────────────────
    _print_descriptive(groups, TOPOLOGY_FEATURES)

    # ── 2. Kruskal-Wallis ──────────────────────────────────────────────────
    kw_results: list[dict] = []
    for feat in TOPOLOGY_FEATURES:
        vals = {k: _extract(v, feat) for k, v in groups.items()}
        H, p = _kruskal_wallis(*vals.values())
        kw_results.append({
            "feature": feat,
            "H_stat":  H,
            "p_value": p,
            "significant_p05": (not math.isnan(p)) and (p < 0.05),
        })
    _print_kruskal(kw_results)

    # ── 3. Pairwise MW-U + Holm ───────────────────────────────────────────
    pw_results: list[dict] = []
    for feat in TOPOLOGY_FEATURES:
        feat_vals = {k: _extract(v, feat) for k, v in groups.items()}
        raw_ps: list[float] = []
        pair_stats: list[dict] = []
        for ga, gb in PAIRS:
            U, p_raw, r = _mannwhitney(feat_vals[ga], feat_vals[gb])
            d = _cohens_d(feat_vals[ga], feat_vals[gb])
            raw_ps.append(p_raw if not math.isnan(p_raw) else 1.0)
            pair_stats.append({
                "feature":         feat,
                "group_a":         ga,
                "group_b":         gb,
                "n_a":             len(feat_vals[ga]),
                "n_b":             len(feat_vals[gb]),
                "U_stat":          U,
                "p_raw":           p_raw,
                "rank_biserial_r": r,
                "cohens_d":        d,
                "effect_label":    _effect_label(r),
            })
        corrected = _holm_correct(raw_ps)
        for stat, p_holm in zip(pair_stats, corrected):
            stat["p_holm"] = p_holm
            stat["significant_holm_p05"] = p_holm < 0.05
            pw_results.append(stat)
    _print_pairwise(pw_results)

    # ── 4. LOOCV ───────────────────────────────────────────────────────────
    loocv_results: dict[str, float] = {}
    all_ans_rows   = easy_rows + hard_rows
    all_rows_bin   = all_ans_rows + unans_rows
    all_labels_bin = [1] * len(all_ans_rows) + [0] * len(unans_rows)

    # Binary: individual features
    for feat in TOPOLOGY_FEATURES:
        vals = [r.get(feat) for r in all_rows_bin]
        if any(v is None for v in vals):
            loocv_results[f"binary_{feat}"] = float("nan")
            continue
        loocv_results[f"binary_{feat}"] = _loocv_auc_binary(
            [[v] for v in vals], all_labels_bin
        )

    # Binary: full topology
    full_X_bin = [[r.get(f, 0.0) or 0.0 for f in TOPOLOGY_FEATURES]
                  for r in all_rows_bin]
    loocv_results["binary_full_topology"] = _loocv_auc_binary(
        full_X_bin, all_labels_bin
    )

    # Multinomial: easy=2, hard=1, unanswerable=0
    all_rows_mc  = easy_rows + hard_rows + unans_rows
    all_labels_mc = [2] * len(easy_rows) + [1] * len(hard_rows) + [0] * len(unans_rows)

    for feat in TOPOLOGY_FEATURES:
        vals = [r.get(feat) for r in all_rows_mc]
        if any(v is None for v in vals):
            loocv_results[f"multi_{feat}"] = float("nan")
            continue
        loocv_results[f"multi_{feat}"] = _loocv_auc_multiclass(
            [[v] for v in vals], all_labels_mc
        )

    full_X_mc = [[r.get(f, 0.0) or 0.0 for f in TOPOLOGY_FEATURES]
                 for r in all_rows_mc]
    loocv_results["multi_full_topology"] = _loocv_auc_multiclass(
        full_X_mc, all_labels_mc
    )

    # Top-3 by max |r| across pairs (per feature)
    feat_max_r = {}
    for feat in TOPOLOGY_FEATURES:
        rs = [abs(r["rank_biserial_r"]) for r in pw_results
              if r["feature"] == feat and not math.isnan(r["rank_biserial_r"])]
        feat_max_r[feat] = max(rs) if rs else 0.0
    top3 = sorted(TOPOLOGY_FEATURES, key=lambda f: feat_max_r[f], reverse=True)[:3]
    loocv_results["multi_top3_by_effect"] = _loocv_auc_multiclass(
        [[r.get(f, 0.0) or 0.0 for f in top3] for r in all_rows_mc],
        all_labels_mc,
    )

    _print_loocv(loocv_results)

    # ── 5. Judge score ─────────────────────────────────────────────────────
    judge_groups = {k: _extract(v, "judge_score") for k, v in groups.items()}
    _print_judge(judge_groups)

    # ── Summary ────────────────────────────────────────────────────────────
    _print_section("Summary")
    kw_sig = [r for r in kw_results if r["significant_p05"]]
    pw_sig = [r for r in pw_results if r["significant_holm_p05"]]
    print(f"  Kruskal-Wallis significant at p<0.05:  {len(kw_sig)}/{len(kw_results)}")
    for r in kw_sig:
        print(f"    • {r['feature']:<26} H={_fmt(r['H_stat'],2)}  p={_fmt(r['p_value'],4)}")
    print(f"\n  Pairwise significant (Holm p<0.05):  {len(pw_sig)}/{len(pw_results)}")
    for r in sorted(pw_sig, key=lambda x: abs(x["rank_biserial_r"]), reverse=True):
        print(f"    • {r['feature']:<26} {r['group_a']} vs {r['group_b']:<14} "
              f"r={_fmt(r['rank_biserial_r'])}  ({r['effect_label']})")
    multi_auc = loocv_results.get("multi_full_topology", float("nan"))
    bin_auc   = loocv_results.get("binary_full_topology", float("nan"))
    print(f"\n  Binary LOOCV AUC (ans vs unans):    {_fmt(bin_auc)}")
    print(f"  Multinomial LOOCV AUC (3-class):    {_fmt(multi_auc)}")

    # ── Output ─────────────────────────────────────────────────────────────
    result = {
        "n_easy":         len(easy_rows),
        "n_hard":         len(hard_rows),
        "n_unanswerable": len(unans_rows),
        "features":       TOPOLOGY_FEATURES,
        "descriptive": {
            k: {f: _descriptive(_extract(v, f)) for f in TOPOLOGY_FEATURES}
            for k, v in groups.items()
        },
        "kruskal_wallis": kw_results,
        "pairwise_mw":    pw_results,
        "loocv_auc":      {k: (None if math.isnan(v) else v)
                           for k, v in loocv_results.items()},
        "judge_score":    {k: _descriptive(v) for k, v in judge_groups.items()},
    }

    if args.output:
        out = Path(args.output)
        with open(out, "w") as f:
            json.dump(result, f, indent=2,
                      default=lambda o: None if (isinstance(o, float)
                                                  and math.isnan(o)) else o)
        print(f"\nJSON results written to {out}")

    if args.html:
        html_path = Path(args.html)
        stem = html_path.parent / html_path.stem
        fig_paths: dict = {}

        if _HAS_PLOT:
            fig1 = str(stem) + "_fig1_boxplots.png"
            fig2 = str(stem) + "_fig2_pairwise_effects.png"
            fig3 = str(stem) + "_fig3_loocv.png"
            fig4 = str(stem) + "_fig4_scatter.png"
            _plot_boxplot_grid(groups, pw_results, fig1)
            _plot_pairwise_effects(pw_results, fig2)
            _plot_loocv_auc(loocv_results, fig3)
            _plot_scatter(groups, fig4)
            fig_paths = {
                "fig1": html_path.stem + "_fig1_boxplots.png",
                "fig2": html_path.stem + "_fig2_pairwise_effects.png",
                "fig3": html_path.stem + "_fig3_loocv.png",
                "fig4": html_path.stem + "_fig4_scatter.png",
            }
            print(f"Figures written to {stem}_fig*.png")
        else:
            print("Warning: matplotlib/seaborn not installed — skipping plots.")

        html = _build_html(groups, kw_results, pw_results,
                           loocv_results, judge_groups, fig_paths)
        with open(html_path, "w") as f:
            f.write(html)
        print(f"HTML report written to {html_path}")


if __name__ == "__main__":
    main()
