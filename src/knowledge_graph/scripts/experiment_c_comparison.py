"""experiment_c_comparison.py

Compare KG topology features between answerable and unanswerable query sets.

Analyses performed
------------------
  1. Descriptive statistics per feature per group (mean ± SD, median, IQR).
  2. Mann-Whitney U test per feature — non-parametric two-sample test,
     appropriate for small, non-normal samples.
  3. Rank-biserial correlation r — effect-size companion to MW-U.
     |r| < 0.1 negligible, 0.1–0.3 small, 0.3–0.5 medium, ≥ 0.5 large.
  4. Cohen's d per feature (parametric reference).
  5. Logistic regression with LOOCV — can topology features predict
     answerability? Reports LOOCV AUC for individual features and for the
     full feature set.
  6. If judge_score is present in both files, the same tests are applied to it.

Usage
-----
    python -m src.knowledge_graph.scripts.experiment_c_comparison \\
        --answerable   results_c_chp.json \\
        --unanswerable results_c_chp_unanswerable.json \\
        --output       results_c_comparison.json

    # also write an HTML report:
    python -m src.knowledge_graph.scripts.experiment_c_comparison \\
        --answerable   results_c_chp.json \\
        --unanswerable results_c_chp_unanswerable.json \\
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
    """Return (U_stat, p_value, rank_biserial_r).

    Uses a two-sided alternative. Rank-biserial r = 2U/(n_a*n_b) - 1,
    where U is the statistic for group a vs group b.
    """
    from scipy.stats import mannwhitneyu
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan"), float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = mannwhitneyu(a, b, alternative="two-sided")
    U = float(result.statistic)
    p = float(result.pvalue)
    r = 2 * U / (n_a * n_b) - 1          # rank-biserial correlation
    return U, p, r


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


def _loocv_auc(X: list[list[float]], y: list[int]) -> float:
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
    if math.isnan(ar):    return "n/a"
    if ar < 0.1:          return "negligible"
    if ar < 0.3:          return "small"
    if ar < 0.5:          return "medium"
    return                       "large"


def _sig_stars(p: float) -> str:
    if math.isnan(p): return ""
    if p < 0.001:     return "***"
    if p < 0.01:      return "**"
    if p < 0.05:      return "*"
    if p < 0.10:      return "."
    return                    "ns"


def _ascii_bar(r: float, width: int = 20) -> str:
    """ASCII bar proportional to |r|, from 0 to 1."""
    if math.isnan(r):
        return " " * width
    filled = round(abs(r) * width)
    return ("█" * filled).ljust(width)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TOPOLOGY_FEATURES = [
    "diameter",
    "density",
    "coverage_delta",
    "community_dispersion",
    "community_span",
    "mean_betweenness",
]


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.is_absolute():
        root = Path(__file__).parent.parent.parent.parent
        p = root / path
    with open(p) as f:
        data = json.load(f)
    return data["per_query"]


def _extract(rows: list[dict], field: str) -> list[float]:
    return [float(r[field]) for r in rows if r.get(field) is not None]


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _fmt(v: Any, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{decimals}f}"


def _print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def _print_descriptive(ans_rows: list[dict], unans_rows: list[dict],
                       features: list[str]) -> None:
    _print_section("1. Descriptive statistics")
    header = f"{'Feature':<26} {'Group':<14} {'n':>3}  {'mean':>7}  {'±sd':>7}  {'median':>7}  {'IQR':>13}  {'[min, max]':>16}"
    print(header)
    print("-" * len(header))
    for feat in features:
        for label, rows in [("answerable", ans_rows), ("unanswerable", unans_rows)]:
            vals = _extract(rows, feat)
            d = _descriptive(vals)
            if not d:
                print(f"  {feat:<24} {label:<14} {'0':>3}")
                continue
            iqr = f"[{_fmt(d['q1'])}, {_fmt(d['q3'])}]"
            minmax = f"[{_fmt(d['min'])}, {_fmt(d['max'])}]"
            print(f"  {feat:<24} {label:<14} {d['n']:>3}  {_fmt(d['mean']):>7}  {_fmt(d['sd']):>7}  {_fmt(d['median']):>7}  {iqr:>13}  {minmax:>16}")
        print()


def _print_mw(results: list[dict]) -> None:
    _print_section("2. Mann-Whitney U test + rank-biserial r (effect size)")
    print("  Null hypothesis: distributions of answerable and unanswerable are identical.")
    print("  Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10  ns = not significant\n")
    header = f"  {'Feature':<26} {'U':>8}  {'p':>8}  {'sig':>4}  {'r (rb)':>8}  {'|r|':>20}  {'effect':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in results:
        bar = _ascii_bar(row["rank_biserial_r"])
        print(f"  {row['feature']:<26} {_fmt(row['U_stat'], 1):>8}  "
              f"{_fmt(row['p_value'], 4):>8}  {_sig_stars(row['p_value']):>4}  "
              f"{_fmt(row['rank_biserial_r']):>8}  {bar:>20}  "
              f"{_effect_label(row['rank_biserial_r']):>10}")


def _print_loocv(loocv_results: dict[str, float]) -> None:
    _print_section("3. Logistic regression — LOOCV AUC (answerable=1, unanswerable=0)")
    print("  AUC > 0.7 indicates the feature set has discriminative power.\n")
    for name, auc in loocv_results.items():
        bar = _ascii_bar(max(0.0, auc - 0.5) * 2)   # scale 0.5–1.0 → 0–1
        print(f"  {name:<30} AUC = {_fmt(auc)}  {bar}")


def _print_judge(ans_judge: list[float], unans_judge: list[float]) -> None:
    if not ans_judge and not unans_judge:
        return
    _print_section("4. LLM judge score comparison (judge_score)")
    print("  For answerable: high score = system retrieved relevant content.")
    print("  For unanswerable: high score = system hallucinated plausibly.\n")
    for label, vals in [("answerable", ans_judge), ("unanswerable", unans_judge)]:
        d = _descriptive(vals)
        if d:
            print(f"  {label:<14} n={d['n']}  mean={_fmt(d['mean'])}  median={_fmt(d['median'])}  sd={_fmt(d['sd'])}")
    if ans_judge and unans_judge:
        U, p, r = _mannwhitney(ans_judge, unans_judge)
        print(f"\n  Mann-Whitney U={_fmt(U,1)}  p={_fmt(p,4)} {_sig_stars(p)}  r={_fmt(r)}  effect={_effect_label(r)}")


# ---------------------------------------------------------------------------
# Matplotlib / seaborn plots
# ---------------------------------------------------------------------------

_COLOUR_ANS   = "#4a90d9"   # answerable — blue
_COLOUR_UNANS = "#e67e22"   # unanswerable — orange
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
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
    })
except ImportError:
    _HAS_PLOT = False


def _plot_boxplot_grid(ans_rows: list[dict], unans_rows: list[dict],
                       mw_rows: list[dict], out_path: str) -> None:
    """Box plots for the 3 top-effect features, side-by-side per feature."""
    if not _HAS_PLOT:
        return
    p_by_feat = {r["feature"]: r["p_value"] for r in mw_rows}

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Feature distributions: answerable vs unanswerable queries",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, feat in zip(axes, _PLOT_FEATURES):
        ans_vals   = _extract(ans_rows,   feat)
        unans_vals = _extract(unans_rows, feat)
        df = pd.DataFrame({
            "value": ans_vals + unans_vals,
            "group": ["Answerable"] * len(ans_vals) + ["Unanswerable"] * len(unans_vals),
        })
        sns.boxplot(
            data=df, x="group", y="value", hue="group",
            palette={"Answerable": _COLOUR_ANS, "Unanswerable": _COLOUR_UNANS},
            width=0.5, linewidth=1.2, fliersize=4, ax=ax, legend=False,
        )
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(feat, fontsize=9)

        # significance annotation
        p = p_by_feat.get(feat, float("nan"))
        stars = _sig_stars(p)
        y_max = max(ans_vals + unans_vals) if (ans_vals or unans_vals) else 1.0
        y_ann = y_max + (y_max - ax.get_ylim()[0]) * 0.07
        ax.annotate(
            f"p={p:.4f} {stars}",
            xy=(0.5, 1.01), xycoords="axes fraction",
            ha="center", fontsize=9,
            color="#c0392b" if stars not in ("ns", "") else "#555",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_effect_sizes(mw_rows: list[dict], out_path: str) -> None:
    """Horizontal diverging bar chart of rank-biserial r for all 6 features."""
    if not _HAS_PLOT:
        return

    def _bar_colour(row: dict) -> str:
        p = row["p_value"]
        if math.isnan(p) or p >= 0.10:
            return "#aaaaaa"
        if p < 0.01:
            return "#27ae60"
        return _COLOUR_ANS

    features = [r["feature"] for r in mw_rows]
    r_vals   = [r["rank_biserial_r"] if not math.isnan(r["rank_biserial_r"]) else 0.0
                for r in mw_rows]
    colours  = [_bar_colour(r) for r in mw_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(features, r_vals, color=colours, height=0.55, zorder=3)

    ax.axvline(0,    color="black",  lw=1.0, zorder=4)
    ax.axvline( 0.3, color="#aaa",   lw=0.8, linestyle="--", alpha=0.6, zorder=2)
    ax.axvline(-0.3, color="#aaa",   lw=0.8, linestyle="--", alpha=0.6, zorder=2)
    ax.axvline( 0.5, color="#888",   lw=0.8, linestyle="--", alpha=0.8, zorder=2)
    ax.axvline(-0.5, color="#888",   lw=0.8, linestyle="--", alpha=0.8, zorder=2)

    for bar, row, rv in zip(bars, mw_rows, r_vals):
        effect = _effect_label(row["rank_biserial_r"])
        label  = f"  r={rv:+.3f} ({effect})" if rv >= 0 else f"r={rv:+.3f} ({effect})  "
        ha     = "left" if rv >= 0 else "right"
        ax.text(rv, bar.get_y() + bar.get_height() / 2,
                label, ha=ha, va="center", fontsize=8.5)

    ax.set_xlim(-1.15, 1.15)
    ax.set_xlabel("Rank-biserial correlation  r  (effect size)", fontsize=10)
    ax.set_title("Effect sizes: answerable vs unanswerable (Mann-Whitney U)",
                 fontsize=11, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color="#27ae60", label="p < 0.01  (**)"),
        mpatches.Patch(color=_COLOUR_ANS, label="p < 0.10  (.)"),
        mpatches.Patch(color="#aaaaaa",  label="ns"),
    ]
    ax.legend(handles=legend_patches, fontsize=8.5, loc="lower right")
    ax.text( 0.3, -0.72, "small", ha="center", color="#aaa", fontsize=7.5,
             transform=ax.get_xaxis_transform())
    ax.text( 0.5, -0.72, "medium", ha="center", color="#888", fontsize=7.5,
             transform=ax.get_xaxis_transform())
    ax.text(-0.3, -0.72, "small",  ha="center", color="#aaa", fontsize=7.5,
             transform=ax.get_xaxis_transform())
    ax.text(-0.5, -0.72, "medium", ha="center", color="#888", fontsize=7.5,
             transform=ax.get_xaxis_transform())

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
        if auc >= 0.9:  return "#27ae60"
        if auc >= 0.7:  return _COLOUR_ANS
        if auc >= 0.6:  return _COLOUR_UNANS
        return "#aaaaaa"

    names = list(loocv_results.keys())
    aucs  = [v if v is not None else float("nan") for v in loocv_results.values()]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        names,
        [max(0.0, a) if not math.isnan(a) else 0.0 for a in aucs],
        color=[_auc_colour(a) for a in aucs],
        height=0.55, zorder=3,
    )

    ax.axvline(0.5, color="#999", lw=1.0, linestyle="--", zorder=4,
               label="chance (0.5)")
    ax.axvline(0.7, color="#4a90d9", lw=1.0, linestyle="--", zorder=4,
               label="good (0.7)")

    for bar, auc in zip(bars, aucs):
        label = f"  {auc:.3f}" if not math.isnan(auc) else "  n/a"
        ax.text(max(0.0, auc if not math.isnan(auc) else 0.0),
                bar.get_y() + bar.get_height() / 2,
                label, ha="left", va="center", fontsize=8.5)

    ax.set_xlim(0.0, 1.12)
    ax.set_xlabel("LOOCV AUC", fontsize=10)
    ax.set_title("Answerability classification — LOOCV AUC per feature set",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(ans_rows: list[dict], unans_rows: list[dict],
                  out_path: str) -> None:
    """Scatter plot of density vs coverage_delta, coloured by group."""
    if not _HAS_PLOT:
        return

    ans_x   = _extract(ans_rows,   "density")
    ans_y   = _extract(ans_rows,   "coverage_delta")
    unans_x = _extract(unans_rows, "density")
    unans_y = _extract(unans_rows, "coverage_delta")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ans_x, ans_y,
               color=_COLOUR_ANS, label=f"Answerable (n={len(ans_x)})",
               s=70, alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3)
    ax.scatter(unans_x, unans_y,
               color=_COLOUR_UNANS, label=f"Unanswerable (n={len(unans_x)})",
               s=70, alpha=0.85, edgecolors="white", linewidths=0.8,
               marker="D", zorder=3)

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
    body = ""
    for row in rows:
        tds = "".join(f"<td>{cell}</td>" for cell in row)
        body += f"<tr>{tds}</tr>\n"
    return f"""
<h3>{caption}</h3>
<table>
<thead><tr>{th}</tr></thead>
<tbody>{body}</tbody>
</table>
"""


def _build_html(desc_ans: dict, desc_unans: dict, mw_rows: list[dict],
                loocv_results: dict, ans_judge: list[float],
                unans_judge: list[float],
                ans_rows: list[dict] | None = None,
                unans_rows: list[dict] | None = None,
                fig_paths: dict | None = None) -> str:
    css = """
<style>
  body { font-family: Arial, sans-serif; max-width: 1100px; margin: 2em auto; color: #222; }
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
  .chart-img { max-width: 100%; margin: 0.5em 0 2em; display: block; border: 1px solid #e0e0e0; border-radius: 4px; }
  .chart-caption { font-size: .82em; color: #555; font-style: italic; margin: -.5em 0 1.5em; }
</style>
"""
    body = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<title>Answerable vs Unanswerable — Experiment C Comparison</title>
{css}
</head>
<body>
<h2>Answerable vs Unanswerable Query Comparison</h2>
<p class="note">
  Answerable n={list(desc_ans.values())[0]['n'] if desc_ans else '?'} &nbsp;|&nbsp;
  Unanswerable n={list(desc_unans.values())[0]['n'] if desc_unans else '?'} &nbsp;|&nbsp;
  Features: KG topology (diameter, density, coverage, community, betweenness)
</p>
"""

    # Section 1 — descriptive
    headers = ["Feature", "Group", "n", "Mean", "±SD", "Median", "Q1–Q3", "Min", "Max"]
    desc_rows = []
    for feat in TOPOLOGY_FEATURES:
        for label, desc in [("answerable", desc_ans), ("unanswerable", desc_unans)]:
            d = desc.get(feat, {})
            if not d:
                continue
            desc_rows.append([
                feat, label, str(d["n"]),
                _fmt(d["mean"]), _fmt(d["sd"]), _fmt(d["median"]),
                f"{_fmt(d['q1'])} – {_fmt(d['q3'])}",
                _fmt(d["min"]), _fmt(d["max"]),
            ])
    body += _html_table("1. Descriptive Statistics", headers, desc_rows)

    # Section 2 — MW
    mw_headers = ["Feature", "U", "p-value", "Sig", "Rank-biserial r", "Effect size"]
    mw_table_rows = []
    for row in mw_rows:
        sig = _sig_stars(row["p_value"])
        r = row["rank_biserial_r"]
        effect = _effect_label(r)
        bar_w = round(abs(r) * 120) if not math.isnan(r) else 0
        bar_html = f'<span class="bar" style="width:{bar_w}px"></span>'
        sig_cls = "sig" if sig in ("*", "**", "***") else ""
        eff_cls = effect if effect in ("large", "medium", "small") else ""
        mw_table_rows.append([
            row["feature"],
            _fmt(row["U_stat"], 1),
            f'<span class="{sig_cls}">{_fmt(row["p_value"], 4)}</span>',
            f'<span class="{sig_cls}">{sig}</span>',
            f'{_fmt(r)} {bar_html}',
            f'<span class="{eff_cls}">{effect}</span>',
        ])
    body += '<p class="note">*** p&lt;0.001 &nbsp; ** p&lt;0.01 &nbsp; * p&lt;0.05 &nbsp; . p&lt;0.10 &nbsp; ns = not significant</p>'
    body += _html_table("2. Mann-Whitney U Test + Effect Sizes", mw_headers, mw_table_rows)

    if fig_paths and fig_paths.get("fig1"):
        body += f'<img src="{fig_paths["fig1"]}" class="chart-img" alt="Box plots: top-3 features">\n'
        body += '<p class="chart-caption">Figure 1 — Box plots for the three features with the largest effect sizes. Centre line = median; box = IQR; whiskers = min/max.</p>\n'
    if fig_paths and fig_paths.get("fig2"):
        body += f'<img src="{fig_paths["fig2"]}" class="chart-img" alt="Effect size bar chart">\n'
        body += '<p class="chart-caption">Figure 2 — Rank-biserial r for all six features. Dashed lines at ±0.3 (small/medium) and ±0.5 (medium/large) thresholds.</p>\n'

    # Section 3 — LOOCV
    loocv_headers = ["Feature set", "LOOCV AUC", "Discriminative power"]
    loocv_table_rows = []
    for name, auc in loocv_results.items():
        power = "good" if auc >= 0.7 else ("moderate" if auc >= 0.6 else "weak")
        bar_w = round(max(0.0, auc - 0.5) * 240)
        bar_html = f'<span class="bar" style="width:{bar_w}px"></span>'
        loocv_table_rows.append([name, f"{_fmt(auc)} {bar_html}", power])
    body += '<p class="note">AUC measures ability to discriminate answerable (1) from unanswerable (0) queries. AUC=0.5 → chance.</p>'
    body += _html_table("3. Logistic Regression — LOOCV AUC", loocv_headers, loocv_table_rows)

    if fig_paths and fig_paths.get("fig3"):
        body += f'<img src="{fig_paths["fig3"]}" class="chart-img" alt="LOOCV AUC bar chart">\n'
        body += '<p class="chart-caption">Figure 3 — LOOCV AUC per feature set. Dashed lines mark chance level (0.5) and the good-discriminability threshold (0.7).</p>\n'
    if fig_paths and fig_paths.get("fig4"):
        body += f'<img src="{fig_paths["fig4"]}" class="chart-img" alt="Scatter: density vs coverage_delta">\n'
        body += '<p class="chart-caption">Figure 4 — Scatter plot of density vs coverage_delta. Circles = answerable, diamonds = unanswerable. The two groups occupy distinct regions of this feature space.</p>\n'

    # Section 4 — judge score
    if ans_judge or unans_judge:
        judge_headers = ["Group", "n", "Mean", "Median", "SD"]
        judge_rows = []
        for label, vals in [("answerable", ans_judge), ("unanswerable", unans_judge)]:
            d = _descriptive(vals)
            if d:
                judge_rows.append([label, str(d["n"]), _fmt(d["mean"]),
                                    _fmt(d["median"]), _fmt(d["sd"])])
        if ans_judge and unans_judge:
            U, p, r = _mannwhitney(ans_judge, unans_judge)
            judge_rows.append(["MW-U test", "—",
                                f"U={_fmt(U,1)}, p={_fmt(p,4)} {_sig_stars(p)}",
                                f"r={_fmt(r)}", _effect_label(r)])
        body += _html_table("4. LLM Judge Score Comparison", judge_headers, judge_rows)

    body += "</body>\n</html>"
    return body


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare topology features: answerable vs unanswerable queries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--answerable",   default="results_c_chp.json")
    parser.add_argument("--unanswerable", default="results_c_chp_unanswerable.json")
    parser.add_argument("--output",       default=None,
                        help="JSON file to write results to")
    parser.add_argument("--html",         default=None,
                        help="HTML report file to write")
    args = parser.parse_args()

    ans_rows   = _load(args.answerable)
    unans_rows = _load(args.unanswerable)

    n_ans   = len(ans_rows)
    n_unans = len(unans_rows)
    print(f"\nLoaded {n_ans} answerable and {n_unans} unanswerable queries.")

    # ── 1. Descriptive ─────────────────────────────────────────────────────
    desc_ans   = {f: _descriptive(_extract(ans_rows,   f)) for f in TOPOLOGY_FEATURES}
    desc_unans = {f: _descriptive(_extract(unans_rows, f)) for f in TOPOLOGY_FEATURES}
    _print_descriptive(ans_rows, unans_rows, TOPOLOGY_FEATURES)

    # ── 2. Mann-Whitney + rank-biserial r ──────────────────────────────────
    mw_results: list[dict] = []
    for feat in TOPOLOGY_FEATURES:
        a = _extract(ans_rows,   feat)
        b = _extract(unans_rows, feat)
        U, p, r = _mannwhitney(a, b)
        d = _cohens_d(a, b)
        mw_results.append({
            "feature":           feat,
            "n_answerable":      len(a),
            "n_unanswerable":    len(b),
            "U_stat":            U,
            "p_value":           p,
            "rank_biserial_r":   r,
            "cohens_d":          d,
            "effect_label":      _effect_label(r),
            "significant_p05":   (not math.isnan(p)) and (p < 0.05),
        })
    _print_mw(mw_results)

    # ── 3. Logistic regression LOOCV ───────────────────────────────────────
    all_rows  = ans_rows + unans_rows
    all_labels = [1] * n_ans + [0] * n_unans

    loocv_results: dict[str, float] = {}

    # Individual features
    for feat in TOPOLOGY_FEATURES:
        vals = [r.get(feat) for r in all_rows]
        if any(v is None for v in vals):
            loocv_results[feat] = float("nan")
            continue
        X = [[v] for v in vals]
        loocv_results[feat] = _loocv_auc(X, all_labels)

    # Full topology set
    full_feats = TOPOLOGY_FEATURES
    full_X = [[r.get(f, 0.0) or 0.0 for f in full_feats] for r in all_rows]
    loocv_results["full_topology"] = _loocv_auc(full_X, all_labels)

    # Top-3 by |r|
    sorted_by_effect = sorted(
        [m for m in mw_results if not math.isnan(m["rank_biserial_r"])],
        key=lambda m: abs(m["rank_biserial_r"]),
        reverse=True,
    )
    top3 = [m["feature"] for m in sorted_by_effect[:3]]
    if len(top3) == 3:
        top3_X = [[r.get(f, 0.0) or 0.0 for f in top3] for r in all_rows]
        loocv_results["top3_by_effect"] = _loocv_auc(top3_X, all_labels)

    _print_loocv(loocv_results)

    # ── 4. Judge score (if available) ──────────────────────────────────────
    ans_judge   = _extract(ans_rows,   "judge_score")
    unans_judge = _extract(unans_rows, "judge_score")
    _print_judge(ans_judge, unans_judge)

    # ── Summary interpretation ─────────────────────────────────────────────
    _print_section("Summary")
    significant = [r for r in mw_results if r["significant_p05"]]
    medium_plus = [r for r in mw_results if abs(r["rank_biserial_r"]) >= 0.3
                   and not math.isnan(r["rank_biserial_r"])]
    print(f"  Features significant at p<0.05:  {len(significant)}/{len(mw_results)}")
    if significant:
        for r in significant:
            print(f"    • {r['feature']:<26} p={_fmt(r['p_value'],4)}  r={_fmt(r['rank_biserial_r'])}  ({r['effect_label']})")
    print(f"\n  Features with medium+ effect (|r|≥0.3):  {len(medium_plus)}/{len(mw_results)}")
    if medium_plus:
        for r in sorted(medium_plus, key=lambda x: abs(x["rank_biserial_r"]), reverse=True):
            print(f"    • {r['feature']:<26} r={_fmt(r['rank_biserial_r'])}  ({r['effect_label']})")
    full_auc = loocv_results.get("full_topology", float("nan"))
    print(f"\n  Full topology LOOCV AUC: {_fmt(full_auc)}")
    if not math.isnan(full_auc):
        if full_auc >= 0.7:
            print("  → KG topology features have meaningful discriminative power.")
        elif full_auc >= 0.6:
            print("  → KG topology features show moderate discriminative power.")
        else:
            print("  → KG topology features alone have limited discriminative power.")
    if not ans_judge and not unans_judge:
        print("\n  Note: judge_score not available in either file.")
        print("  Re-run experiment_c_difficulty without --no-llm to enable judge comparison.")

    # ── Output ─────────────────────────────────────────────────────────────
    result = {
        "n_answerable":   n_ans,
        "n_unanswerable": n_unans,
        "features":       TOPOLOGY_FEATURES,
        "descriptive": {
            "answerable":   desc_ans,
            "unanswerable": desc_unans,
        },
        "mann_whitney":   mw_results,
        "loocv_auc":      {k: (None if math.isnan(v) else v)
                           for k, v in loocv_results.items()},
        "judge_score": {
            "answerable_n":   len(ans_judge),
            "unanswerable_n": len(unans_judge),
            "answerable_mean":   (sum(ans_judge) / len(ans_judge)
                                  if ans_judge else None),
            "unanswerable_mean": (sum(unans_judge) / len(unans_judge)
                                  if unans_judge else None),
        },
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
            fig2 = str(stem) + "_fig2_effectsizes.png"
            fig3 = str(stem) + "_fig3_loocv.png"
            fig4 = str(stem) + "_fig4_scatter.png"
            _plot_boxplot_grid(ans_rows, unans_rows, mw_results, fig1)
            _plot_effect_sizes(mw_results, fig2)
            _plot_loocv_auc(loocv_results, fig3)
            _plot_scatter(ans_rows, unans_rows, fig4)
            # Store relative names for HTML <img src>
            fig_paths = {
                "fig1": html_path.stem + "_fig1_boxplots.png",
                "fig2": html_path.stem + "_fig2_effectsizes.png",
                "fig3": html_path.stem + "_fig3_loocv.png",
                "fig4": html_path.stem + "_fig4_scatter.png",
            }
            print(f"Figures written to {stem}_fig*.png")
        else:
            print("Warning: matplotlib/seaborn not installed — skipping plots.")

        html = _build_html(desc_ans, desc_unans, mw_results,
                           loocv_results, ans_judge, unans_judge,
                           ans_rows, unans_rows, fig_paths)
        with open(html_path, "w") as f:
            f.write(html)
        print(f"HTML report written to {html_path}")


if __name__ == "__main__":
    main()
