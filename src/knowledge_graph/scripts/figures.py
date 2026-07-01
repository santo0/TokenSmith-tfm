"""Thesis figure and table generation script.

Each figure/table is an independent function. Run all:
    python -m src.knowledge_graph.scripts.figures

Run a single item by name:
    python -m src.knowledge_graph.scripts.figures recall_vs_k

Series C figures
----------------
mean_distance_groups        : strip-over-box of mean_distance across easy/hard/unanswerable
circularity_scatter         : scatter of mean_distance vs FAISS recall@10 (null correlation check)
canonicalization_forest     : forest plot of Cohen's d shift (raw vs canonical) per feature

Series B figures
----------------
recall_vs_k                 : (F1) recall-vs-list-length line plot; FAISS / KG / RRF crossover
incremental_ablation        : (F2) ablation bar chart; recall@10 as components accumulate
per_band_recall             : (F3) grouped bars; easy/hard × FAISS/FAISS+summary/ensemble

Series A figures
----------------
a1_diversity_scatter        : (F-A1) hallucination rate vs F1 for LLM/KeyBERT/YAKE
a2_recall_vs_length         : (F-A2) recall vs chunk-length tertile; adaptive/fixed-5/freeform

Tables (saved as .tex to figures/)
-----------------------------------
table_b5_canonicalization   : (T6) canonical vs raw KGNode, P/R@{5,10,20}
table_per_band_recall       : (T7) FAISS/FAISS+summary/ensemble recall@10 by difficulty band
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).parent.parent.parent.parent
OUT_DIR = ROOT / "figures"

COLOURS = {
    "easy":         "#27ae60",
    "hard":         "#e67e22",
    "unanswerable": "#c0392b",
}

GROUP_ORDER = ["easy", "hard", "unanswerable"]
GROUP_LABELS = ["Easy", "Hard", "Unanswerable"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(exist_ok=True)


def _load_difficulty_labels() -> dict[str, str]:
    """Return {query_id: 'easy'|'hard'} from benchmarks.yaml."""
    with open(ROOT / "tests" / "benchmarks.yaml") as f:
        bms = yaml.safe_load(f)["benchmarks"]
    return {
        b["id"]: b["difficulty"]
        for b in bms
        if b.get("difficulty") in ("easy", "hard")
    }


def _load_mean_distance_groups() -> dict[str, list[float]]:
    """Load per-query mean_distance grouped by difficulty.

    Sources
    -------
    answerable   : results_c_interconcept_distance.json
    unanswerable : results_c_interconcept_distance_unanswerable.json
    """
    labels = _load_difficulty_labels()
    groups: dict[str, list[float]] = {"easy": [], "hard": [], "unanswerable": []}

    with open(ROOT / "results_c_interconcept_distance.json") as f:
        for row in json.load(f)["per_query"]:
            v = row.get("mean_distance")
            label = labels.get(row["id"])
            if v is not None and label in groups:
                groups[label].append(float(v))

    with open(ROOT / "results_c_interconcept_distance_unanswerable.json") as f:
        for row in json.load(f)["per_query"]:
            v = row.get("mean_distance")
            if v is not None:
                groups["unanswerable"].append(float(v))

    return groups


def _load_circularity_pairs() -> tuple[list[float], list[float], list[str]]:
    """Return (xs, ys, group_labels) for the circularity scatter.

    xs           : mean_distance per answerable query
    ys           : FAISS dense recall@10 per answerable query
    group_labels : 'easy' or 'hard' for each point

    Sources
    -------
    mean_distance : results_c_interconcept_distance.json
    dense R@10    : results_b2.json  (column 'dense_R@10')
    """
    labels = _load_difficulty_labels()

    ip_rows: dict[str, float] = {}
    with open(ROOT / "results_c_interconcept_distance.json") as f:
        for row in json.load(f)["per_query"]:
            v = row.get("mean_distance")
            if v is not None:
                ip_rows[row["id"]] = float(v)

    b2_rows: dict[str, float] = {}
    with open(ROOT / "results_b2.json") as f:
        for row in json.load(f)["per_query"]:
            v = row.get("dense_R@10")
            if v is not None:
                b2_rows[row["id"]] = float(v)

    xs, ys, grps = [], [], []
    for qid, dist in ip_rows.items():
        recall = b2_rows.get(qid)
        label = labels.get(qid)
        if recall is not None and label in ("easy", "hard"):
            xs.append(dist)
            ys.append(recall)
            grps.append(label)

    return xs, ys, grps


# ---------------------------------------------------------------------------
# Figure 1 — mean_distance strip-over-box
# ---------------------------------------------------------------------------

def fig_mean_distance_groups(save: bool = True) -> plt.Figure:
    """Strip-over-box plot of mean_distance across easy / hard / unanswerable."""
    groups = _load_mean_distance_groups()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    positions = [1, 2, 3]
    data = [groups[g] for g in GROUP_ORDER]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0, color="#555555"),
        capprops=dict(linewidth=1.0, color="#555555"),
        flierprops=dict(marker=""),
        zorder=2,
    )
    for patch, group in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(COLOURS[group])
        patch.set_alpha(0.25)
        patch.set_edgecolor(COLOURS[group])

    rng = np.random.default_rng(42)
    for pos, group, vals in zip(positions, GROUP_ORDER, data):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            color=COLOURS[group],
            edgecolors="white",
            linewidths=0.6,
            s=40,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(GROUP_LABELS, fontsize=10)
    ax.set_ylabel("Mean inter-concept distance (hops)", fontsize=10)
    ax.set_xlabel("Query difficulty group", fontsize=10)
    ax.set_xlim(0.4, 3.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save:
        _ensure_out_dir()
        path = OUT_DIR / "mean_distance_groups.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — circularity scatter
# ---------------------------------------------------------------------------

def fig_circularity_scatter(save: bool = True) -> plt.Figure:
    """Scatter of mean_distance vs FAISS dense recall@10 (null correlation check)."""
    xs, ys, grps = _load_circularity_pairs()

    # Compute Spearman in-script; fall back to known values if data unavailable
    rho, p = -0.028, 0.885
    if xs:
        from scipy.stats import spearmanr
        result = spearmanr(xs, ys)
        rho, p = float(result.statistic), float(result.pvalue)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    markers = {"easy": "o", "hard": "s"}
    for group in ("easy", "hard"):
        gx = [x for x, g in zip(xs, grps) if g == group]
        gy = [y for y, g in zip(ys, grps) if g == group]
        ax.scatter(
            gx, gy,
            color=COLOURS[group],
            marker=markers[group],
            edgecolors="white",
            linewidths=0.6,
            s=60,
            alpha=0.85,
            label=group.capitalize(),
            zorder=3,
        )

    ax.set_xlabel("Mean inter-concept distance (hops)", fontsize=10)
    ax.set_ylabel("FAISS dense recall@10", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"Circularity check\nSpearman $\\rho$ = {rho:.3f},  $p$ = {p:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save:
        _ensure_out_dir()
        path = OUT_DIR / "circularity_scatter.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — canonicalization forest plot
# ---------------------------------------------------------------------------

_FOREST_DATA = [
    ("community_span",       -3.033, -1.964),
    ("coverage_delta",       -2.175, -1.828),
    ("mean_distance",        -1.737, -1.803),
    ("community_dispersion", -1.557, -1.576),
    ("diameter",             -1.384, -1.303),
    ("density",              -0.049, -0.249),
]

_FOREST_LABELS = {
    "community_span":       "community span",
    "coverage_delta":       "coverage delta",
    "mean_distance":        "mean distance",
    "community_dispersion": "community dispersion",
    "diameter":             "diameter",
    "density":              "density",
}


def fig_canonicalization_forest(save: bool = True) -> plt.Figure:
    """Forest plot: per-feature Cohen's d shift from raw to canonical graph."""
    rows = sorted(_FOREST_DATA, key=lambda r: abs(r[1]), reverse=True)

    features = [r[0] for r in rows]
    d_canon = [r[1] for r in rows]
    d_raw = [r[2] for r in rows]
    y_pos = list(range(len(rows)))
    y_labels = [_FOREST_LABELS[f] for f in features]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)

    for i, (dc, dr) in enumerate(zip(d_canon, d_raw)):
        ax.plot([dr, dc], [i, i], color="#cccccc", linewidth=1.2, zorder=1)

    ax.scatter(
        d_raw, y_pos,
        facecolors="none", edgecolors="#95a5a6",
        s=55, linewidths=1.5, zorder=2, label="Raw",
    )
    ax.scatter(
        d_canon, y_pos,
        facecolors="#2c3e50", edgecolors="#2c3e50",
        s=55, linewidths=1.5, zorder=3, label="Canonical",
    )

    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("Cohen's d (easy vs unanswerable)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    fig.tight_layout()
    if save:
        _ensure_out_dir()
        path = OUT_DIR / "canonicalization_forest.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# Series B data loaders
# ---------------------------------------------------------------------------

def _load_b2_macro() -> dict[str, dict[str, float]]:
    """Return macro recall by retriever and k from results_b2.json."""
    with open(ROOT / "results_b2.json") as f:
        macro = json.load(f)["macro"]
    return {
        "FAISS":    {5: macro["dense_R@5"],    10: macro["dense_R@10"],    20: macro["dense_R@20"]},
        "KG-only":  {5: macro["kg_R@5"],       10: macro["kg_R@10"],       20: macro["kg_R@20"]},
        "RRF": {5: macro["combined_R@5"], 10: macro["combined_R@10"], 20: macro["combined_R@20"]},
    }


def _load_b4_ablation() -> dict[str, float]:
    """Return recall per config from results_b4.json macro."""
    with open(ROOT / "results_b4.json") as f:
        macro = json.load(f)["macro"]
    return {cfg: m["recall"] for cfg, m in macro.items()}


def _load_b4_per_band() -> dict[str, dict[str, float]]:
    """Return {band: {cfg: mean_recall}} from results_b4.json per_query + labels.

    Bands: 'easy', 'hard'. Configs: 'faiss', 'faiss+summary', 'all_five'.
    """
    labels = _load_difficulty_labels()
    with open(ROOT / "results_b4.json") as f:
        per_query = json.load(f)["per_query"]

    cfgs = ["faiss", "faiss+summary", "all_five"]
    result: dict[str, dict[str, list[float]]] = {
        "easy": {c: [] for c in cfgs},
        "hard": {c: [] for c in cfgs},
    }
    for row in per_query:
        band = labels.get(row["id"])
        if band not in result:
            continue
        for cfg in cfgs:
            v = row.get(f"{cfg}_recall")
            if v is not None:
                result[band][cfg].append(float(v))

    return {
        band: {cfg: round(sum(vals) / len(vals), 4) for cfg, vals in cfg_map.items()}
        for band, cfg_map in result.items()
    }


# ---------------------------------------------------------------------------
# F1 — recall-vs-k crossover line plot
# ---------------------------------------------------------------------------

def fig_recall_vs_k(save: bool = True) -> plt.Figure:
    """(F1) Recall vs list length for FAISS, KG-only, and RRF fusion.

    Shows the crossover where RRF overtakes FAISS at k=20 — the visual
    argument for KG's conditional value as k grows.
    """
    data = _load_b2_macro()
    ks = [5, 10, 20]

    line_styles = {
        "FAISS":   dict(color="#2980b9", marker="o", linestyle="-",  linewidth=1.8),
        "KG-only": dict(color="#e67e22", marker="s", linestyle="--", linewidth=1.5),
        "RRF":     dict(color="#27ae60", marker="^", linestyle="-",  linewidth=1.8),
    }

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

    for label, style in line_styles.items():
        ys = [data[label][k] for k in ks]
        ax.plot(ks, ys, label=label, **style, zorder=3)
        for k, y in zip(ks, ys):
            ax.annotate(f"{y:.3f}", xy=(k, y), xytext=(0, 7),
                        textcoords="offset points", ha="center",
                        fontsize=8, color=style["color"])

    # Crossover annotation
    rrf_20 = data["RRF"][20]
    faiss_20 = data["FAISS"][20]
    if rrf_20 > faiss_20:
        ax.annotate(
            "RRF overtakes\nFAISS at k=20",
            xy=(20, (rrf_20 + faiss_20) / 2),
            xytext=(17.5, (rrf_20 + faiss_20) / 2 - 0.06),
            fontsize=8, color="#27ae60",
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.0),
        )

    ax.set_xlabel("k (list length)", fontsize=10)
    ax.set_ylabel("Recall@k", fontsize=10)
    ax.set_xticks(ks)
    ax.set_ylim(0.2, 1.0)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save:
        _ensure_out_dir()
        path = OUT_DIR / "recall_vs_k.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# F2 — incremental ablation bar chart
# ---------------------------------------------------------------------------

_ABLATION_CONFIGS = [
    ("faiss",         "FAISS"),
    ("faiss+kg",      "FAISS + KG"),
    ("faiss+summary", "FAISS + Summary"),
    ("all_five",      "All five"),
]


def fig_incremental_ablation(save: bool = True) -> plt.Figure:
    """(F2) Recall@10 as retriever components accumulate (ablation bar chart).

    Order: FAISS → +KG (flat) → +Summary (main gain) → All five.
    The honest picture: KG adds nothing on top of FAISS alone; the
    summary component drives the gain; all-five adds a small further step.
    """
    ablation = _load_b4_ablation()

    cfg_keys = [c for c, _ in _ABLATION_CONFIGS]
    cfg_labels = [l for _, l in _ABLATION_CONFIGS]
    recalls = [ablation[c] for c in cfg_keys]

    bar_colours = ["#2980b9", "#95a5a6", "#e67e22", "#27ae60"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    bars = ax.bar(cfg_labels, recalls, color=bar_colours, width=0.55,
                  edgecolor="white", linewidth=0.8, zorder=3)

    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.004,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Delta annotations
    for i in range(1, len(recalls)):
        delta = recalls[i] - recalls[i - 1]
        sign = "+" if delta >= 0 else ""
        mid_x = i
        ax.annotate(
            f"{sign}{delta:.3f}",
            xy=(mid_x, recalls[i] + 0.013),
            ha="center", fontsize=8, color="#555555",
        )

    ax.set_ylabel("Recall@10", fontsize=10)
    ax.set_ylim(0.80, 0.93)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()

    if save:
        _ensure_out_dir()
        path = OUT_DIR / "incremental_ablation.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# F3 — per-band recall grouped bars
# ---------------------------------------------------------------------------

def fig_per_band_recall(save: bool = True) -> plt.Figure:
    """(F3) Recall@10 by difficulty band (easy/hard) × retriever configuration.

    Visual partner to T7. Shows whether ensemble gains concentrate in the
    hard band (RQ1-RQ2 link) or are uniform across bands.
    """
    band_data = _load_b4_per_band()

    cfgs = ["faiss", "faiss+summary", "all_five"]
    cfg_labels = ["FAISS", "FAISS + Summary", "All five"]
    bands = ["easy", "hard"]
    band_display = ["Easy", "Hard"]

    x = np.arange(len(bands))
    width = 0.22
    offsets = [-width, 0, width]
    cfg_colours = ["#2980b9", "#e67e22", "#27ae60"]

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

    for i, (cfg, label, colour) in enumerate(zip(cfgs, cfg_labels, cfg_colours)):
        vals = [band_data[band][cfg] for band in bands]
        bars = ax.bar(x + offsets[i], vals, width=width * 0.95,
                      label=label, color=colour, edgecolor="white",
                      linewidth=0.8, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(band_display, fontsize=10)
    ax.set_ylabel("Recall@10", fontsize=10)
    ax.set_ylim(0.80, 0.94)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save:
        _ensure_out_dir()
        path = OUT_DIR / "per_band_recall.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# T6 — B5 canonicalization table (LaTeX)
# ---------------------------------------------------------------------------

def table_b5_canonicalization(save: bool = True) -> str:
    """(T6) Canonical vs raw KGNode, P/R at {5,10,20}. Saves as figures/t6_b5_canon.tex."""
    src = ROOT / "results_b5_canonicalization.json"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found — run 'Exp B5 | Canonicalization and retrieval quality' first."
        )
    with open(src) as f:
        data = json.load(f)

    macro = data["macro"]
    ks = [5, 10, 20]

    rows = []
    for label in ("canonical", "raw"):
        cells = []
        for k in ks:
            cells.append(f"{macro[f'{label}_P@{k}']:.4f}")
            cells.append(f"{macro[f'{label}_R@{k}']:.4f}")
        rows.append((label.capitalize(), cells))

    # Delta row
    delta_cells = []
    for k in ks:
        dp = macro[f"canonical_P@{k}"] - macro[f"raw_P@{k}"]
        dr = macro[f"canonical_R@{k}"] - macro[f"raw_R@{k}"]
        delta_cells.append(f"{dp:+.4f}")
        delta_cells.append(f"{dr:+.4f}")
    rows.append(("$\\Delta$ (can$-$raw)", delta_cells))

    col_header = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{$k={k}$}}" for k in ks
    )
    sub_header = " & ".join(["P & R"] * len(ks))

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{B5: Canonical vs raw \\texttt{KGNodeRetriever} (IDF-weighted). "
        "Precision and recall at $k\\in\\{5,10,20\\}$.}",
        "\\label{tab:b5-canonicalization}",
        "\\begin{tabular}{l" + "cc" * len(ks) + "}",
        "\\toprule",
        f"Graph & {col_header} \\\\",
        f" & {sub_header} \\\\",
        "\\midrule",
    ]
    for i, (row_label, cells) in enumerate(rows):
        if i == len(rows) - 1:
            lines.append("\\midrule")
        lines.append(f"{row_label} & {' & '.join(cells)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    tex = "\n".join(lines)
    if save:
        _ensure_out_dir()
        path = OUT_DIR / "t6_b5_canon.tex"
        path.write_text(tex)
        print(f"Saved: {path}")
    print(tex)
    return tex


# ---------------------------------------------------------------------------
# T7 — per-difficulty-band breakdown (LaTeX)
# ---------------------------------------------------------------------------

def table_per_band_recall(save: bool = True) -> str:
    """(T7) Recall@10 by difficulty band × retriever. Saves as figures/t7_per_band.tex."""
    band_data = _load_b4_per_band()
    labels = _load_difficulty_labels()

    with open(ROOT / "results_b4.json") as f:
        per_query = json.load(f)["per_query"]

    n_easy = sum(1 for r in per_query if labels.get(r["id"]) == "easy")
    n_hard = sum(1 for r in per_query if labels.get(r["id"]) == "hard")

    cfg_display = [
        ("faiss",         "FAISS"),
        ("faiss+summary", "FAISS + Summary"),
        ("all_five",      "All five"),
    ]

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Recall@10 by difficulty band. "
        "\\emph{All five} is the full ensemble from B4. "
        "Unanswerable queries excluded (delegated, not retrieved locally).}",
        "\\label{tab:per-band-recall}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        f"Retriever & Easy ($n={n_easy}$) & Hard ($n={n_hard}$) \\\\",
        "\\midrule",
    ]
    for cfg_key, cfg_label in cfg_display:
        e = band_data["easy"][cfg_key]
        h = band_data["hard"][cfg_key]
        lines.append(f"{cfg_label} & {e:.4f} & {h:.4f} \\\\")

    # Delta (all_five - faiss) per band
    lines.append("\\midrule")
    de = band_data["easy"]["all_five"] - band_data["easy"]["faiss"]
    dh = band_data["hard"]["all_five"] - band_data["hard"]["faiss"]
    lines.append(f"$\\Delta$ (all five $-$ FAISS) & {de:+.4f} & {dh:+.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    tex = "\n".join(lines)
    if save:
        _ensure_out_dir()
        path = OUT_DIR / "t7_per_band.tex"
        path.write_text(tex)
        print(f"Saved: {path}")
    print(tex)
    return tex


# ---------------------------------------------------------------------------
# F-A1 — keyword extraction diversity / agreement scatter
# ---------------------------------------------------------------------------

def fig_a1_diversity_scatter(save: bool = True) -> plt.Figure:
    """(F-A1) Scatter of hallucination rate vs F1 for three extraction methods."""
    with open(ROOT / "results_a1.json") as f:
        a1 = json.load(f)["summary"]

    methods = ["llm", "keybert", "yake"]
    labels = ["LLM", "KeyBERT", "YAKE"]
    colours = ["#2980b9", "#8e44ad", "#27ae60"]
    markers = ["o", "s", "^"]

    xs = [a1[m]["hallucination_rate"]["mean"] for m in methods]
    ys = [a1[m]["f1"]["mean"] for m in methods]
    x_errs = [a1[m]["hallucination_rate"]["std"] for m in methods]
    y_errs = [a1[m]["f1"]["std"] for m in methods]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)

    for x, y, xe, ye, lbl, col, mk in zip(xs, ys, x_errs, y_errs, labels, colours, markers):
        ax.errorbar(
            x, y, xerr=xe, yerr=ye,
            fmt=mk, color=col, markersize=9,
            elinewidth=1.2, capsize=4, capthick=1.2,
            label=lbl, zorder=3,
        )

    ax.set_xlabel(
        "Hallucination rate (fraction of extracted keywords absent from source text)",
        fontsize=9,
    )
    ax.set_ylabel("F1 vs gold labels", fontsize=9)
    ax.set_xlim(-0.05, 0.80)
    ax.set_ylim(-0.02, 0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    if save:
        _ensure_out_dir()
        path = OUT_DIR / "a1_diversity_scatter.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# F-A2 — recall vs chunk length (three extraction strategies)
# ---------------------------------------------------------------------------

def _load_a2b_per_band() -> dict[str, dict[str, dict]]:
    """Compute mean/std recall per length band per condition from records.

    Returns nested dict: band -> condition -> {'mean': float, 'std': float}.
    Conditions: 'A' (adaptive), 'B' (fixed-5), 'C' (freeform).
    Bands derived from by_length thresholds in results_a2b.json.
    """
    with open(ROOT / "results_a2b.json") as f:
        a2b = json.load(f)

    t33 = a2b["by_length"]["thresholds"]["t33"]
    t66 = a2b["by_length"]["thresholds"]["t66"]

    def _band(wc: int) -> str:
        if wc <= t33:
            return "short"
        if wc <= t66:
            return "medium"
        return "long"

    from collections import defaultdict
    recalls: dict = defaultdict(lambda: defaultdict(list))
    for rec in a2b["records"]:
        b = _band(rec["word_count"])
        for cond in ("A", "B", "C"):
            recalls[b][cond].append(rec[cond]["recall"])

    result: dict = {}
    for band in ("short", "medium", "long"):
        result[band] = {}
        for cond in ("A", "B", "C"):
            vals = recalls[band][cond]
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            result[band][cond] = {"mean": mean, "std": std, "n": len(vals)}
    return result


def fig_a2_recall_vs_length(save: bool = True) -> plt.Figure:
    """(F-A2) Recall vs chunk-length tertile for adaptive / fixed-5 / freeform."""
    band_data = _load_a2b_per_band()

    x_labels = ["Short\n(<231 words)", "Medium\n(231-317)", "Long\n(>317)"]
    x_pos = [0, 1, 2]
    bands = ["short", "medium", "long"]

    cond_meta = [
        ("A", "Adaptive",  "#2980b9", "-",  "o"),
        ("B", "Fixed-5",   "#e67e22", "--", "s"),
        ("C", "Freeform",  "#8e44ad", ":",  "^"),
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

    for cond, label, col, ls, mk in cond_meta:
        means = [band_data[b][cond]["mean"] for b in bands]
        stds = [band_data[b][cond]["std"] for b in bands]
        ax.plot(x_pos, means, color=col, linestyle=ls, marker=mk,
                markersize=7, linewidth=1.6, label=label, zorder=3)
        ax.errorbar(x_pos, means, yerr=stds,
                    fmt="none", ecolor=col, elinewidth=1.0,
                    capsize=4, capthick=1.0, alpha=0.6, zorder=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Recall vs gold labels", fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    if save:
        _ensure_out_dir()
        path = OUT_DIR / "a2_recall_vs_length.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# Registry and CLI
# ---------------------------------------------------------------------------

FIGURES: dict[str, callable] = {
    "mean_distance_groups":     fig_mean_distance_groups,
    "circularity_scatter":      fig_circularity_scatter,
    "canonicalization_forest":  fig_canonicalization_forest,
    # Series B figures
    "recall_vs_k":              fig_recall_vs_k,
    "incremental_ablation":     fig_incremental_ablation,
    "per_band_recall":          fig_per_band_recall,
    # Tables
    "table_b5_canonicalization": table_b5_canonicalization,
    "table_per_band_recall":     table_per_band_recall,
    # Series A figures
    "a1_diversity_scatter":     fig_a1_diversity_scatter,
    "a2_recall_vs_length":      fig_a2_recall_vs_length,
}


def main() -> None:
    targets = sys.argv[1:] or list(FIGURES)
    for name in targets:
        if name not in FIGURES:
            print(f"Unknown figure '{name}'. Available: {list(FIGURES)}")
            sys.exit(1)
        print(f"Generating: {name}")
        FIGURES[name]()


if __name__ == "__main__":
    main()
