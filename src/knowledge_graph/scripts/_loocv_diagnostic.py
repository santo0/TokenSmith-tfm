"""Diagnostic: trace LOOCV internals for the suspect features.

Run from repo root:
    python -m src.knowledge_graph.scripts._loocv_diagnostic
"""

from __future__ import annotations
import json
import math
import warnings
from pathlib import Path


ROOT = Path(__file__).parent.parent.parent.parent


def _load(path: str) -> list[dict]:
    p = ROOT / path
    with open(p) as f:
        data = json.load(f)
    return data["per_query"]


def _load_extra(path: str) -> dict[str, dict]:
    p = ROOT / path
    with open(p) as f:
        data = json.load(f)
    return {r["id"]: r for r in data["per_query"]}


# ── reproduce the exact same row-merging as comparison.py ───────────────────

def build_rows():
    import yaml
    ans_rows   = _load("results_c.json")
    unans_rows = _load("results_c_unanswerable.json")

    with open(ROOT / "tests/benchmarks.yaml") as f:
        bm = yaml.safe_load(f)
    difficulties = {e["id"]: e["difficulty"]
                    for e in bm["benchmarks"]
                    if e.get("difficulty") in ("easy", "hard")}

    easy_rows, hard_rows = [], []
    for row in ans_rows:
        qid = row.get("id") or row.get("query_id")
        if difficulties.get(qid) == "easy":
            easy_rows.append(row)
        elif difficulties.get(qid) == "hard":
            hard_rows.append(row)

    # merge GC features
    gc_ans   = _load_extra("results_c_global_centrality.json")
    gc_unans = _load_extra("results_c_global_centrality_unanswerable.json")
    GC_FEATURES   = ["mean_pagerank", "gc_mean_betweenness", "mean_degree"]
    GC_FIELD_MAP  = {"gc_mean_betweenness": "mean_betweenness"}

    for row in easy_rows + hard_rows:
        qid = row.get("id") or row.get("query_id")
        src = gc_ans.get(qid, {})
        for dest in GC_FEATURES:
            src_k = GC_FIELD_MAP.get(dest, dest)
            row[dest] = src.get(src_k)

    for row in unans_rows:
        qid = row.get("id") or row.get("query_id")
        src = gc_unans.get(qid, {})
        for dest in GC_FEATURES:
            src_k = GC_FIELD_MAP.get(dest, dest)
            row[dest] = src.get(src_k)

    return easy_rows, hard_rows, unans_rows


# ── instrumented binary LOOCV ────────────────────────────────────────────────

def loocv_binary_debug(X, y, feat_name: str):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    Xarr = np.array(X, dtype=float)
    yarr = np.array(y, dtype=int)

    print(f"\n{'='*70}")
    print(f"  LOOCV debug: feature = {feat_name!r}")
    print(f"  n_samples={len(yarr)}, n_class1={sum(yarr==1)}, n_class0={sum(yarr==0)}")
    print(f"  X stats: min={Xarr.min():.6f}, max={Xarr.max():.6f}, "
          f"mean={Xarr.mean():.6f}")
    print(f"  X for class-1 (answerable): mean={Xarr[yarr==1].mean():.6f}")
    print(f"  X for class-0 (unanswerable): mean={Xarr[yarr==0].mean():.6f}")
    print(f"{'='*70}")

    if len(set(y)) < 2:
        print("  → Only one class present; returning nan.")
        return float("nan")

    scaler = StandardScaler()
    loo = LeaveOneOut()
    probs: list[float] = []

    inverted_folds = 0
    bad_pos_idx_folds = 0

    for fold_i, (train_idx, test_idx) in enumerate(loo.split(Xarr)):
        X_tr, X_te = Xarr[train_idx], Xarr[test_idx]
        y_tr = yarr[train_idx]

        if len(set(y_tr)) < 2:
            probs.append(0.5)
            if fold_i < 5:
                print(f"  fold {fold_i}: single-class training → appended 0.5")
            continue

        Xs_tr = scaler.fit_transform(X_tr)
        Xs_te = scaler.transform(X_te)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(max_iter=2000)
            clf.fit(Xs_tr, y_tr)

        prob = clf.predict_proba(Xs_te)[0]
        classes_list = list(clf.classes_)
        has_class1 = 1 in clf.classes_

        if not has_class1:
            bad_pos_idx_folds += 1
            pos_idx = 0
        else:
            pos_idx = classes_list.index(1)

        score = prob[pos_idx]
        true_label = int(yarr[test_idx[0]])

        # Is this fold "inverted"?  Score for class-1 sample is low.
        if true_label == 1 and score < 0.5:
            inverted_folds += 1

        probs.append(score)

        # Print first 5 folds + any fold where pos_idx != 1
        if fold_i < 5 or not has_class1:
            coef = float(clf.coef_[0][0]) if clf.coef_.shape[1] > 0 else float("nan")
            print(f"  fold {fold_i:2d}: y_test={true_label}, "
                  f"classes={classes_list}, pos_idx={pos_idx}, "
                  f"prob={[f'{p:.4f}' for p in prob]}, "
                  f"score={score:.4f}, coef={coef:.6f}, "
                  f"{'⚠ no class1' if not has_class1 else ''}")

    auc = float(roc_auc_score(yarr, probs))
    print(f"\n  Result AUC={auc:.4f}")
    print(f"  Folds with pos_idx bug (1 not in clf.classes_): {bad_pos_idx_folds}")
    print(f"  Folds where score<0.5 for answerable sample: {inverted_folds}/{sum(yarr==1)}")

    # Show the score distributions
    ans_scores  = [probs[i] for i in range(len(probs)) if yarr[i] == 1]
    unans_scores = [probs[i] for i in range(len(probs)) if yarr[i] == 0]
    import numpy as np
    print(f"  Answerable scores:   min={min(ans_scores):.4f}, "
          f"max={max(ans_scores):.4f}, mean={np.mean(ans_scores):.4f}")
    print(f"  Unanswerable scores: min={min(unans_scores):.4f}, "
          f"max={max(unans_scores):.4f}, mean={np.mean(unans_scores):.4f}")

    # Concordant pair count
    concordant = sum(1 for a in ans_scores for u in unans_scores if a > u)
    total_pairs = len(ans_scores) * len(unans_scores)
    print(f"  Concordant pairs: {concordant}/{total_pairs} "
          f"({100*concordant/total_pairs:.1f}%)")

    return auc


# ── QG two-group multiclass test ─────────────────────────────────────────────

def test_qg_multiclass():
    """Check whether _loocv_auc_multiclass errors when unanswerable group is empty."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler, label_binarize

    print(f"\n{'='*70}")
    print("  QG multiclass test: 2 populated groups (easy=2, hard=1, unans empty)")

    # Simulate QG-like data: 11 easy, 18 hard, 0 unanswerable
    rng = np.random.default_rng(42)
    X_easy = rng.normal(1.23, 0.29, (11, 1))
    X_hard = rng.normal(1.46, 0.24, (18, 1))
    Xarr = np.vstack([X_easy, X_hard])
    mc_labels = [2]*11 + [1]*18
    yarr = np.array(mc_labels, dtype=int)
    classes = sorted(set(mc_labels))  # [1, 2]

    print(f"  n_total={len(yarr)}, classes={classes}")

    scaler = StandardScaler()
    loo = LeaveOneOut()
    all_probs = []

    for train_idx, test_idx in loo.split(Xarr):
        X_tr, X_te = Xarr[train_idx], Xarr[test_idx]
        y_tr = yarr[train_idx]
        if len(set(y_tr)) < 2:
            all_probs.append([1/len(classes)] * len(classes))
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
    print(f"  Y_bin shape: {Y_bin.shape}")
    print(f"  Y_bin.shape[1]: {Y_bin.shape[1]}")

    if Y_bin.shape[1] == 1:
        print("  → shape[1]==1, returning nan (BUG: 2-class is treated as 1-column)")
        return

    try:
        auc = float(roc_auc_score(
            Y_bin, np.array(all_probs), multi_class="ovr", average="macro"
        ))
        print(f"  → roc_auc_score succeeded, AUC={auc:.4f}")
    except Exception as e:
        print(f"  → roc_auc_score RAISED: {type(e).__name__}: {e}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    easy_rows, hard_rows, unans_rows = build_rows()

    GC_FEATURES = ["mean_pagerank", "gc_mean_betweenness", "mean_degree"]
    TOPO_FEATURES = ["mean_betweenness"]

    easy_ok   = [r for r in easy_rows  if all(r.get(f) is not None for f in GC_FEATURES)]
    hard_ok   = [r for r in hard_rows  if all(r.get(f) is not None for f in GC_FEATURES)]
    unans_ok  = [r for r in unans_rows if all(r.get(f) is not None for f in GC_FEATURES)]

    print(f"Rows with complete GC features: easy={len(easy_ok)}, "
          f"hard={len(hard_ok)}, unans={len(unans_ok)}")

    all_ans_ok = easy_ok + hard_ok
    all_bin_ok = all_ans_ok + unans_ok
    bin_labels = [1] * len(all_ans_ok) + [0] * len(unans_ok)

    # ── Test each suspect feature ─────────────────────────────────────────────
    for feat in ["gc_mean_betweenness", "mean_pagerank", "mean_degree"]:
        vals = [r.get(feat) for r in all_bin_ok]
        assert all(v is not None for v in vals), f"None in {feat}"
        loocv_binary_debug([[v] for v in vals], bin_labels, feat)

    # ── Topology mean_betweenness ─────────────────────────────────────────────
    topo_ok = [r for r in easy_ok + hard_ok + unans_ok
               if r.get("mean_betweenness") is not None]
    topo_labels = [1]*len(easy_ok+hard_ok) + [0]*len(unans_ok)
    # (topo_ok is same as all_bin_ok for this feature)
    topo_vals = [r.get("mean_betweenness") for r in all_bin_ok]
    loocv_binary_debug([[v] for v in topo_vals], bin_labels, "mean_betweenness (topology)")

    # ── QG multiclass test ───────────────────────────────────────────────────
    test_qg_multiclass()


if __name__ == "__main__":
    main()
