"""A4 — Embedding fallback threshold sweep.

Sweeps the embedding fallback threshold over a configurable grid (default
[0.70, 0.98] in steps of 0.02) to find the value that maximises macro-F1
for query node seeding.

Two independent sweeps:

  Sweep 1 (S1) — varies the threshold applied to the embedding fallback in
                  extract_query_nodes (the CanonicalLookup fallback path).
                  Records macro P/R/F1, mean fraction of matched nodes that
                  arrived via embedding fallback, and mean seed set size.

  Sweep 2 (S3) — varies the per-ngram threshold (embed each query n-gram,
                  no overlap constraint), holding S1's threshold fixed at
                  its sweep-1 optimum.

All n-gram embeddings are pre-computed once before the sweep; only the
threshold comparison changes per step, so the sweep is cheap regardless
of grid size.

Stability is reported as the F1 range within ±0.04 of each optimum.

Usage:
    python -m src.knowledge_graph.scripts.experiment_a4_threshold_sweep \\
        --run-dir data/knowledge_graph/runs/latest \\
        --benchmarks tests/benchmarks.yaml \\
        --output results_a4.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity as sk_cos_sim


def _prf(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    tp = len(predicted & gold)
    p = tp / len(predicted)
    r = tp / len(gold)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A4: Sweep embedding fallback threshold for optimal seeding F1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model (must match KG build)",
    )
    parser.add_argument("--threshold-min", type=float, default=0.70)
    parser.add_argument("--threshold-max", type=float, default=0.98)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    from src.knowledge_graph.io import (
        load_graph_and_chunks,
        load_canonicalization_data,
        resolve_run_dir,
    )
    from src.knowledge_graph.query import TERM_BLACKLIST
    from src.knowledge_graph.ngrams import KW_PATTERN, extract_ngrams_with_spans
    from src.knowledge_graph.normalizer import Normalizer
    from src.knowledge_graph.scripts.eval_utils import load_benchmarks

    root = Path(__file__).parent.parent.parent.parent
    run_dir_path = Path(args.run_dir)
    if not run_dir_path.is_absolute():
        run_dir_path = root / run_dir_path

    print(f"Loading KG from {run_dir_path}...")
    graph, _ = load_graph_and_chunks(str(run_dir_path))
    resolved = resolve_run_dir(str(run_dir_path))
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    if not syn_table:
        raise SystemExit("No canonicalization data found in run dir.")

    normalizer = Normalizer()
    benchmarks = load_benchmarks(args.benchmarks)
    labeled = [b for b in benchmarks if b.get("keywords")]
    print(f"Labeled benchmarks: {len(labeled)}")

    # Build threshold grid
    thresholds: list[float] = []
    t = args.threshold_min
    while t <= args.threshold_max + 1e-9:
        thresholds.append(round(t, 4))
        t += args.threshold_step
    print(f"Threshold grid: {thresholds[0]:.2f} → {thresholds[-1]:.2f}  ({len(thresholds)} steps)")

    # ── Pre-process queries ───────────────────────────────────────────────────
    # For each benchmark, extract n-grams and classify each normalized form:
    #   "direct"  — graph.has_node(norm): already a graph node
    #   "synonym" — norm in syn_table and canon is in graph: synonym table hit
    #   "emb"     — all others; cosine similarity computed below
    #
    # Each candidate is a 4-tuple: (norm, positions, path, canonical)
    # For "emb" candidates, canonical is filled from emb_cache after batch encoding.

    # Tuple type: (norm, positions, path, canonical)
    Candidate = tuple[str, frozenset, str, str]

    all_emb_norms: set[str] = set()
    query_candidates: list[list[Candidate]] = []

    for bm in labeled:
        cands: list[Candidate] = []
        for ngram, positions in extract_ngrams_with_spans(bm["question"], KW_PATTERN):
            if ngram.lower() in TERM_BLACKLIST:
                continue
            normalized = normalizer.normalize([ngram])
            if not normalized:
                continue
            norm = normalized[0]

            if graph.has_node(norm):
                cands.append((norm, positions, "direct", norm))
            elif norm in syn_table and graph.has_node(syn_table[norm]):
                cands.append((norm, positions, "synonym", syn_table[norm]))
            else:
                all_emb_norms.add(norm)
                cands.append((norm, positions, "emb", ""))  # canonical filled below

        query_candidates.append(cands)

    # ── Batch embed all embedding candidates ─────────────────────────────────
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embed_model)

    # emb_cache: norm → (best_canonical_in_graph, cosine_sim)
    emb_cache: dict[str, tuple[str, float]] = {}
    emb_norm_list = sorted(all_emb_norms)

    if emb_norm_list:
        print(f"Embedding {len(emb_norm_list)} unique fallback candidates...")
        cand_embs = model.encode(emb_norm_list, show_progress_bar=True).astype(np.float32)
        sims_matrix = sk_cos_sim(cand_embs, can_emb)          # (n_cands, n_canonical)
        best_idx = np.argmax(sims_matrix, axis=1)
        best_sim_vals = sims_matrix[np.arange(len(emb_norm_list)), best_idx]
        for i, norm in enumerate(emb_norm_list):
            best_canonical = can_kw[int(best_idx[i])]
            if graph.has_node(best_canonical):
                emb_cache[norm] = (best_canonical, float(best_sim_vals[i]))

    gold_norms = [set(normalizer.normalize(bm["keywords"])) for bm in labeled]

    # ── Sweep S1 ─────────────────────────────────────────────────────────────
    # S1 = extract_query_nodes: n-gram + canonical lookup + greedy non-overlapping.
    # Direct hits score 2.0, synonym hits score 1.0, embedding hits score = sim.
    # Greedy selection: sort by (score desc, span_len desc), claim token positions.

    def _eval_s1(threshold: float) -> tuple[dict, list[dict]]:
        per_query: list[dict] = []
        for i, cands in enumerate(query_candidates):
            gold = gold_norms[i]

            scored: list[tuple[str, frozenset, float, str]] = []
            for norm, positions, path, canonical in cands:
                if path == "direct":
                    scored.append((canonical, positions, 2.0, "direct"))
                elif path == "synonym":
                    scored.append((canonical, positions, 1.0, "synonym"))
                elif path == "emb" and norm in emb_cache:
                    emb_canon, emb_sim = emb_cache[norm]
                    if emb_sim >= threshold:
                        scored.append((emb_canon, positions, emb_sim, "emb"))

            scored.sort(key=lambda x: (x[2], len(x[1])), reverse=True)

            used: set[int] = set()
            seen: set[str] = set()
            nodes: list[str] = []
            path_counts = {"direct": 0, "synonym": 0, "emb": 0}

            for term, positions, _, path in scored:
                if positions & used:
                    continue
                used |= positions
                if term not in seen:
                    seen.add(term)
                    nodes.append(term)
                    path_counts[path] += 1

            pred = set(normalizer.normalize(nodes))
            p, r, f1 = _prf(pred, gold)
            total = sum(path_counts.values())
            per_query.append({
                "id": labeled[i]["id"],
                "p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4),
                "seed_n": len(nodes),
                "emb_frac": round(path_counts["emb"] / total, 4) if total > 0 else 0.0,
                "path_counts": path_counts,
            })

        macro = {
            "threshold": threshold,
            "macro_P": round(statistics.mean(q["p"] for q in per_query), 4),
            "macro_R": round(statistics.mean(q["r"] for q in per_query), 4),
            "macro_F1": round(statistics.mean(q["f1"] for q in per_query), 4),
            "mean_seed_n": round(statistics.mean(q["seed_n"] for q in per_query), 3),
            "mean_emb_frac": round(statistics.mean(q["emb_frac"] for q in per_query), 4),
        }
        return macro, per_query

    print(f"\nSweep 1 (S1): {len(thresholds)} steps...")
    sweep_s1_macro: list[dict] = []
    sweep_s1_per_query: list[list[dict]] = []
    for thr in thresholds:
        macro, per_q = _eval_s1(thr)
        sweep_s1_macro.append(macro)
        sweep_s1_per_query.append(per_q)

    W1 = 62
    print(f"\n{'─' * W1}")
    print(f"  S1 sweep — threshold vs macro P / R / F1 / emb-fraction / seed-n")
    print(f"{'─' * W1}")
    print(f"  {'thr':>5}  {'P':>7}  {'R':>7}  {'F1':>7}  {'emb%':>7}  {'seed':>6}")
    print(f"{'─' * W1}")
    for row in sweep_s1_macro:
        print(
            f"  {row['threshold']:>5.2f}  "
            f"{row['macro_P']:>7.4f}  "
            f"{row['macro_R']:>7.4f}  "
            f"{row['macro_F1']:>7.4f}  "
            f"{row['mean_emb_frac']:>7.4f}  "
            f"{row['mean_seed_n']:>6.2f}"
        )
    print(f"{'─' * W1}")

    opt_s1 = max(sweep_s1_macro, key=lambda x: x["macro_F1"])
    opt_s1_idx = sweep_s1_macro.index(opt_s1)
    print(f"  Optimal: threshold={opt_s1['threshold']:.2f}  macro-F1={opt_s1['macro_F1']:.4f}")

    if args.verbose:
        print(f"\n  Per-query breakdown at threshold={opt_s1['threshold']:.2f}:")
        for q in sweep_s1_per_query[opt_s1_idx]:
            print(
                f"    [{q['id']}]  P={q['p']:.3f}  R={q['r']:.3f}  F1={q['f1']:.3f}"
                f"  seed={q['seed_n']}  emb%={q['emb_frac']:.2f}  {q['path_counts']}"
            )

    # ── Sweep S3 ─────────────────────────────────────────────────────────────
    # S3 = per-ngram embedding, no overlap constraint.
    # Direct and synonym hits are always included (sim ≈ 1.0 in both cases).
    # Embedding hits are included if sim >= threshold.

    def _eval_s3(threshold: float) -> tuple[dict, list[dict]]:
        per_query: list[dict] = []
        for i, cands in enumerate(query_candidates):
            gold = gold_norms[i]
            nodes: set[str] = set()
            for norm, _, path, canonical in cands:
                if path in ("direct", "synonym"):
                    nodes.add(canonical)
                elif path == "emb" and norm in emb_cache:
                    emb_canon, emb_sim = emb_cache[norm]
                    if emb_sim >= threshold:
                        nodes.add(emb_canon)
            pred = set(normalizer.normalize(list(nodes)))
            p, r, f1 = _prf(pred, gold)
            per_query.append({
                "id": labeled[i]["id"],
                "p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4),
                "seed_n": len(nodes),
            })

        macro = {
            "threshold": threshold,
            "macro_P": round(statistics.mean(q["p"] for q in per_query), 4),
            "macro_R": round(statistics.mean(q["r"] for q in per_query), 4),
            "macro_F1": round(statistics.mean(q["f1"] for q in per_query), 4),
            "mean_seed_n": round(statistics.mean(q["seed_n"] for q in per_query), 3),
        }
        return macro, per_query

    print(f"\nSweep 2 (S3): {len(thresholds)} steps...")
    sweep_s3_macro: list[dict] = []
    sweep_s3_per_query: list[list[dict]] = []
    for thr in thresholds:
        macro, per_q = _eval_s3(thr)
        sweep_s3_macro.append(macro)
        sweep_s3_per_query.append(per_q)

    W2 = 52
    print(f"\n{'─' * W2}")
    print(f"  S3 sweep — threshold vs macro P / R / F1 / seed-n")
    print(f"{'─' * W2}")
    print(f"  {'thr':>5}  {'P':>7}  {'R':>7}  {'F1':>7}  {'seed':>6}")
    print(f"{'─' * W2}")
    for row in sweep_s3_macro:
        print(
            f"  {row['threshold']:>5.2f}  "
            f"{row['macro_P']:>7.4f}  "
            f"{row['macro_R']:>7.4f}  "
            f"{row['macro_F1']:>7.4f}  "
            f"{row['mean_seed_n']:>6.2f}"
        )
    print(f"{'─' * W2}")

    opt_s3 = max(sweep_s3_macro, key=lambda x: x["macro_F1"])
    opt_s3_idx = sweep_s3_macro.index(opt_s3)
    print(f"  Optimal: threshold={opt_s3['threshold']:.2f}  macro-F1={opt_s3['macro_F1']:.4f}")

    if args.verbose:
        print(f"\n  Per-query breakdown at threshold={opt_s3['threshold']:.2f}:")
        for q in sweep_s3_per_query[opt_s3_idx]:
            print(
                f"    [{q['id']}]  P={q['p']:.3f}  R={q['r']:.3f}  F1={q['f1']:.3f}"
                f"  seed={q['seed_n']}"
            )

    # ── Stability ─────────────────────────────────────────────────────────────
    def _stability(sweep: list[dict], opt_threshold: float, window: float = 0.04) -> dict:
        f1_by_t = {row["threshold"]: row["macro_F1"] for row in sweep}
        nearby = {t: f1 for t, f1 in f1_by_t.items() if abs(t - opt_threshold) <= window + 1e-9}
        f1_vals = list(nearby.values())
        return {
            "opt_threshold": opt_threshold,
            "opt_f1": f1_by_t[opt_threshold],
            "window": window,
            "f1_in_window": {round(t, 2): round(f1, 4) for t, f1 in sorted(nearby.items())},
            "f1_range_in_window": round(max(f1_vals) - min(f1_vals), 4),
        }

    stab_s1 = _stability(sweep_s1_macro, opt_s1["threshold"])
    stab_s3 = _stability(sweep_s3_macro, opt_s3["threshold"])

    print(f"\n  Stability (±0.04 around optimum):")
    print(f"    S1: F1 range = {stab_s1['f1_range_in_window']:.4f}  {stab_s1['f1_in_window']}")
    print(f"    S3: F1 range = {stab_s3['f1_range_in_window']:.4f}  {stab_s3['f1_in_window']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    current = 0.85
    s1_at_current = next((r for r in sweep_s1_macro if abs(r["threshold"] - current) < 1e-9), None)
    s3_at_current = next((r for r in sweep_s3_macro if abs(r["threshold"] - current) < 1e-9), None)

    print(f"\n  Current default (0.85):")
    if s1_at_current:
        delta_s1 = round(opt_s1["macro_F1"] - s1_at_current["macro_F1"], 4)
        print(f"    S1  macro-F1={s1_at_current['macro_F1']:.4f}  (optimal gain: {delta_s1:+.4f})")
    if s3_at_current:
        delta_s3 = round(opt_s3["macro_F1"] - s3_at_current["macro_F1"], 4)
        print(f"    S3  macro-F1={s3_at_current['macro_F1']:.4f}  (optimal gain: {delta_s3:+.4f})")

    result = {
        "sweep_s1": sweep_s1_macro,
        "sweep_s3": sweep_s3_macro,
        "optimal": {
            "s1_threshold": opt_s1["threshold"],
            "s1_macro_F1": opt_s1["macro_F1"],
            "s3_threshold": opt_s3["threshold"],
            "s3_macro_F1": opt_s3["macro_F1"],
        },
        "at_default_0.85": {
            "s1": s1_at_current,
            "s3": s3_at_current,
        },
        "stability": {
            "s1": stab_s1,
            "s3": stab_s3,
        },
        "per_query_at_optimum": {
            "s1": sweep_s1_per_query[opt_s1_idx],
            "s3": sweep_s3_per_query[opt_s3_idx],
        },
        "config": {
            "embed_model": args.embed_model,
            "thresholds": thresholds,
            "n_benchmarks": len(labeled),
        },
    }

    if args.output:
        out = Path(args.output)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults written to {out}")


if __name__ == "__main__":
    load_dotenv()
    main()
