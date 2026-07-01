"""A2b — Adaptive vs fixed vs free-form top-n keyword extraction quality.

Loads annotated chunks (produced by annotate_chunks.py), runs three extraction
conditions via OpenRouterClient, then evaluates each against the gold keywords.

Conditions:
  A  adaptive   top_n = ceil(sqrt(word_count)) per chunk
  B  fixed-5    top_n = 5 for all chunks
  C  free-form  no count constraint; model decides

Metrics per condition (macro-averaged over chunks):
  precision, recall, F1 vs gold keywords
  mean extracted count, mean gold count

Usage:
    python -m src.knowledge_graph.scripts.experiment_a2b_adaptive_topn \\
        --annotated-chunks annotated_chunks.json \\
        --model google/gemini-2.5-flash \\
        --output results_a2b.json -v
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from pathlib import Path

from dotenv import load_dotenv


_FREEFORM_PROMPT = (
    "You are a linguistic analysis expert. Analyze the provided text and identify "
    "the most relevant and descriptive keywords or short phrases (1-3 words). "
    "Focus on technical terms, proper nouns, and central concepts. "
    "Return the result as a raw JSON list of strings. "
    "Do not include any other text or explanation."
)


def _parse_keywords(content: str) -> list[str]:
    try:
        kws = json.loads(content)
        if isinstance(kws, list):
            return [str(k) for k in kws]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*?\]", content, re.DOTALL)
    if match:
        try:
            kws = json.loads(match.group(0))
            if isinstance(kws, list):
                return [str(k) for k in kws]
        except json.JSONDecodeError:
            pass
    return []


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _metrics(extracted: list[str], reference: list[str]) -> dict[str, float]:
    ext_set = {_normalize(k) for k in extracted if k}
    ref_set = {_normalize(k) for k in reference if k}
    if not ext_set and not ref_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not ext_set or not ref_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(ext_set & ref_set)
    precision = tp / len(ext_set)
    recall = tp / len(ref_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _build_request(system: str, text: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Documents: {text}"},
        ]
    }


def _aggregate(scores: list[dict[str, float]], key: str) -> dict[str, float]:
    vals = [s[key] for s in scores]
    return {
        f"mean_{key}": round(statistics.mean(vals), 4),
        f"std_{key}": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A2b: Adaptive vs fixed vs free-form top-n keyword extraction quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotated-chunks",
        default="annotated_chunks.json",
        help="JSON produced by annotate_chunks.py",
    )
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None, help="Write results JSON here")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("No OpenRouter API key. Set OPENROUTER_API_KEY or pass --api-key.")

    from src.knowledge_graph.openrouter_client import OpenRouterClient
    from src.knowledge_graph.prompts import OPENROUTER_KEYWORD_EXTRACTION_PROMPT

    root = Path(__file__).parent.parent.parent.parent
    ann_path = Path(args.annotated_chunks)
    if not ann_path.is_absolute():
        ann_path = root / ann_path

    with open(ann_path) as f:
        ann_data = json.load(f)

    docs = [r for r in ann_data["records"] if r.get("keywords") is not None]
    print(f"Loaded {len(docs)} annotated chunks from {ann_path}.")

    client = OpenRouterClient(api_key, retries=2)

    # ── Build requests for each condition ────────────────────────────────────
    top_ns_a: list[int] = []
    requests_a: list[dict] = []
    requests_b: list[dict] = []
    requests_c: list[dict] = []

    for doc in docs:
        text = doc["text"]
        word_count = doc.get("word_count", len(text.split()))
        top_n_a = math.ceil(math.sqrt(word_count))
        top_ns_a.append(top_n_a)
        requests_a.append(_build_request(OPENROUTER_KEYWORD_EXTRACTION_PROMPT.format(top_n=top_n_a), text))
        requests_b.append(_build_request(OPENROUTER_KEYWORD_EXTRACTION_PROMPT.format(top_n=5), text))
        requests_c.append(_build_request(_FREEFORM_PROMPT, text))

    # ── Run conditions sequentially ───────────────────────────────────────────
    print(f"Running condition A (adaptive) — {len(requests_a)} requests to {args.model}...")
    outcomes_a = client.chat_many(requests_a, model=args.model)
    print("  done.")
    print(f"Running condition B (fixed-5) — {len(requests_b)} requests to {args.model}...")
    outcomes_b = client.chat_many(requests_b, model=args.model)
    print("  done.")
    print(f"Running condition C (free-form) — {len(requests_c)} requests to {args.model}...")
    outcomes_c = client.chat_many(requests_c, model=args.model)
    print("  done.")

    # ── Score each chunk ──────────────────────────────────────────────────────
    records: list[dict] = []
    scores_a: list[dict] = []
    scores_b: list[dict] = []
    scores_c: list[dict] = []
    counts_a: list[int] = []
    counts_b: list[int] = []
    counts_c: list[int] = []
    ref_counts: list[int] = []

    def _parse_safe(outcome: object) -> list[str]:
        if isinstance(outcome, Exception):
            return []
        return _parse_keywords(str(outcome))

    for i, doc in enumerate(docs):
        text = doc["text"]
        reference: list[str] = doc["keywords"]
        word_count = doc.get("word_count", len(text.split()))

        kws_a = _parse_safe(outcomes_a[i])
        kws_b = _parse_safe(outcomes_b[i])
        kws_c = _parse_safe(outcomes_c[i])

        m_a = _metrics(kws_a, reference)
        m_b = _metrics(kws_b, reference)
        m_c = _metrics(kws_c, reference)

        scores_a.append(m_a)
        scores_b.append(m_b)
        scores_c.append(m_c)
        counts_a.append(len(kws_a))
        counts_b.append(len(kws_b))
        counts_c.append(len(kws_c))
        ref_counts.append(len(reference))

        if args.verbose:
            print(
                f"  chunk[{i:3d}] id={doc['chunk_id']} words={word_count:4d} ref={len(reference):2d} "
                f"| A: n={top_ns_a[i]:2d} ext={len(kws_a):2d} f1={m_a['f1']:.3f} "
                f"| B: ext={len(kws_b):2d} f1={m_b['f1']:.3f} "
                f"| C: ext={len(kws_c):2d} f1={m_c['f1']:.3f}"
            )

        records.append({
            "chunk_id": doc["chunk_id"],
            "section": doc.get("section", ""),
            "word_count": word_count,
            "reference_count": len(reference),
            "A": {"top_n": top_ns_a[i], "extracted_count": len(kws_a), **m_a},
            "B": {"top_n": 5, "extracted_count": len(kws_b), **m_b},
            "C": {"top_n": None, "extracted_count": len(kws_c), **m_c},
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(records)
    mean_ref = round(statistics.mean(ref_counts), 2)
    std_ref = round(statistics.stdev(ref_counts) if n > 1 else 0.0, 2)
    var_ref = round(statistics.variance(ref_counts) if n > 1 else 0.0, 2)

    def _cond_summary(scores: list[dict], counts: list[int]) -> dict:
        m = len(counts)
        result: dict = {}
        for metric in ("precision", "recall", "f1"):
            result.update(_aggregate(scores, metric))
        result["mean_extracted_count"] = round(statistics.mean(counts), 2)
        result["std_extracted_count"] = round(statistics.stdev(counts) if m > 1 else 0.0, 2)
        result["var_extracted_count"] = round(statistics.variance(counts) if m > 1 else 0.0, 2)
        return result

    conditions = {
        "A_adaptive": _cond_summary(scores_a, counts_a),
        "B_fixed5": _cond_summary(scores_b, counts_b),
        "C_freeform": _cond_summary(scores_c, counts_c),
    }

    # ── Length bucket breakdown ───────────────────────────────────────────────
    sorted_wc = sorted(r["word_count"] for r in records)
    t33 = sorted_wc[len(sorted_wc) // 3]
    t66 = sorted_wc[2 * len(sorted_wc) // 3]

    def _bucket(wc: int) -> str:
        return "short" if wc <= t33 else ("medium" if wc <= t66 else "long")

    grouped: dict[str, dict] = {
        b: {"scores_a": [], "scores_b": [], "scores_c": [],
            "counts_a": [], "counts_b": [], "counts_c": [], "ref_counts": []}
        for b in ("short", "medium", "long")
    }
    for r in records:
        g = grouped[_bucket(r["word_count"])]
        for cond, key in (("A", "scores_a"), ("B", "scores_b"), ("C", "scores_c")):
            g[key].append({k: r[cond][k] for k in ("precision", "recall", "f1")})
        g["counts_a"].append(r["A"]["extracted_count"])
        g["counts_b"].append(r["B"]["extracted_count"])
        g["counts_c"].append(r["C"]["extracted_count"])
        g["ref_counts"].append(r["reference_count"])

    by_length: dict[str, dict] = {}
    for bname, g in grouped.items():
        bn = len(g["ref_counts"])
        by_length[bname] = {
            "n": bn,
            "mean_reference_count": round(statistics.mean(g["ref_counts"]), 2),
            "var_reference_count": round(statistics.variance(g["ref_counts"]) if bn > 1 else 0.0, 2),
            "A_adaptive": _cond_summary(g["scores_a"], g["counts_a"]),
            "B_fixed5":   _cond_summary(g["scores_b"], g["counts_b"]),
            "C_freeform":  _cond_summary(g["scores_c"], g["counts_c"]),
        }

    # ── Print tables ──────────────────────────────────────────────────────────
    W = 72

    def _print_cond_table(cond_dict: dict[str, dict]) -> None:
        print(f"{'Condition':<18} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Ext μ':>7} {'Ext σ':>7} {'Ext σ²':>7}")
        print("-" * W)
        for cname, s in cond_dict.items():
            print(
                f"{cname:<18} "
                f"{s['mean_precision']:>7.4f} "
                f"{s['mean_recall']:>7.4f} "
                f"{s['mean_f1']:>7.4f} "
                f"{s['mean_extracted_count']:>7.1f} "
                f"{s['std_extracted_count']:>7.2f} "
                f"{s['var_extracted_count']:>7.2f}"
            )

    print(f"\n{'=' * W}")
    print(f"A2b Results  (n={n}, model={args.model})")
    print(f"Reference keyword count — μ={mean_ref:.1f}  σ={std_ref:.2f}  σ²={var_ref:.2f}")
    print(f"{'=' * W}")
    _print_cond_table(conditions)
    print("=" * W)
    best = max(conditions, key=lambda k: conditions[k]["mean_f1"])
    print(f"Best F1: {best}  ({conditions[best]['mean_f1']:.4f})")

    print(f"\n{'─' * W}")
    print(f"By chunk length  (tertile thresholds: short ≤ {t33}w, medium ≤ {t66}w, long > {t66}w)")
    for bname, bs in by_length.items():
        print(f"\n  [{bname.upper()}]  n={bs['n']}  ref μ={bs['mean_reference_count']:.1f}  ref σ²={bs['var_reference_count']:.2f}")
        _print_cond_table({k: bs[k] for k in ("A_adaptive", "B_fixed5", "C_freeform")})
    print("─" * W)

    # ── Write output ──────────────────────────────────────────────────────────
    summary = {
        "config": {
            "model": args.model,
            "n": n,
            "annotated_chunks": str(ann_path),
        },
        "reference_count": {"mean": mean_ref, "std": std_ref, "var": var_ref},
        "conditions": conditions,
        "by_length": {"thresholds": {"t33": t33, "t66": t66}, **by_length},
        "records": records,
    }

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    load_dotenv()
    main()
