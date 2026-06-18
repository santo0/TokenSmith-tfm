"""A1 — LLM extracts more useful keywords than YAKE/KeyBERT.

Loads annotated chunks (produced by annotate_chunks.py) and compares three
extraction families — LLM, KeyBERT, YAKE — against the gold keyword annotations.

Metrics (macro-averaged over chunks):
  precision, recall, F1 vs gold keywords
  hallucination rate: keywords absent from the source text

Usage:
    python -m src.knowledge_graph.scripts.experiment_a1_extractor_quality \\
        --annotated-chunks annotated_chunks.json \\
        --llm-model google/gemini-2.5-flash \\
        --output results_a1.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from pathlib import Path

from dotenv import load_dotenv


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


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _hallucination_rate(keywords: list[str], source_text: str) -> float:
    if not keywords:
        return 0.0
    source_norm = source_text.lower()
    hallucinated = sum(
        1 for kw in keywords
        if re.sub(r"[^\w\s]", "", kw.lower()).strip() not in source_norm
    )
    return hallucinated / len(keywords)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A1: Compare LLM / KeyBERT / YAKE keyword extraction quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotated-chunks",
        default="annotated_chunks.json",
        help="JSON produced by annotate_chunks.py",
    )
    parser.add_argument("--llm-model", default="google/gemini-2.5-flash-preview")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")

    root = Path(__file__).parent.parent.parent.parent
    ann_path = Path(args.annotated_chunks)
    if not ann_path.is_absolute():
        ann_path = root / ann_path

    with open(ann_path) as f:
        ann_data = json.load(f)

    records = [r for r in ann_data["records"] if r.get("keywords") is not None]
    print(f"Loaded {len(records)} annotated chunks from {ann_path}.")

    from src.knowledge_graph.models import Chunk
    from src.knowledge_graph.normalizer import Normalizer

    normalizer = Normalizer()

    # Build extractors
    llm_extractor = None
    if api_key:
        from src.knowledge_graph.extractors.openrouter_extractor import OpenRouterExtractor
        llm_extractor = OpenRouterExtractor(
            api_key=api_key, model=args.llm_model, top_n=args.top_n, adaptive_top_n=False
        )
    else:
        print("No API key — LLM extractor disabled.")

    kb_extractor = None
    try:
        from src.knowledge_graph.extractors.keybert_extractor import KeyBERTExtractor
        kb_extractor = KeyBERTExtractor(top_n=args.top_n)
    except Exception as e:
        print(f"KeyBERT not available: {e}")

    yake_extractor = None
    try:
        from src.knowledge_graph.extractors.yake_extractor import YakeExtractor
        yake_extractor = YakeExtractor(top_n=args.top_n)
    except Exception as e:
        print(f"YAKE not available: {e}")

    extractor_names = [
        ("llm", llm_extractor),
        ("keybert", kb_extractor),
        ("yake", yake_extractor),
    ]

    per_chunk: list[dict] = []

    for rec in records:
        chunk_id = rec["chunk_id"]
        text = rec["text"]
        gold_raw: list[str] = rec["keywords"]
        gold_norm = {_normalize(k) for k in gold_raw}

        chunk_obj = Chunk(id=chunk_id, text=text, metadata={})
        row: dict = {"chunk_id": chunk_id, "section": rec.get("section", "")}

        for name, extractor in extractor_names:
            if extractor is None:
                continue
            try:
                results = extractor.extract([chunk_obj])
                all_kws: list[str] = []
                for r in results:
                    all_kws.extend(r.keywords)
                pred_norm = {_normalize(k) for k in normalizer.normalize(all_kws)}
            except Exception as e:
                print(f"  chunk_id={chunk_id} [{name}] failed: {e}")
                pred_norm = set()
                all_kws = []

            p, r, f1 = _prf(pred_norm, gold_norm)
            hall = _hallucination_rate(all_kws, text)

            row[name] = {
                "P": round(p, 4),
                "R": round(r, 4),
                "F1": round(f1, 4),
                "hallucination_rate": round(hall, 4),
                "extracted_count": len(all_kws),
            }

            if args.verbose:
                print(
                    f"  chunk_id={chunk_id} [{name:8s}]: "
                    f"P={p:.3f} R={r:.3f} F1={f1:.3f} hall={hall:.1%}"
                )

        per_chunk.append(row)

    # Aggregate
    def _agg(metric: str, extractor: str) -> dict:
        vals = [r[extractor][metric] for r in per_chunk if extractor in r]
        if not vals:
            return {}
        return {
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
        }

    summary: dict = {}
    W = 70
    print(f"\n{'=' * W}")
    print(f"A1 Results  (n={len(per_chunk)}, top_n={args.top_n})")
    print(f"{'=' * W}")
    header = f"{'Extractor':<12} {'Mean P':>8} {'Mean R':>8} {'Mean F1':>8} {'Hall %':>8}"
    print(header)
    print("-" * W)

    for name, extractor in extractor_names:
        if extractor is None or not any(name in r for r in per_chunk):
            continue
        agg_p = _agg("P", name)
        agg_r = _agg("R", name)
        agg_f1 = _agg("F1", name)
        agg_h = _agg("hallucination_rate", name)
        print(
            f"{name:<12} "
            f"{agg_p['mean']:>8.4f} "
            f"{agg_r['mean']:>8.4f} "
            f"{agg_f1['mean']:>8.4f} "
            f"{agg_h['mean']:>7.1%}"
        )
        summary[name] = {
            "precision": agg_p,
            "recall": agg_r,
            "f1": agg_f1,
            "hallucination_rate": agg_h,
        }

    print("=" * W)

    result = {
        "config": {
            "annotated_chunks": str(ann_path),
            "llm_model": args.llm_model,
            "top_n": args.top_n,
            "n_chunks": len(per_chunk),
        },
        "summary": summary,
        "per_chunk": per_chunk,
    }

    if args.output:
        out = Path(args.output)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {out}")


if __name__ == "__main__":
    load_dotenv()
    main()
