"""A2 — Adaptive top-n approximates gold keyword count better than fixed top-n.

Two modes:

  annotated (default): load annotated_chunks.json produced by annotate_chunks.py
    and use the gold keyword count as the reference. No LLM call required.

  freeform (--chunks-pkl): sample raw chunks, call the LLM with no top-n cap,
    and use the resulting count as the reference. Kept for backward compatibility.

Metric: MAE between predicted count (adaptive / fixed) and actual count.

Usage:
    python -m src.knowledge_graph.scripts.experiment_a2_keyword_count \\
        --annotated-chunks annotated_chunks.json \\
        --output results_a2.json

    python -m src.knowledge_graph.scripts.experiment_a2_keyword_count \\
        --chunks-pkl index/sections/textbook_index_chunks.pkl \\
        --sample 100 \\
        --model google/gemini-3-flash-preview \\
        --output results_a2.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
from pathlib import Path

from dotenv import load_dotenv


def _parse_keywords(content: str) -> list[str]:
    try:
        kws = json.loads(content)
        if isinstance(kws, list):
            return kws
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*?\]", content, re.DOTALL)
    if match:
        try:
            kws = json.loads(match.group(0))
            if isinstance(kws, list):
                return kws
        except json.JSONDecodeError:
            pass
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A2: Compare adaptive vs fixed top-n to gold keyword count.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotated-chunks",
        default="annotated_chunks.json",
        help="JSON produced by annotate_chunks.py (primary mode)",
    )
    parser.add_argument(
        "--chunks-pkl",
        default=None,
        help="Fallback: path to chunks pickle; triggers freeform LLM counting mode",
    )
    parser.add_argument("--sample", type=int, default=100, help="Chunks to sample (pkl mode only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        default="google/gemini-3-flash-preview",
        help="OpenRouter model (pkl / freeform mode only)",
    )
    parser.add_argument("--fixed-k", type=int, default=10, help="Fixed top-n baseline")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None, help="Write results JSON here")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    root = Path(__file__).parent.parent.parent.parent

    records = []

    if args.chunks_pkl is not None:
        # ── Freeform LLM mode (backward-compatible) ───────────────────────────
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("No OpenRouter API key. Set OPENROUTER_API_KEY or pass --api-key.")

        from src.knowledge_graph.openrouter_client import OpenRouterClient

        client = OpenRouterClient(api_key, retries=2)

        chunks_path = Path(args.chunks_pkl)
        if not chunks_path.is_absolute():
            chunks_path = root / chunks_path

        with open(chunks_path, "rb") as f:
            all_chunks: list[str] = pickle.load(f)

        print(f"Loaded {len(all_chunks)} chunks. Sampling {args.sample}...")
        random.seed(args.seed)
        indices = random.sample(range(len(all_chunks)), min(args.sample, len(all_chunks)))
        sampled = [(i, all_chunks[i]) for i in indices]

        FREE_FORM_PROMPT = (
            "You are a linguistic analysis expert. Analyze the provided text and identify "
            "the most relevant and descriptive keywords or short phrases (1-3 words). "
            "Focus on technical terms, proper nouns, and central concepts. "
            "Return the result as a raw JSON list of strings. "
            "Do not include any other text or explanation."
        )
        requests_ = [
            {
                "messages": [
                    {"role": "system", "content": FREE_FORM_PROMPT},
                    {"role": "user", "content": f"Documents: {text}"},
                ]
            }
            for _, text in sampled
        ]
        print(f"Sending {len(requests_)} requests to {args.model}...")
        outcomes = client.chat_many(requests_, model=args.model)

        for (chunk_idx, text), outcome in zip(sampled, outcomes):
            word_count = len(text.split())
            adaptive = math.ceil(math.sqrt(word_count))
            if isinstance(outcome, Exception):
                print(f"  Chunk {chunk_idx}: FAILED — {outcome}")
                actual = None
            else:
                kws = _parse_keywords(outcome)
                actual = len(kws)
                if args.verbose:
                    print(f"  Chunk {chunk_idx}: words={word_count}, actual={actual}, "
                          f"adaptive={adaptive}, fixed={args.fixed_k}")
            records.append({
                "chunk_idx": chunk_idx,
                "word_count": word_count,
                "actual_count": actual,
                "adaptive_pred": adaptive,
                "fixed_pred": args.fixed_k,
            })

    else:
        # ── Annotated chunks mode (primary) ───────────────────────────────────
        ann_path = Path(args.annotated_chunks)
        if not ann_path.is_absolute():
            ann_path = root / ann_path

        with open(ann_path) as f:
            ann_data = json.load(f)

        raw = [r for r in ann_data["records"] if r.get("keywords") is not None]
        print(f"Loaded {len(raw)} annotated chunks from {ann_path}.")

        for rec in raw:
            word_count = rec.get("word_count", len(rec["text"].split()))
            adaptive = math.ceil(math.sqrt(word_count))
            actual = len(rec["keywords"])
            if args.verbose:
                print(f"  chunk_id={rec['chunk_id']}: words={word_count}, "
                      f"gold={actual}, adaptive={adaptive}, fixed={args.fixed_k}")
            records.append({
                "chunk_id": rec["chunk_id"],
                "word_count": word_count,
                "actual_count": actual,
                "adaptive_pred": adaptive,
                "fixed_pred": args.fixed_k,
            })

    # Compute MAE on successful extractions
    valid = [r for r in records if r["actual_count"] is not None]
    n = len(valid)
    if n == 0:
        print("No valid extractions — cannot compute metrics.")
        return

    mae_adaptive = sum(abs(r["adaptive_pred"] - r["actual_count"]) for r in valid) / n
    mae_fixed = sum(abs(r["fixed_pred"] - r["actual_count"]) for r in valid) / n
    mean_actual = sum(r["actual_count"] for r in valid) / n
    mean_adaptive = sum(r["adaptive_pred"] for r in valid) / n

    print(f"\n{'=' * 50}")
    print(f"A2 Results (n={n} successful extractions)")
    print(f"{'=' * 50}")
    print(f"  Mean actual gold keyword count : {mean_actual:.2f}")
    print(f"  Mean adaptive prediction      : {mean_adaptive:.2f}  (ceil(sqrt(words)))")
    print(f"  Fixed k                        : {args.fixed_k}")
    print(f"  MAE — adaptive                : {mae_adaptive:.4f}")
    print(f"  MAE — fixed (k={args.fixed_k})          : {mae_fixed:.4f}")
    better = "adaptive" if mae_adaptive < mae_fixed else "fixed"
    print(f"  Winner                         : {better}")
    print(f"{'=' * 50}")

    summary = {
        "n": n,
        "mean_actual_count": round(mean_actual, 4),
        "mean_adaptive_pred": round(mean_adaptive, 4),
        "fixed_k": args.fixed_k,
        "mae_adaptive": round(mae_adaptive, 4),
        "mae_fixed": round(mae_fixed, 4),
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
