"""annotate_chunks — LLM gold-standard keyword annotation for textbook chunks.

Samples N chunks from the textbook index and extracts reference keywords using
a powerful LLM with no count constraint. The output serves as gold-standard
annotations in A1 and A2b experiments in place of INSPEC.

Usage:
    python -m src.knowledge_graph.scripts.annotate_chunks \\
        --sample 50 \\
        --model anthropic/claude-opus-4-8 \\
        --output annotated_chunks.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
from pathlib import Path

from dotenv import load_dotenv


_ANNOTATION_PROMPT = (
    "You are a database systems expert with deep knowledge of relational databases, "
    "transaction management, query processing, storage engines, and distributed systems. "
    "Analyze the provided text and identify the keywords and short phrases (1-3 words) "
    "that most meaningfully capture the core concepts of this passage. "
    "Extract only the terms that clearly and distinctly reflect what this text is about: "
    "technical concepts, definitions, algorithms, data structures, and central ideas. "
    "Omit generic filler terms and near-duplicate variants of the same concept. "
    "Return the result as a raw JSON list of strings. "
    "Do not include any other text or explanation in your response."
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate textbook chunks with LLM-extracted gold keywords.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifacts-dir",
        default="index/sections",
        help="Directory containing textbook_index_chunks.pkl and textbook_index_meta.pkl",
    )
    parser.add_argument("--index-prefix", default="textbook_index")
    parser.add_argument("--sample", type=int, default=50, help="Number of chunks to annotate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        default="anthropic/claude-opus-4-8",
        help="OpenRouter model used for annotation",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default="annotated_chunks.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("No OpenRouter API key. Set OPENROUTER_API_KEY or pass --api-key.")

    from src.knowledge_graph.openrouter_client import OpenRouterClient

    root = Path(__file__).parent.parent.parent.parent
    artifacts_dir = root / args.artifacts_dir
    chunks_path = artifacts_dir / f"{args.index_prefix}_chunks.pkl"
    meta_path = artifacts_dir / f"{args.index_prefix}_meta.pkl"

    with open(chunks_path, "rb") as f:
        all_chunks: list[str] = pickle.load(f)
    with open(meta_path, "rb") as f:
        all_meta: list[dict] = pickle.load(f)

    assert len(all_chunks) == len(all_meta), "chunks and metadata length mismatch"

    n = min(args.sample, len(all_chunks))
    print(f"Loaded {len(all_chunks)} chunks. Sampling {n}...")
    random.seed(args.seed)
    indices = random.sample(range(len(all_chunks)), n)

    sampled_chunks = [all_chunks[i] for i in indices]
    sampled_meta = [all_meta[i] for i in indices]

    client = OpenRouterClient(api_key, retries=3)

    requests_ = [
        {
            "messages": [
                {"role": "system", "content": _ANNOTATION_PROMPT},
                {"role": "user", "content": f"Text:\n{text}"},
            ]
        }
        for text in sampled_chunks
    ]

    print(f"Annotating {len(requests_)} chunks with {args.model}...")
    outcomes = client.chat_many(requests_, model=args.model)
    print("Done.")

    records: list[dict] = []
    n_failed = 0

    for text, meta, outcome in zip(sampled_chunks, sampled_meta, outcomes):
        chunk_id = meta["chunk_id"]
        word_count = meta.get("word_len", len(text.split()))

        if isinstance(outcome, Exception):
            print(f"  chunk_id={chunk_id}: FAILED — {outcome}")
            keywords: list[str] | None = None
            n_failed += 1
        else:
            keywords = _parse_keywords(str(outcome))
            if args.verbose:
                print(f"  chunk_id={chunk_id} [{meta['section'][:50]}]: {len(keywords)} keywords")

        records.append({
            "chunk_id": chunk_id,
            "section": meta["section"],
            "section_path": meta["section_path"],
            "word_count": word_count,
            "text": text,
            "keywords": keywords,
        })

    n_ok = len(records) - n_failed
    print(f"\n{n_ok}/{len(records)} chunks annotated successfully.")
    if n_ok > 0:
        mean_kw = sum(len(r["keywords"]) for r in records if r["keywords"] is not None) / n_ok
        print(f"Mean keywords per chunk: {mean_kw:.1f}")

    result = {
        "config": {
            "model": args.model,
            "sample": n,
            "seed": args.seed,
            "index_prefix": args.index_prefix,
        },
        "records": records,
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = root / out_path
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Annotations written to {out_path}")


if __name__ == "__main__":
    load_dotenv()
    main()
