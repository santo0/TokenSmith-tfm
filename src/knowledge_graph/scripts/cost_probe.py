"""
cost_probe.py — measure real API token spend on N chunks, extrapolate to full corpus.

Runs the three LLM-heavy offline stages (keyword extraction, canonicalization,
summary tree) on a sample of chunks using actual OpenRouter API calls.  Token
usage is read from the response metadata and used to compute real cost, then
scaled to the full corpus.

Canonicalization scales with the number of unique keywords, not chunks
(vocabulary growth is sub-linear), so its scale factor is derived from
the full extractions file when available.

Usage:
    python -m src.knowledge_graph.scripts.cost_probe [--n-chunks 50] [--offset 0]
"""
import argparse
import json
import logging
import os
import sys
import tempfile
import time

import networkx as nx
from dotenv import load_dotenv

from src.knowledge_graph.build import load_chunks, get_index_paths
from src.knowledge_graph.canonicalizer import Canonicalizer
from src.knowledge_graph.extractors import OpenRouterExtractor
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.section_tree import build_section_tree
from src.knowledge_graph.summary_tree import build_summary_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL              = "openai/gpt-4o-mini"
PRICE_INPUT_PER_M  = 0.50   # USD per million input tokens
PRICE_OUTPUT_PER_M = 3.00   # USD per million output tokens
EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
CORPUS_DESCRIPTION = "Database System Concepts, 7th edition by Silberschatz et al."
FULL_EXTRACTIONS   = "data/knowledge_graph/runs/latest/input/extractions.json"


# ── Token-tracking client ─────────────────────────────────────────────────────

class TrackingClient(OpenRouterClient):
    """OpenRouterClient that accumulates per-stage token usage from response metadata."""

    def __init__(self, api_key: str, **kwargs) -> None:
        super().__init__(api_key, **kwargs)
        self._stage = "unknown"
        self._usage: dict[str, dict] = {}

    def set_stage(self, name: str) -> None:
        self._stage = name
        self._usage.setdefault(name, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})

    def chat_with_meta(self, model, messages, response_format=None, timeout=60):
        content, meta = super().chat_with_meta(model, messages, response_format, timeout)
        s = self._usage.setdefault(
            self._stage, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        )
        s["calls"] += 1
        if meta:
            u = meta.get("usage", {})
            s["prompt_tokens"]     += u.get("prompt_tokens", 0)
            s["completion_tokens"] += u.get("completion_tokens", 0)
        return content, meta

    def stage_usage(self) -> dict[str, dict]:
        return dict(self._usage)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens     * PRICE_INPUT_PER_M  / 1_000_000
        + completion_tokens * PRICE_OUTPUT_PER_M / 1_000_000
    )


def _full_unique_kw_count() -> int:
    """Unique keyword count from the pre-existing full extractions, if available."""
    if not os.path.exists(FULL_EXTRACTIONS):
        return 0
    with open(FULL_EXTRACTIONS) as f:
        data = json.load(f)
    return len({kw.strip().lower() for e in data for kw in e["keywords"]})


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Measure real API cost on N chunks and extrapolate to full corpus."
    )
    parser.add_argument("--n-chunks", type=int, default=50,
                        help="Number of chunks to probe (default: 50)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Starting chunk index (default: 0)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set")

    client = TrackingClient(api_key)

    # ── Load chunks ───────────────────────────────────────────────────────────
    chunks_pkl, meta_pkl = get_index_paths(partial=False)
    all_chunks = load_chunks(chunks_pkl, meta_pkl)
    total_chunks = len(all_chunks)
    sample = all_chunks[args.offset : args.offset + args.n_chunks]
    logger.info("Sampled %d / %d chunks (offset=%d)", len(sample), total_chunks, args.offset)

    # ── Stage 1: Keyword extraction ───────────────────────────────────────────
    logger.info("=== Stage 1: keyword extraction (%d chunks) ===", len(sample))
    client.set_stage("extraction")
    extractor = OpenRouterExtractor(api_key=api_key, model=MODEL, adaptive_top_n=True)
    extractor._client = client  # inject tracking client
    t0 = time.perf_counter()
    extractions = extractor.extract(sample)
    logger.info("  Done in %.1fs", time.perf_counter() - t0)

    sample_unique_kw = len({kw.strip().lower() for e in extractions for kw in e.keywords})
    logger.info("  Unique keywords in sample: %d", sample_unique_kw)

    # ── Stage 2: Canonicalization ─────────────────────────────────────────────
    logger.info("=== Stage 2: canonicalization (%d unique keywords) ===", sample_unique_kw)
    client.set_stage("canonicalization")
    canonicalizer = Canonicalizer(
        corpus_description=CORPUS_DESCRIPTION,
        api_key=api_key,
        embedding_model=EMBED_MODEL,
        llm_model=MODEL,
    )
    canonicalizer._client = client  # inject tracking client
    t0 = time.perf_counter()
    extractions, _canon = canonicalizer.canonicalize(extractions)
    logger.info("  Done in %.1fs", time.perf_counter() - t0)

    # ── Stage 3: Summary tree ─────────────────────────────────────────────────
    logger.info("=== Stage 3: summary tree ===")
    client.set_stage("summarization")
    section_tree = build_section_tree(sample, nx.Graph())
    chunk_texts = {c.id: c.text for c in sample}
    with tempfile.TemporaryDirectory() as tmp_dir:
        t0 = time.perf_counter()
        build_summary_index(
            client=client,
            summary_model=MODEL,
            section_tree=section_tree,
            chunks=chunk_texts,
            embed_model=EMBED_MODEL,
            chunk_window=3,
            run_dir=tmp_dir,
        )
        logger.info("  Done in %.1fs", time.perf_counter() - t0)

    # ── Scale factors ─────────────────────────────────────────────────────────
    usage = client.stage_usage()
    chunk_scale = total_chunks / len(sample)

    full_unique_kw = _full_unique_kw_count()
    if full_unique_kw and sample_unique_kw:
        canon_scale = full_unique_kw / sample_unique_kw
        canon_scale_note = f"vocabulary scale ({sample_unique_kw} → {full_unique_kw} unique kw)"
    else:
        canon_scale = chunk_scale
        canon_scale_note = f"chunk scale (full extractions not found at {FULL_EXTRACTIONS})"
        logger.warning("Full extractions not found; using chunk scale for canonicalization")

    scales = {
        "extraction":     (chunk_scale, f"chunk scale ({len(sample)} → {total_chunks})"),
        "canonicalization": (canon_scale, canon_scale_note),
        "summarization":  (chunk_scale, f"chunk scale ({len(sample)} → {total_chunks})"),
    }

    # ── Report ────────────────────────────────────────────────────────────────
    W = 78
    print(f"\n{'═' * W}")
    print(f"  Cost probe  │  {len(sample)} / {total_chunks} chunks  │  {MODEL}")
    print(f"  Pricing     │  ${PRICE_INPUT_PER_M}/M input  │  ${PRICE_OUTPUT_PER_M}/M output")
    print(f"{'═' * W}")

    print(f"\n  {'Stage':<24} {'Calls':>6}  {'Input tok':>11}  {'Output tok':>11}  {'Cost':>10}")
    print(f"  {'─' * 66}")
    sample_cost = 0.0
    for stage, s in usage.items():
        c = _usd(s["prompt_tokens"], s["completion_tokens"])
        sample_cost += c
        print(
            f"  {stage:<24} {s['calls']:>6,}  {s['prompt_tokens']:>11,}"
            f"  {s['completion_tokens']:>11,}  ${c:>9.4f}"
        )
    print(f"  {'─' * 66}")
    tot_in  = sum(s["prompt_tokens"]     for s in usage.values())
    tot_out = sum(s["completion_tokens"] for s in usage.values())
    tot_calls = sum(s["calls"] for s in usage.values())
    print(
        f"  {'SAMPLE TOTAL':<24} {tot_calls:>6,}  {tot_in:>11,}  {tot_out:>11,}  ${sample_cost:>9.4f}"
    )

    print(f"\n  Extrapolation → full corpus ({total_chunks} chunks):")
    print(f"  {'Stage':<24} {'Scale':>9}  {'Est. calls':>10}  {'Est. cost':>10}  Note")
    print(f"  {'─' * 70}")
    full_cost = 0.0
    for stage, s in usage.items():
        scale, note = scales[stage]
        est_cost  = _usd(s["prompt_tokens"], s["completion_tokens"]) * scale
        est_calls = round(s["calls"] * scale)
        full_cost += est_cost
        print(f"  {stage:<24} {scale:>8.1f}x  {est_calls:>10,}  ${est_cost:>9.4f}  {note}")
    print(f"  {'─' * 70}")
    print(f"  {'FULL CORPUS TOTAL':<24} {'':>9}  {'':>10}  ${full_cost:>9.4f}")
    print(f"\n{'═' * W}\n")

    # ── Persist results ───────────────────────────────────────────────────────
    result = {
        "model": MODEL,
        "price_input_per_M": PRICE_INPUT_PER_M,
        "price_output_per_M": PRICE_OUTPUT_PER_M,
        "n_sample": len(sample),
        "offset": args.offset,
        "total_chunks": total_chunks,
        "sample_unique_kw": sample_unique_kw,
        "full_unique_kw": full_unique_kw,
        "sample": {
            stage: {
                **s,
                "cost_usd": round(_usd(s["prompt_tokens"], s["completion_tokens"]), 6),
            }
            for stage, s in usage.items()
        },
        "sample_cost_usd": round(sample_cost, 6),
        "full_estimate": {
            stage: {
                "scale": scales[stage][0],
                "scale_note": scales[stage][1],
                "est_calls": round(usage[stage]["calls"] * scales[stage][0]),
                "est_cost_usd": round(
                    _usd(usage[stage]["prompt_tokens"], usage[stage]["completion_tokens"])
                    * scales[stage][0],
                    4,
                ),
            }
            for stage in usage
        },
        "full_total_cost_usd": round(full_cost, 4),
    }
    out = "results_cost_probe.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
