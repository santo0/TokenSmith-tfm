#!/usr/bin/env python3
"""Benchmark LLM keyword extraction across multiple OpenRouter models.

Metrics
-------
- Latency  : per-chunk wall time (mean / p50 / p95), total elapsed
- Cost     : estimated from token usage × OpenRouter published pricing
- Accuracy : Jaccard similarity vs. Claude Opus (SOTA reference)
"""
import argparse
import json
import logging
import os
import random
import re
import statistics
from math import sqrt
from time import strftime

import requests
from dotenv import load_dotenv

from src.knowledge_graph.build import CHUNKS_PKL, META_PKL, get_index_paths, load_chunks
from src.knowledge_graph.models import Chunk
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.prompts import OPENROUTER_KEYWORD_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

_DEFAULT_SOTA_MODEL = "anthropic/claude-opus-4.7"


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def fetch_model_pricing(api_key: str) -> dict[str, dict[str, float]]:
    """Return ``{model_id: {"prompt": $/M_tok, "completion": $/M_tok}}``."""
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        pricing: dict[str, dict[str, float]] = {}
        for entry in resp.json().get("data", []):
            mid = entry.get("id", "")
            p = entry.get("pricing", {})
            try:
                pricing[mid] = {
                    "prompt": float(p.get("prompt", 0)) * 1_000_000,
                    "completion": float(p.get("completion", 0)) * 1_000_000,
                }
            except (TypeError, ValueError):
                pass
        logger.info("Fetched pricing for %d models.", len(pricing))
        return pricing
    except Exception as e:
        logger.warning(
            "Could not fetch live pricing (%s) — cost will be None.", e)
        return {}


def estimate_cost(usage: dict, pricing: dict, model: str) -> float | None:
    p = pricing.get(model)
    if p is None:
        return None
    return (
        usage.get("prompt_tokens", 0) * p["prompt"] / 1_000_000
        + usage.get("completion_tokens", 0) * p["completion"] / 1_000_000
    )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _build_messages(chunk: Chunk, top_n: int) -> list[dict]:
    return [
        {
            "role": "system",
            "content": OPENROUTER_KEYWORD_EXTRACTION_PROMPT.format(top_n=top_n),
        },
        {"role": "user", "content": f"Documents: {chunk.text}"},
    ]


def _parse_keywords(content: str) -> list[str]:
    try:
        kws = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            kws = json.loads(match.group(0))
        else:
            raise ValueError(f"Cannot parse JSON list from: {content!r}")
    if not isinstance(kws, list):
        raise ValueError(f"Response is not a list: {kws!r}")
    return [str(k) for k in kws]


def _top_n_for(chunk: Chunk, adaptive: bool, base_top_n: int) -> int:
    if adaptive:
        return max(1, int(sqrt(len(chunk.text.split()))))
    return base_top_n


# ---------------------------------------------------------------------------
# Per-model extraction
# ---------------------------------------------------------------------------


def run_extraction(
    client: OpenRouterClient,
    model: str,
    chunks: list[Chunk],
    top_n: int,
    adaptive: bool,
    timeout: int,
) -> list[dict]:
    """Run *model* on *chunks* concurrently; return per-chunk result dicts."""
    reqs = [
        {"messages": _build_messages(c, _top_n_for(c, adaptive, top_n))}
        for c in chunks
    ]
    raw = client.chat_many_with_meta(reqs, model=model, timeout=timeout)

    results = []
    for chunk, (content_or_exc, meta, latency) in zip(chunks, raw):
        usage = (meta or {}).get("usage", {})
        if isinstance(content_or_exc, Exception):
            logger.error("[%s] chunk %d failed: %s",
                         model, chunk.id, content_or_exc)
            results.append(
                {
                    "chunk_id": chunk.id,
                    "keywords": [],
                    "latency_s": latency,
                    "usage": usage,
                    "error": str(content_or_exc),
                }
            )
        else:
            try:
                keywords = _parse_keywords(content_or_exc)
            except Exception as e:
                logger.error("[%s] chunk %d parse error: %s",
                             model, chunk.id, e)
                keywords = []
            results.append(
                {
                    "chunk_id": chunk.id,
                    "keywords": keywords,
                    "latency_s": latency,
                    "usage": usage,
                    "error": None,
                }
            )
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def jaccard(a: list[str], b: list[str]) -> float:
    sa = {k.strip().lower() for k in a if k.strip()}
    sb = {k.strip().lower() for k in b if k.strip()}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def compute_jaccard(
    candidate_results: list[dict],
    reference: dict[int, list[str]],
) -> dict:
    scores = [
        jaccard(r["keywords"], reference[r["chunk_id"]])
        for r in candidate_results
        if reference.get(r["chunk_id"])
    ]
    if not scores:
        return {"mean": None, "median": None, "std": None, "n": 0}
    return {
        "mean": round(statistics.mean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "std": round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 4),
        "n": len(scores),
    }


def aggregate_latency(results: list[dict]) -> dict:
    lats = sorted(r["latency_s"] for r in results)
    if not lats:
        return {}
    n = len(lats)
    return {
        "mean_s": round(statistics.mean(lats), 3),
        "median_s": round(statistics.median(lats), 3),
        "p95_s": round(lats[int(n * 0.95)], 3),
        "total_s": round(sum(lats), 3),
    }


def aggregate_cost(results: list[dict], pricing: dict, model: str) -> dict:
    costs, total_prompt, total_completion = [], 0, 0
    for r in results:
        usage = r.get("usage", {})
        total_prompt += usage.get("prompt_tokens", 0)
        total_completion += usage.get("completion_tokens", 0)
        c = estimate_cost(usage, pricing, model)
        if c is not None:
            costs.append(c)
    return {
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "estimated_total_usd": round(sum(costs), 6) if costs else None,
        "estimated_per_chunk_usd": round(statistics.mean(costs), 6) if costs else None,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _fmt(v, kind: str) -> str:
    if v is None:
        return "—"
    if kind == "f":
        return f"{v:.3f}"
    if kind == "$":
        return f"${v:.4f}"
    return str(v)


def print_summary(models: list[str], model_stats: dict[str, dict]) -> None:
    col_m = max(30, max((len(m) for m in models), default=0) + 2)
    col_v = 12

    header = (
        f"{'Model':<{col_m}}"
        f"  {'latency(p50)':>{col_v}}"
        f"  {'latency(p95)':>{col_v}}"
        f"  {'cost(USD)':>{col_v}}"
        f"  {'tokens':>{col_v}}"
        f"  {'jaccard':>{col_v}}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for model in models:
        stats = model_stats.get(model, {})
        lat = stats.get("latency", {})
        cost = stats.get("cost", {})
        j = stats.get("jaccard", {})
        print(
            f"{model:<{col_m}}"
            f"  {_fmt(lat.get('median_s'), 'f'):>{col_v}}"
            f"  {_fmt(lat.get('p95_s'), 'f'):>{col_v}}"
            f"  {_fmt(cost.get('estimated_total_usd'), '$'):>{col_v}}"
            f"  {_fmt(cost.get('total_tokens'), 'i'):>{col_v}}"
            f"  {_fmt(j.get('mean'), 'f'):>{col_v}}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(args: argparse.Namespace) -> dict:
    all_chunks = load_chunks(args.chunks_path, args.meta_path)
    chunks = random.Random(args.seed).sample(
        all_chunks, min(args.n_chunks, len(all_chunks)))
    logger.info("Sampled %d / %d chunks (seed=%d)",
                len(chunks), len(all_chunks), args.seed)

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "No API key — pass --api-key or set OPENROUTER_API_KEY.")

    client = OpenRouterClient(
        api_key, retries=args.retries, max_workers=args.max_workers)
    pricing = fetch_model_pricing(api_key)

    logger.info("Running SOTA reference: %s", args.sota_model)
    sota_results = run_extraction(
        client, args.sota_model, chunks, args.top_n, args.adaptive_top_n, args.timeout
    )
    reference: dict[int, list[str]] = {
        r["chunk_id"]: r["keywords"] for r in sota_results}

    model_stats: dict[str, dict] = {}
    model_chunk_results: dict[str, list[dict]] = {}

    for model in args.models:
        logger.info("Benchmarking: %s", model)
        results = run_extraction(
            client, model, chunks, args.top_n, args.adaptive_top_n, args.timeout
        )
        model_chunk_results[model] = results
        model_stats[model] = {
            "latency": aggregate_latency(results),
            "cost": aggregate_cost(results, pricing, model),
            "jaccard": compute_jaccard(results, reference),
        }

    return {
        "config": {
            "n_chunks": len(chunks),
            "top_n": args.top_n,
            "adaptive_top_n": args.adaptive_top_n,
            "seed": args.seed,
            "retries": args.retries,
            "max_workers": args.max_workers,
            "timeout": args.timeout,
            "sota_model": args.sota_model,
            "candidate_models": args.models,
        },
        "candidate_stats": model_stats,
        "per_chunk": model_chunk_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark LLM keyword extraction against a Claude Opus reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["google/gemini-3.1-flash-lite-preview",
                 "openai/gpt-oss-120b", "deepseek/deepseek-v3.2"],
        metavar="MODEL",
        help="Candidate models to benchmark.",
    )
    parser.add_argument(
        "--sota-model",
        default=_DEFAULT_SOTA_MODEL,
        metavar="MODEL",
        help="Reference model used to build ground-truth keywords.",
    )
    parser.add_argument("--n-chunks", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--adaptive-top-n", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--chunks-path", default=None)
    parser.add_argument("--meta-path", default=None)
    parser.add_argument(
        "--partial",
        action="store_true",
        default=False,
        help="Use the partial index (index/partial_sections/) instead of the full index",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    default_chunks, default_meta = get_index_paths(args.partial)
    if args.chunks_path is None:
        args.chunks_path = default_chunks
    if args.meta_path is None:
        args.meta_path = default_meta
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results = run_benchmark(args)
    print_summary(results["config"]["candidate_models"],
                  results["candidate_stats"])

    out_path = args.output or os.path.join(
        "data", f"benchmark_llm_{strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    load_dotenv()
    main()
