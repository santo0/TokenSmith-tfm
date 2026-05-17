"""
tools/chunk_tagger/server.py

FastAPI backend for the Chunk Tagger tool.
Serves the single-page frontend and exposes retrieval + LLM endpoints
for annotating ideal_retrieved_chunks in tests/benchmarks.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from dotenv import load_dotenv
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── project root (two levels up from this file) ──────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# ── shared mutable state populated during lifespan startup ───────────────────
_st: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bm25_to_chunk_ids(list_idx_scores: dict) -> dict:
    """Convert BM25 list-index keys to chunk_id keys."""
    cmap = _st["chunk_id_map"]
    return {cmap[i]: s for i, s in list_idx_scores.items() if i < len(cmap)}


def _build_sections_tree(metadata: list) -> dict:
    """
    Build {chapter: {parent_section: {leaf_section: [chunk_ids]}}} from metadata.

    section_path format: "Chapter X Section A.B Title [Section A.B.C Sub-title]"
    Split on Section boundaries using a lookahead so section titles with
    uppercase letters (e.g. "Database-System") are parsed correctly.
    """
    tree: dict = {}

    for m in metadata:
        path = m["section_path"]
        cid = m["chunk_id"]

        # Split at every "Section X.Y" boundary; chapter is always the first part
        parts = re.split(r"(?=\bSection\s+[\d])", path)
        chapter = parts[0].strip() or "Unknown"
        sec_parts = [p.strip() for p in parts[1:] if p.strip()]

        if not sec_parts:
            parent = leaf = chapter
        elif len(sec_parts) == 1:
            parent = leaf = sec_parts[0]
        else:
            parent = sec_parts[0]
            leaf = sec_parts[-1]

        tree.setdefault(chapter, {})
        tree[chapter].setdefault(parent, {})
        tree[chapter][parent].setdefault(leaf, [])
        if cid not in tree[chapter][parent][leaf]:
            tree[chapter][parent][leaf].append(cid)

    return tree


def _collect_section_titles(metadata: list) -> list[str]:
    """Return sorted unique section title strings for the LLM hint prompt."""
    return sorted({m["section"] for m in metadata})


def _call_llm(prompt: str, json_mode: bool = False) -> str:
    """Call OpenRouter LLM. Raises RuntimeError if no key configured."""
    key = _st.get("openrouter_key")
    if not key:
        raise RuntimeError("No OPENROUTER_API_KEY configured")
    from src.knowledge_graph.openrouter_client import OpenRouterClient
    client = OpenRouterClient(key)
    fmt = {"type": "json_object"} if json_mode else None
    return client.chat(
        "google/gemini-2.0-flash-lite-001",
        [{"role": "user", "content": prompt}],
        response_format=fmt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: load all artifacts on startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    from src.retriever import load_artifacts, FAISSRetriever, BM25Retriever
    from src.config import RAGConfig

    cfg = RAGConfig.from_yaml(ROOT / "config" / "config.yaml")
    artifacts_dir = ROOT / "index" / "sections"

    faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(
        artifacts_dir, "textbook_index"
    )

    chunk_id_map = [m["chunk_id"] for m in metadata]
    chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_id_map)}

    chunk_info = {
        m["chunk_id"]: {
            "section": m["section"],
            "section_path": m["section_path"],
            "text": chunks[i],
            "page_numbers": m.get("page_numbers", []),
            "text_preview": m.get("text_preview", ""),
        }
        for i, m in enumerate(metadata)
    }

    faiss_ret = FAISSRetriever(faiss_index, cfg.embed_model, chunk_id_map=chunk_id_map)
    bm25_ret = BM25Retriever(bm25_index)

    # Optional SectionSummaryRetriever
    summary_ret = None
    kg_run_dir = ROOT / "data" / "knowledge_graph" / "runs" / "latest"
    try:
        from src.knowledge_graph.io import load_summary_data
        from src.knowledge_graph.query import SectionSummaryRetriever
        s_idx, s_entries = load_summary_data(str(kg_run_dir))
        if s_idx is not None:
            summary_ret = SectionSummaryRetriever(s_idx, s_entries)
            print("SectionSummaryRetriever loaded.")
    except Exception as e:
        print(f"SectionSummaryRetriever unavailable: {e}")

    benchmarks_path = ROOT / "tests" / "benchmarks.yaml"
    with open(benchmarks_path, encoding="utf-8") as f:
        bm_data = yaml.safe_load(f)
    benchmarks = bm_data.get("benchmarks", [])

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    llm_available = bool(openrouter_key)
    if llm_available:
        print("LLM features enabled (OpenRouter).")

    _st.update({
        "chunks": chunks,
        "metadata": metadata,
        "chunk_id_map": chunk_id_map,
        "chunk_id_to_idx": chunk_id_to_idx,
        "chunk_info": chunk_info,
        "sections_tree": _build_sections_tree(metadata),
        "section_titles": _collect_section_titles(metadata),
        "benchmarks": benchmarks,
        "benchmarks_path": str(benchmarks_path),
        "faiss_ret": faiss_ret,
        "bm25_ret": bm25_ret,
        "summary_ret": summary_ret,
        "openrouter_key": openrouter_key,
        "llm_available": llm_available,
    })
    print("Chunk Tagger ready.")
    yield
    print("Chunk Tagger shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Chunk Tagger", lifespan=lifespan)

_BENCHMARK_KEYS = ("id", "question", "keywords", "ideal_retrieved_chunks",
                   "sections", "broad", "mode", "notes")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    html = pathlib.Path(__file__).parent / "index.html"
    return FileResponse(html, media_type="text/html")


@app.get("/api/data")
async def get_data():
    benchmarks_light = [
        {k: b.get(k) for k in _BENCHMARK_KEYS}
        for b in _st["benchmarks"]
    ]
    return JSONResponse({
        "sections_tree": _st["sections_tree"],
        "benchmarks": benchmarks_light,
        "llm_available": _st["llm_available"],
        "has_summary": _st["summary_ret"] is not None,
    })


@app.get("/api/chunk/{chunk_id}")
async def get_chunk(chunk_id: int):
    info = _st["chunk_info"].get(chunk_id)
    if info is None:
        raise HTTPException(404, f"chunk_id {chunk_id} not found")
    return JSONResponse(info)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 30


@app.post("/api/retrieve")
async def retrieve(req: RetrieveRequest):
    chunks = _st["chunks"]
    top_k = max(req.top_k, 10)

    faiss_scores: dict = _st["faiss_ret"].get_scores(req.query, top_k, chunks)

    bm25_raw: dict = _st["bm25_ret"].get_scores(req.query, top_k, chunks)
    bm25_scores = _bm25_to_chunk_ids(bm25_raw)

    summary_scores: dict = {}
    if _st["summary_ret"]:
        summary_scores = _st["summary_ret"].get_scores(req.query, top_k, chunks)

    # Normalise scores to [0,1] within each retriever for fair comparison
    def _norm(d: dict) -> dict:
        if not d:
            return d
        mx = max(d.values())
        return {k: v / mx for k, v in d.items()} if mx > 0 else d

    return JSONResponse({
        "faiss": _norm(faiss_scores),
        "bm25": _norm(bm25_scores),
        "summary": _norm(summary_scores),
    })


class LLMHintRequest(BaseModel):
    query: str
    keywords: List[str] = []


@app.post("/api/llm-hint")
async def llm_hint(req: LLMHintRequest):
    if not _st["llm_available"]:
        raise HTTPException(503, "LLM not available (no OPENROUTER_API_KEY)")

    section_list = "\n".join(_st["section_titles"])
    kw_str = ", ".join(req.keywords) if req.keywords else "(none)"
    prompt = (
        "You are a database textbook expert. "
        "Given a query, identify which sections of the textbook most likely contain relevant content.\n\n"
        f"Query: {req.query}\n"
        f"Keywords: {kw_str}\n\n"
        "Available sections (one per line):\n"
        f"{section_list}\n\n"
        "Return JSON only, using exact section strings from the list above:\n"
        '{"sections": ["Section X.Y Title", ...], "reasoning": "brief explanation"}\n'
        "Return at most 10 section titles."
    )
    try:
        raw = _call_llm(prompt, json_mode=True)
        parsed = json.loads(raw)
        return JSONResponse({
            "suggested_sections": parsed.get("sections", []),
            "reasoning": parsed.get("reasoning", ""),
        })
    except Exception as e:
        raise HTTPException(500, f"LLM call failed: {e}")


class LLMScoreRequest(BaseModel):
    query: str
    chunk_ids: List[int]


@app.post("/api/llm-score")
async def llm_score(req: LLMScoreRequest):
    if not _st["llm_available"]:
        raise HTTPException(503, "LLM not available (no OPENROUTER_API_KEY)")

    chunk_ids = req.chunk_ids[:30]  # cap to control token budget
    chunk_info = _st["chunk_info"]

    blocks = []
    for cid in chunk_ids:
        info = chunk_info.get(cid)
        if info:
            text = info["text"][:500].replace('"', "'")
            blocks.append(f'chunk_id {cid} [{info["section"]}]:\n"{text}"')

    if not blocks:
        return JSONResponse({"scores": {}})

    chunks_text = "\n\n---\n\n".join(blocks)
    prompt = (
        "Rate the relevance of each chunk to the query on a scale of 0.0 to 1.0.\n"
        "1.0 = directly and fully answers the query. 0.0 = completely unrelated.\n\n"
        f"Query: {req.query}\n\n"
        f"Chunks:\n{chunks_text}\n\n"
        "Return JSON only, including all chunk_ids:\n"
        '{"scores": {"<chunk_id>": <float 0.0-1.0>, ...}}'
    )
    try:
        raw = _call_llm(prompt, json_mode=True)
        parsed = json.loads(raw)
        scores = parsed.get("scores") or {}
        # Fallback: LLM may return scores at the top level instead of nested
        if not scores and all(isinstance(v, (int, float)) for v in parsed.values()):
            scores = parsed
        return JSONResponse({"scores": scores})
    except Exception as e:
        raise HTTPException(500, f"LLM call failed: {e}")


class SaveRequest(BaseModel):
    benchmark_id: str
    ideal_retrieved_chunks: List[int]


@app.post("/api/save")
async def save(req: SaveRequest):
    path = pathlib.Path(_st["benchmarks_path"])
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    found = False
    for entry in data.get("benchmarks", []):
        if entry.get("id") == req.benchmark_id:
            entry["ideal_retrieved_chunks"] = sorted(req.ideal_retrieved_chunks)
            found = True
            break

    if not found:
        raise HTTPException(404, f"Benchmark '{req.benchmark_id}' not found")

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Update in-memory state
    for b in _st["benchmarks"]:
        if b.get("id") == req.benchmark_id:
            b["ideal_retrieved_chunks"] = sorted(req.ideal_retrieved_chunks)
            break

    return JSONResponse({"ok": True})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Chunk Tagger server")
    parser.add_argument("--port", type=int, default=7654)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    uvicorn.run(
        "tools.chunk_tagger.server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
