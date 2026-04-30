"""
Testbed — loads all pipeline data structures from the latest run so you can
call specific functions manually below the separator.

Usage:
    conda run -n tokensmith python -m src.knowledge_graph.scripts.testbed
"""
import os
import logging

import numpy as np
from dotenv import load_dotenv

from src.knowledge_graph.build import RUNS_DIR
from src.knowledge_graph.io import (
    resolve_run_dir,
    load_run_chunks,
    load_graph,
    load_canonicalization_data,
    load_summary_data,
    load_keyword_index,
)
from src.knowledge_graph.section_tree import load_section_tree
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.canonicalizer import Canonicalizer
from src.knowledge_graph.extractors.openrouter_extractor import OpenRouterExtractor
from src.knowledge_graph.models import ExtractionResult, Chunk

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
load_dotenv()

# ── Load run artifacts ────────────────────────────────────────────────────────

RUN_DIR = resolve_run_dir(RUNS_DIR)
print(f"Run dir: {RUN_DIR}")

chunks: dict[int, str] = load_run_chunks(os.path.join(RUN_DIR, "chunks.json"))
graph = load_graph(os.path.join(RUN_DIR, "graph.json"))
section_tree = load_section_tree(RUN_DIR)

synonym_table, canonical_keywords, canonical_embeddings = load_canonicalization_data(RUN_DIR)
keyword_index = load_keyword_index(RUN_DIR)
summary_index, summary_entries = load_summary_data(RUN_DIR)

# ── Build live objects (need OPENROUTER_API_KEY in .env) ─────────────────────

api_key = os.environ.get("OPENROUTER_API_KEY", "")
client = OpenRouterClient(api_key, retries=1)

# ── Quick summary ─────────────────────────────────────────────────────────────

print(f"  chunks          : {len(chunks)}")
print(f"  graph nodes     : {graph.number_of_nodes()}")
print(f"  graph edges     : {graph.number_of_edges()}")
print(f"  canonical kws   : {len(canonical_keywords) if canonical_keywords else 'n/a'}")
print(f"  summary entries : {len(summary_entries) if summary_entries else 'n/a'}")
print(f"  section tree    : {len(section_tree.root.children)} top-level nodes")

# ─────────────────────────────────────────────────────────────────────────────
# Write your code below
# ─────────────────────────────────────────────────────────────────────────────

# Load raw (pre-canonicalization) extraction for chunk 855
import json

raw_extractions_path = os.path.join(RUN_DIR, "input", "extractions.json")
with open(raw_extractions_path, "r", encoding="utf-8") as f:
    raw_extractions = json.load(f)

chunk_855_data = next((e for e in raw_extractions if e["chunk_id"] == 855), None)
assert chunk_855_data is not None, "Chunk 855 not found in extractions"
extraction_855 = ExtractionResult(chunk_id=855, keywords=chunk_855_data["keywords"])

print(f"\nChunk 855 raw keywords ({len(extraction_855.keywords)}): {extraction_855.keywords}")

# Build a Canonicalizer matching the run config
CORPUS_DESCRIPTION = "Database System Concepts, 7th edition by Silberschatz et al."
canonicalizer = Canonicalizer(
    corpus_description=CORPUS_DESCRIPTION,
    api_key=api_key,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
)

# Canonicalize only chunk 855 — set breakpoints below to debug
updated, canon_result = canonicalizer.canonicalize([extraction_855])

print(f"\nCanonical keywords ({len(updated[0].keywords)}): {updated[0].keywords}")
print(f"Stats: {canon_result.stats}")

