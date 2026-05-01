"""Populate the `sections` field for every benchmark entry using LLM annotation.

For each benchmark, prompts an LLM with the query, expected answer, and the
full list of textbook section headings (levels 2–3), then records which sections
the LLM identifies as likely to contain the relevant content.

The result is written back to the benchmarks YAML as a `sections` list of
heading strings. The user should manually verify the annotations afterward.

Usage:
    python -m src.knowledge_graph.scripts.populate_benchmark_sections \\
        --run-dir data/knowledge_graph/runs/latest \\
        --benchmarks tests/benchmarks.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_SYSTEM_PROMPT = (
    "You are a database textbook expert. Given a question and its expected answer, "
    "identify which sections of the textbook are most likely to contain the "
    "information needed to answer it. "
    "Return ONLY a JSON array of section heading strings, exactly as they appear "
    "in the provided list — no other text."
)


def _build_section_list(section_tree) -> tuple[str, list[str]]:
    """Return a formatted string listing all level-2/3 sections and the flat heading list."""
    grouped: dict[int, list] = defaultdict(list)
    for node in section_tree.node_index.values():
        if node.level in (2, 3):
            grouped[node.chapter].append(node)

    for chapter in grouped:
        grouped[chapter].sort(key=lambda n: n.section_number)

    lines: list[str] = []
    all_headings: list[str] = []
    for chapter in sorted(grouped):
        lines.append(f"Chapter {chapter}:")
        for node in grouped[chapter]:
            indent = "  " if node.level == 2 else "    "
            lines.append(f"{indent}{node.heading}")
            all_headings.append(node.heading)

    return "\n".join(lines), all_headings


def _parse_response(raw: str, valid_headings: set[str], bm_id: str) -> list[str]:
    """Extract JSON array from LLM response and validate headings."""
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    try:
        items = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [error]   {bm_id} — JSON parse failed: {e}")
        return []

    if not isinstance(items, list):
        print(f"  [error]   {bm_id} — LLM did not return a list")
        return []

    result: list[str] = []
    for h in items:
        if h in valid_headings:
            result.append(h)
        else:
            print(f"  [warning] {bm_id} — unrecognized heading (skipped): {h!r}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate `sections` field in benchmarks.yaml via LLM annotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: overwrite --benchmarks in-place)",
    )
    parser.add_argument("--model", default="anthropic/claude-opus-4.7")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip benchmark entries that already have a `sections` field",
    )
    args = parser.parse_args()

    from src.knowledge_graph.io import load_graph_chunks_and_tree
    from src.knowledge_graph.openrouter_client import OpenRouterClient

    root = Path(__file__).parent.parent.parent.parent
    run_path = Path(args.run_dir)
    if not run_path.is_absolute():
        run_path = root / run_path
    benchmarks_path = Path(args.benchmarks)
    if not benchmarks_path.is_absolute():
        benchmarks_path = root / benchmarks_path
    output_path = Path(args.output) if args.output else benchmarks_path
    if not output_path.is_absolute():
        output_path = root / output_path

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading section tree from {run_path}...")
    _graph, _chunks, section_tree = load_graph_chunks_and_tree(str(run_path))

    if section_tree is None:
        print("ERROR: section_tree.json not found in run directory.", file=sys.stderr)
        sys.exit(1)

    section_list_str, all_headings = _build_section_list(section_tree)
    valid_headings = set(all_headings)
    print(f"Loaded {len(all_headings)} sections (levels 2–3).")

    print(f"Loading benchmarks from {benchmarks_path}...")
    with open(benchmarks_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    benchmarks: list[dict] = data.get("benchmarks", [])
    client = OpenRouterClient(api_key=api_key)

    n_updated = n_skipped = n_no_question = 0

    for entry in benchmarks:
        bm_id = entry.get("id", "?")
        question = entry.get("question")
        if not question:
            n_no_question += 1
            continue

        if args.skip_existing and "sections" in entry:
            n_skipped += 1
            print(f"  [skip]    {bm_id} — already has sections")
            continue

        expected = entry.get("expected_answer", "")
        user_content = (
            f"Question: {question}\n\n"
            f"Expected answer: {expected}\n\n"
            f"Sections of the textbook (organized by chapter):\n{section_list_str}\n\n"
            "Return a JSON array of section headings that are most relevant to answering "
            "this question. Include all sections where the answer content is likely found."
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        print(f"  [querying] {bm_id}...")
        try:
            raw = client.chat(args.model, messages, timeout=90)
        except Exception as e:
            print(f"  [error]   {bm_id} — LLM call failed: {e}")
            entry["sections"] = []
            n_updated += 1
            continue

        sections = _parse_response(raw, valid_headings, bm_id)
        entry["sections"] = sections
        n_updated += 1
        print(f"  [updated] {bm_id} → {sections}")

    print(f"\nWriting updated benchmarks to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Done. Updated: {n_updated}, Skipped: {n_skipped}, No question: {n_no_question}")


if __name__ == "__main__":
    load_dotenv()
    main()
