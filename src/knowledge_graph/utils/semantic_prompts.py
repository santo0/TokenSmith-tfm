"""Prompts, relation taxonomy, and default weights for the semantic KG."""

RELATION_TYPES: list[str] = [
    "DEFINES",
    "IS_A",
    "HAS_PROPERTY",
    "USES",
    "REQUIRES",
    "PART_OF",
    "CAUSES",
    "CONTRASTS_WITH",
    "IMPLEMENTED_BY",
    "EXAMPLE_OF",
]

RELATION_DESCRIPTIONS: dict[str, str] = {
    "DEFINES": "A defines / is the definition of B",
    "IS_A": "A is a subtype, subclass, or specific kind of B",
    "HAS_PROPERTY": "A has attribute, property, or characteristic B",
    "USES": "Component or algorithm A uses mechanism or structure B",
    "REQUIRES": "A requires B as a precondition or dependency",
    "PART_OF": "A is a component or structural part of B",
    "CAUSES": "Event or condition A causes, leads to, or results in B",
    "CONTRASTS_WITH": "A is contrasted with, distinct from, or compared against B",
    "IMPLEMENTED_BY": "Abstract concept A is implemented or realized by concrete mechanism B",
    "EXAMPLE_OF": "A is given in the text as a specific example of B",
}

# Default edge weights used by SemanticKGRetriever when scoring neighbors.
# Higher = more important signal for retrieval.
DEFAULT_RELATION_WEIGHTS: dict[str, float] = {
    "DEFINES": 1.0,
    "IS_A": 0.9,
    "HAS_PROPERTY": 0.8,
    "USES": 0.7,
    "REQUIRES": 0.7,
    "PART_OF": 0.7,
    "CAUSES": 0.8,
    "CONTRASTS_WITH": 0.6,
    "IMPLEMENTED_BY": 0.75,
    "EXAMPLE_OF": 0.5,
}

# Relation types grouped by the query intent they best match.
# Used by QueryIntentClassifier to boost relevant edges.
INTENT_TO_RELATIONS: dict[str, list[str]] = {
    "definitional": ["DEFINES", "IS_A", "EXAMPLE_OF"],
    "structural": ["PART_OF", "HAS_PROPERTY", "IMPLEMENTED_BY"],
    "causal": ["CAUSES", "REQUIRES"],
    "comparative": ["CONTRASTS_WITH"],
    "procedural": ["USES", "REQUIRES", "IMPLEMENTED_BY"],
}

# ── Triple extraction prompts ──────────────────────────────────────────────────

def _format_relation_list() -> str:
    lines = []
    for rel in RELATION_TYPES:
        lines.append(f"  - {rel}: {RELATION_DESCRIPTIONS[rel]}")
    return "\n".join(lines)


TRIPLE_EXTRACTION_SYSTEM_PROMPT = (
    "You are a knowledge graph construction expert analysing a technical textbook.\n"
    "Your task is to extract factual semantic relationships between technical concepts "
    "that are explicitly stated or directly implied in the provided text.\n"
    "Be precise and conservative: only extract relationships that are clearly supported "
    "by the text. Prefer shorter noun phrases (1-3 words) for subjects and objects.\n"
    "Do not invent relationships that are not present in the text."
)

_RELATION_LIST_STR = _format_relation_list()

TRIPLE_EXTRACTION_USER_TEMPLATE = """\
Extract semantic triples from the following textbook chunks.

Allowed relation types:
{relation_list}

{keyword_section}
Chunks:
{chunks_section}

Return JSON only — no markdown, no explanation:
{{
  "chunks": [
    {{
      "chunk_id": <int>,
      "triples": [
        {{"subject": "...", "relation": "<RELATION_TYPE>", "object": "..."}},
        ...
      ]
    }}
  ]
}}

Rules:
1. Use only relation types from the list above.
2. Subjects and objects must be noun phrases (1-4 words), not full sentences.
3. If a chunk contains no clear relationships, return an empty triples list.
4. Maximum 8 triples per chunk.
5. Both subject and object must be distinct concepts (not the same string).
"""

KEYWORD_SECTION_TEMPLATE = "Preferred concepts (use these as subjects/objects when possible):\n{keywords}\n\n"

CHUNK_ENTRY_TEMPLATE = "--- [Chunk {chunk_id}] ---\n{text}\n"

# ── Query intent classification prompts ───────────────────────────────────────

INTENT_CLASSIFICATION_SYSTEM_PROMPT = (
    "You are a query analysis assistant for a database systems textbook Q&A system.\n"
    "Classify the intent of a user query by selecting the 1-3 most relevant relation "
    "types from a fixed taxonomy. Return only JSON."
)

INTENT_CLASSIFICATION_USER_TEMPLATE = """\
Classify the intent of this query by selecting the most relevant relation types.

Available relation types:
{relation_list}

Query: "{query}"

Return JSON only:
{{"intent_relations": ["REL_TYPE_1", ...], "reason": "brief explanation"}}

Guidance:
- DEFINES / IS_A / EXAMPLE_OF  →  definitional queries ("What is X?", "Define X")
- HAS_PROPERTY / USES / REQUIRES  →  property or mechanism queries ("How does X work?")
- CAUSES  →  causal queries ("Why does X happen?", "What leads to X?")
- CONTRASTS_WITH  →  comparison queries ("How does X differ from Y?")
- PART_OF / IMPLEMENTED_BY  →  structural or implementation queries ("What is X part of?")
"""
