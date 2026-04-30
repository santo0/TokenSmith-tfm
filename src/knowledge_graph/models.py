from dataclasses import dataclass, field
from typing import Any
from enum import Enum

import numpy as np
import yaml


@dataclass
class Chunk:
    id: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    chunk_id: int
    keywords: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalizationResult:
    synonym_table: dict[str, str]
    canonical_keywords: list[str]
    canonical_embeddings: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryFeatures:
    # D1 — Corpus Coverage
    query_concept_count: int = 0
    matched_concept_count: int = 0
    corpus_coverage: float = 0.0          # δ_cov ∈ [0, 1]
    # D2 — Retrieval Confidence
    retrieval_confidence: float = 0.0     # δ_conf ∈ [0, 1]; 0 when no scores provided
    # D3 — Context Capacity
    relevant_token_count: int = 0
    context_capacity: float = 0.0         # δ_cap ≥ 0 (may exceed 1)
    # D4 — Topological Complexity
    component_count: int = 0
    edge_density: float = 0.0
    subgraph_diameter: float = 0.0
    topological_complexity: float = 0.0   # δ_top ∈ [0, 1]
    subgraph_node_count: int = 0
    subgraph_edge_count: int = 0
    # D5 — Community Dispersion
    community_count: int = 0
    community_dispersion: float = 0.0     # δ_com ∈ [0, 1]

    def to_dict(self) -> dict:
        return {
            "query_concept_count": self.query_concept_count,
            "matched_concept_count": self.matched_concept_count,
            "corpus_coverage": self.corpus_coverage,
            "retrieval_confidence": self.retrieval_confidence,
            "relevant_token_count": self.relevant_token_count,
            "context_capacity": self.context_capacity,
            "component_count": self.component_count,
            "edge_density": self.edge_density,
            "subgraph_diameter": self.subgraph_diameter,
            "topological_complexity": self.topological_complexity,
            "subgraph_node_count": self.subgraph_node_count,
            "subgraph_edge_count": self.subgraph_edge_count,
            "community_count": self.community_count,
            "community_dispersion": self.community_dispersion,
        }


class DifficultyCategory(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class DifficultyComponents:
    corpus_coverage: float = 0.0
    retrieval_confidence: float = 0.0
    context_capacity: float = 0.0
    topological_complexity: float = 0.0
    community_dispersion: float = 0.0

    def to_dict(self) -> dict:
        return {
            "corpus_coverage": self.corpus_coverage,
            "retrieval_confidence": self.retrieval_confidence,
            "context_capacity": self.context_capacity,
            "topological_complexity": self.topological_complexity,
            "community_dispersion": self.community_dispersion,
        }


@dataclass
class DifficultyScore:
    score: float = 0.0
    category: DifficultyCategory = DifficultyCategory.EASY
    components: DifficultyComponents = field(default_factory=DifficultyComponents)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "category": self.category.value,
            "components": self.components.to_dict(),
        }


@dataclass
class QueryAnalysisResult:
    query: str
    features: QueryFeatures
    difficulty: DifficultyScore

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "features": self.features.to_dict(),
            "difficulty": self.difficulty.to_dict(),
        }


@dataclass
class RunMetadata:
    """Configuration and execution statistics for a pipeline run."""

    config: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "statistics": self.statistics,
        }


@dataclass
class ExtractorConfig:
    type: str = "json"
    extractions: str | None = None
    model: str = "qwen/qwen3-next-80b-a3b-instruct"
    adaptive_top_n: bool = False
    keybert_model: str = "all-MiniLM-L6-v2"
    slm_model_path: str = "models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
    slm_threads: int = 8


@dataclass
class CanonicalizationConfig:
    llm_model: str = "google/gemini-3-flash-preview"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.78
    max_group_size: int = 30
    batch_size: int = 15


@dataclass
class SummaryTreeConfig:
    summary_model: str = "google/gemini-3-flash-preview"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_window: int = 3


@dataclass
class KGPipelineConfig:
    corpus_description: str = ""
    min_cooccurrence: int = 0
    top_n: int = 10
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    partial: bool = False
    chapter: int | None = None
    exclude_chapters: list[int] = field(default_factory=list)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    canonicalization: CanonicalizationConfig = field(
        default_factory=CanonicalizationConfig
    )
    summary_tree: SummaryTreeConfig = field(
        default_factory=SummaryTreeConfig
    )

    @classmethod
    def from_yaml(cls, path: str) -> "KGPipelineConfig":
        """Load the ``kg_pipeline`` section from a project config YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        kg = dict(data.get("kg_pipeline", {}))
        canon_data = kg.pop("canonicalization", {})
        summary_tree_data = kg.pop("summary_tree", {})
        extractor_data = kg.pop("extractor", {})
        return cls(
            **kg,
            extractor=ExtractorConfig(**extractor_data),
            canonicalization=CanonicalizationConfig(**canon_data),
            summary_tree=SummaryTreeConfig(**summary_tree_data),
        )
