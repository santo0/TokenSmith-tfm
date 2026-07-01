# TokenSmith + KG Routing

This repository is a fork of [TokenSmith](https://github.com/georgia-tech-db/TokenSmith), extended with a keyword graph layer as part of the master's thesis "Graph-Structural Heuristics for Adaptive Query Routing in Resource-Constrained RAG", submitted to the Facultat d'Informàtica de Barcelona (FIB), Universitat Politècnica de Catalunya (UPC), 2026.

The thesis contribution is contained entirely in [`src/knowledge_graph/`](src/knowledge_graph/). The rest of the codebase is the upstream TokenSmith system, a local-first RAG engine for querying textbooks and technical documents with local LLMs.

<img width="1255" height="843" alt="tokensmith" src="https://github.com/user-attachments/assets/b36d6227-8cec-4f71-aacc-fccdd1285378" />

## Abstract

Retrieval-augmented generation (RAG) improves the reliability of large language models (LLMs), but its computational demands restrict deployment. In multi-user settings, a central GPU server bottlenecks on routine queries that a smaller system could handle. This thesis proposes an edge-first distributed architecture: CPU-only devices, each holding a curated local corpus, answer routine queries on-device and delegate to an expensive GPU-backed server only when the local data cannot support a reliable answer. For the delegation to pay off, the routing decision must be taken before any local retrieval budget is spent. The proposed solution is a dual-role keyword co-occurrence graph (KG), implemented within the TokenSmith system and built offline from concepts extracted from the corpus. The KG expands local retrieval through subgraph search and informs pre-retrieval routing through the topology of a query's matched concepts. Evaluation of ten structural features over a database-systems benchmark of 34 answerable and 10 unanswerable queries gives three findings. Queries the corpus cannot cover resolve to wider subgraphs spanning more communities, so a topology-only test flags them before any retrieval runs, with large effect sizes that hold with or without entity canonicalization. Within the answerable set, by contrast, no feature separates easy from hard queries once corrected for multiple comparisons, so graph topology detects the queries a corpus cannot answer without grading the difficulty of those it can. Fusing an offline summary tree with dense retrieval improves recall on CPU over either alone. These results show that graph topology supplies a cheap pre-retrieval signal, computed before any model forward pass, that can reserve server compute for the queries a local device cannot answer at all, and they locate where the original three-band routing design holds and where it must be reviewed.

## Thesis Contributions

Three concrete contributions come out of this work, all implemented in `src/knowledge_graph/`:

**Dual-role keyword co-occurrence graph.** A single graph built offline from LLM-extracted corpus concepts serves two purposes at query time: BFS subgraph expansion improves local retrieval recall on CPU, and structural features of the matched-concept subgraph provide a routing signal before any retrieval runs.

**Topology-based answerability detection.** Unanswerable queries (those whose answer lies outside the local corpus) produce subgraphs that span significantly more connected components and Leiden communities than answerable ones. This signal has large effect sizes (Cohen's d > 0.8) and is robust to whether entity canonicalization is applied.

**Summary-tree fusion.** Combining an offline LLM-generated section-summary index with dense FAISS retrieval yields higher recall on CPU than either retriever alone, providing a practical path to richer context without a GPU.

## Experiment Scripts

All experiment scripts live in [`src/knowledge_graph/scripts/`](src/knowledge_graph/scripts/) and are run as Python modules from the repo root with `conda activate tokensmith`. Most require a built KG run directory (default: `data/knowledge_graph/runs/latest`).

### Data preparation

```shell
# Sample N chunks and annotate them with LLM gold-standard keywords (used by A1/A2b)
python -m src.knowledge_graph.scripts.annotate_chunks \
    --sample 50 --model google/gemini-2.5-flash --output annotated_chunks.json

# Populate the `sections` field in benchmarks YAML using LLM annotation
python -m src.knowledge_graph.scripts.populate_benchmark_sections \
    --run-dir data/knowledge_graph/runs/latest \
    --benchmarks tests/benchmarks.yaml

# Pre-compute and cache canonicalization (avoids re-running LLM on every experiment)
python -m src.knowledge_graph.scripts.generate_canon_cache \
    --output debug/canonicalization_cache.json
```

### Series A — Extractor quality

```shell
# A1: compare LLM vs KeyBERT vs YAKE keyword extraction against gold annotations
python -m src.knowledge_graph.scripts.experiment_a1_extractor_quality \
    --annotated-chunks annotated_chunks.json \
    --llm-model google/gemini-2.5-flash \
    --output results_a1.json

# A2: keyword count sensitivity sweep
python -m src.knowledge_graph.scripts.experiment_a2_keyword_count \
    --output results_a2.json

# A2b: adaptive top_n vs fixed count
python -m src.knowledge_graph.scripts.experiment_a2b_adaptive_topn \
    --annotated-chunks annotated_chunks.json --output results_a2b.json

# A3: effect of seed-keyword selection strategy
python -m src.knowledge_graph.scripts.experiment_a3_seed_keywords --output results_a3.json

# A4: co-occurrence threshold sweep
python -m src.knowledge_graph.scripts.experiment_a4_threshold_sweep --output results_a4.json
```

### Series B — Retrieval quality

```shell
# B2: subgraph expansion recall vs dense-only (FAISS)
python -m src.knowledge_graph.scripts.experiment_b2_subgraph_recall \
    --run-dir data/knowledge_graph/runs/latest \
    --artifacts-dir index/sections \
    --output results_b2.json

# B4: full ablation across 8 retriever configurations
python -m src.knowledge_graph.scripts.experiment_b4_full_ablation \
    --run-dir data/knowledge_graph/runs/latest \
    --artifacts-dir index/sections \
    --output results_b4.json

# B5: canonicalized vs raw graph retrieval comparison
python -m src.knowledge_graph.scripts.experiment_b5_canonicalization \
    --canonical-run-dir data/knowledge_graph/runs/latest \
    --raw-run-dir data/knowledge_graph/runs/<raw-timestamp> \
    --output results_b5_canonicalization.json
```

### Series C — Topology features and routing signal

```shell
# C: compute topology features for answerable queries
python -m src.knowledge_graph.scripts.experiment_c_difficulty \
    --run-dir data/knowledge_graph/runs/latest \
    --benchmarks tests/benchmarks_chp.yaml \
    --output results_c.json

# C (unanswerable): same features for unanswerable queries
python -m src.knowledge_graph.scripts.experiment_c_difficulty \
    --run-dir data/knowledge_graph/runs/latest \
    --benchmarks tests/benchmarks_chp_unanswerable.yaml \
    --output results_c_unanswerable.json

# C series: unified statistical analysis (easy / hard / unanswerable)
python -m src.knowledge_graph.scripts.experiment_c_series \
    --topology results_c.json \
    --topology-unans results_c_unanswerable.json \
    --benchmarks tests/benchmarks_chp.yaml \
    --output results_c_series.json

# C comparison: Kruskal-Wallis, Cohen's d, LOOCV AUC across all feature groups
python -m src.knowledge_graph.scripts.experiment_c_comparison \
    --output results_c_comparison.json
```

### Utilities

```shell
# Analyse a single query: topology features, community dispersion, coverage
python -m src.knowledge_graph.scripts.analyze_query \
    --query "What is ARIES?" --run-dir data/knowledge_graph/runs/latest

# Visualize the subgraph a query resolves to (saves a PNG)
python -m src.knowledge_graph.scripts.visualize_query_subgraph \
    --query "How does MVCC handle write conflicts?" \
    --output query_viz.png

# Detect and cache Leiden communities on the graph
python -m src.knowledge_graph.scripts.leiden_communities \
    --run-dir data/knowledge_graph/runs/latest \
    --output communities.json --visualize communities.png

# Reproduce all thesis figures and LaTeX tables
python -m src.knowledge_graph.scripts.figures
# Or a single figure by name:
python -m src.knowledge_graph.scripts.figures mean_distance_groups
```

---

## TokenSmith (Base System)

The sections below document the upstream TokenSmith system. See [src/knowledge_graph/README.md](src/knowledge_graph/README.md) for documentation specific to the thesis contribution.

### Capabilities

* Parse and index PDF documents
* Semantic retrieval with FAISS
* Local inference via `llama.cpp` (GGUF models)
* Acceleration: Metal (Apple Silicon), CUDA (NVIDIA), or CPU
* Configurable chunking (tokens or characters)
* Optional indexing progress visualization
* Table preservation during indexing (flag-based)

### Requirements

* Python 3.9+
* Conda/Miniconda
* System prerequisites:
  * macOS: Xcode Command Line Tools
  * Linux: GCC, make, CMake
  * Windows: Visual Studio Build Tools

### Quick Start

#### 1) Clone the repository and download the models

```shell
git clone https://github.com/georgia-tech-db/TokenSmith.git
cd TokenSmith
```

Create the model directories and put the appropriate models in them.

```shell
mkdir -p models/generators models/embedders
```

With the following config:

```yaml
embed_model: "models/embedders/Qwen3-Embedding-4B-Q5_K_M.gguf"
model_path: "models/generators/qwen2.5-1.5b-instruct-q5_k_m.gguf"
```

Download the corresponding files from:
- https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/tree/main
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/main

#### 2) Build (creates env, builds llama.cpp, installs deps)

```shell
make build
```

##### Troubleshooting: NumPy version conflict

If you see `NumPy 1.x cannot be run in NumPy 2.x` errors:

```shell
conda activate tokensmith
conda uninstall faiss-cpu -y
conda install -c conda-forge faiss-cpu
```

This downgrades NumPy to a version compatible with FAISS. The pip version of FAISS cannot be used here because of multiple OpenMP instantiations on Apple Silicon.

#### 3) Activate the environment

```shell
conda activate tokensmith
```

#### 4) Prepare documents

```shell
mkdir -p data/chapters
cp your-documents.pdf data/chapters/
```

#### 5) Extract PDF to markdown

```shell
make run-extract
```

This generates markdown files under `data/`.

#### 6) Index documents

```shell
make run-index
```

With custom parameters:

```shell
make run-index ARGS="--chunk_mode chars --visualize"
```

To index a portion of the document:

```shell
make run-index-partial CHAPTERS="1 2"
```

To add chapters to an existing index:

```shell
make run-add-chapters-partial CHAPTERS="3"
```

#### 7) Build the knowledge graph (thesis contribution)

See [src/knowledge_graph/README.md](src/knowledge_graph/README.md) for the full pipeline. The short version:

```shell
# Extract keywords with a cloud LLM (requires OPENROUTER_API_KEY)
python -m src.knowledge_graph.scripts.llm_extract_keywords \
  --model google/gemini-2.5-flash

# Build the graph from cached extractions
python -m src.knowledge_graph.scripts.run_kg_pipeline
```

#### 8) Chat

```shell
python -m src.main chat
```

For a partial index:

```shell
python -m src.main chat --partial
```

#### 9) Deactivate

```shell
conda deactivate
```

### Configuration

Priority (highest → lowest):

1. `--config` CLI argument
2. `~/.config/tokensmith/config.yaml`
3. `config/config.yaml`

#### Example

```yaml
embed_model: "models/embedders/all-MiniLM-L6-v2"
top_k: 5
max_gen_tokens: 400
halo_mode: "none"
seg_filter: null

model_path: "models/generators/qwen2.5-0.5b-instruct-q5_k_m.gguf"

chunk_mode: "tokens"
chunk_tokens: 500
chunk_size_char: 20000

kg_pipeline:
  corpus_description: "Database System Concepts, 7th edition by Silberschatz et al."
  min_cooccurrence: 0
  top_n: 10
```

### Usage

```shell
# Basic indexing
make run-index

# Partial indexing
make run-index-partial CHAPTERS="1 2"

# Add chapters to an existing index
make run-add-chapters-partial CHAPTERS="3"

# Index a specific PDF range
make run-index ARGS="--pdf_range <start>-<end> --chunk_mode <tokens|chars>"

# Chat with custom settings
python -m src.main chat --config <path_to_yaml> --model_path <path_to_gguf>
```

### Command-Line Arguments

#### Core

* `mode`: `index` or `chat`
* `--config`: path to YAML config
* `--pdf_dir`: directory with PDFs
* `--index_prefix`: prefix for index files
* `--model_path`: path to GGUF model
* `--partial`: instantiate chat from a partial index

#### Indexing

* `--pdf_range`: e.g., `1-10`
* `--chunk_mode`: `tokens` or `chars`
* `--chunk_tokens`: default 500
* `--chunk_size_char`: default 20000
* `--keep_tables`
* `--visualize`

### Development

```shell
make help        # list all targets
make build       # full build (env + llama.cpp + deps)
make test        # run test suite
make clean       # remove build artifacts
make update-env  # sync conda env from environment.yml
make export-env  # export current env to environment.yml
make show-deps   # list installed packages
```

### Testing

```shell
pytest tests/
pytest tests/ -s
pytest tests/ --benchmark-ids="test" -s
```

* Tests call the same `get_answer()` pipeline used by chat
* Metrics: semantic similarity, BLEU, keyword matching, text similarity
* Outputs: terminal logs and HTML report

Benchmark definitions are in `tests/benchmarks*.yaml`. The database-systems benchmark used in the thesis is `tests/benchmarks_chp.yaml` (answerable) and `tests/benchmarks_chp_unanswerable.yaml`.

Documentation: see `tests/README.md`.
