from contextlib import contextmanager
import argparse
import json
import logging
import os
import random
from time import strftime

import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

from src.knowledge_graph.build import CHUNKS_PKL, META_PKL, get_index_paths, load_chunks
from src.knowledge_graph.extractors import (
    BaseExtractor,
    KeyBERTExtractor,
    OpenRouterExtractor,
    YakeExtractor,
)
from src.knowledge_graph.models import Chunk, ExtractionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample_chunks(chunks: list[Chunk], n: int, seed: int) -> list[Chunk]:
    rng = random.Random(seed)
    if n >= len(chunks):
        result = list(chunks)
        rng.shuffle(result)
        return result
    return rng.sample(chunks, n)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _load_embed_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _batch_embed_for_method(
    chunks: list[Chunk],
    results: list[ExtractionResult],
    embed_model,
) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
    """Three batched encode() calls for a method's results.

    Returns:
        chunk_embs          : (n_chunks, dim)
        joined_kw_embs      : (n_chunks, dim)  — keywords joined by space
        per_chunk_kw_embs   : list[n_chunks] of list[top_n] of (dim,) arrays
    """
    result_map: dict[int, list[str]] = {
        r.chunk_id: r.keywords for r in results}

    chunk_texts = [c.text for c in chunks]
    joined_kw_texts = [
        " ".join(result_map.get(c.id, [])) or c.text for c in chunks
    ]

    # Flatten individual keywords; track slice indices to reassemble
    flat_kws: list[str] = []
    slices: list[tuple[int, int]] = []
    for c in chunks:
        kws = result_map.get(c.id, [])
        start = len(flat_kws)
        flat_kws.extend(kws if kws else [c.text])  # fallback keeps shape valid
        slices.append((start, len(flat_kws)))

    chunk_embs: np.ndarray = embed_model.encode(
        chunk_texts, show_progress_bar=False, normalize_embeddings=False
    )
    joined_kw_embs: np.ndarray = embed_model.encode(
        joined_kw_texts, show_progress_bar=False, normalize_embeddings=False
    )
    flat_embs: np.ndarray = embed_model.encode(
        flat_kws, show_progress_bar=False, normalize_embeddings=False
    )

    per_chunk_kw_embs: list[list[np.ndarray]] = []
    for i, c in enumerate(chunks):
        s, e = slices[i]
        kws = result_map.get(c.id, [])
        if kws:
            per_chunk_kw_embs.append([flat_embs[j] for j in range(s, e)])
        else:
            per_chunk_kw_embs.append([])

    return chunk_embs, joined_kw_embs, per_chunk_kw_embs


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------


# def _fit_tfidf(chunks: list[Chunk]):

#     texts = [c.text for c in chunks]
#     id_to_row = {c.id: i for i, c in enumerate(chunks)}
#     vectorizer = TfidfVectorizer()
#     matrix = vectorizer.fit_transform(texts)
#     return vectorizer, matrix, id_to_row

def get_global_vectorizer(
        all_corpus_texts: list[Chunk],
) -> tuple[TfidfVectorizer, dict[int, int]]:
    texts = [c.text for c in all_corpus_texts]
    id_to_row = {c.id: i for i, c in enumerate(all_corpus_texts)}
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    vectorizer.fit(texts)
    # mat = vectorizer.transform([c.text for c in chunks])
    return vectorizer,  id_to_row


def calculate_tfidf_informativeness(
        keywords: list[str],
        chunk_text: str,
        vectorizer: TfidfVectorizer
) -> float:
    # Transform the specific chunk text to get TF based on Global IDF
    matrix = vectorizer.transform([chunk_text])
    vocab = vectorizer.vocabulary_

    scores = []
    for kw in keywords:
        kw = kw.lower()
        # Try to get the score for the WHOLE phrase first (if bigrams enabled)
        col = vocab.get(kw)
        if col is not None:
            scores.append(matrix[0, col])
        else:
            # Fallback: Average of individual words
            word_scores = [matrix[0, vocab[w]]
                           for w in kw.split() if w in vocab]
            if word_scores:
                scores.append(np.mean(word_scores))

    return float(np.mean(scores)) if scores else 0.0

# ---------------------------------------------------------------------------
# Adaptive top-n extraction
# ---------------------------------------------------------------------------


def _compute_adaptive_top_n(chunk: Chunk) -> int:
    """Return per-chunk top_n scaled by text word count."""
    from math import sqrt

    return max(1, int(sqrt(len(chunk.text.split()))))


@contextmanager
def _override_top_n(extractor: BaseExtractor, top_n: int):
    """Temporarily replace extractor.top_n (and kw_extractor for YAKE)."""
    old_top_n = extractor.top_n
    extractor.top_n = top_n

    # YakeExtractor wraps a yake.KeywordExtractor whose `top` must also change.
    old_kw_extractor = getattr(extractor, "kw_extractor", None)
    if old_kw_extractor is not None:
        import yake

        extractor.kw_extractor = yake.KeywordExtractor(
            lan=getattr(extractor, "language", "en"),
            n=3,
            top=top_n,
            dedupLim=getattr(extractor, "deduplicate_threshold", 0.9),
        )
    try:
        yield
    finally:
        extractor.top_n = old_top_n
        if old_kw_extractor is not None:
            extractor.kw_extractor = old_kw_extractor


def _extract_adaptive(
    extractor: BaseExtractor,
    chunks: list[Chunk],
    base_top_n: int,
    adaptive: bool,
) -> tuple[list[ExtractionResult], dict[int, int]]:
    """Run extraction, optionally with per-chunk adaptive top_n.

    Returns:
        results         : list of ExtractionResult in the same order as chunks
        top_n_per_chunk : chunk_id → top_n actually used
    """
    if not adaptive:
        top_n_map = {c.id: base_top_n for c in chunks}
        return extractor.extract(chunks), top_n_map

    # Group chunks by their computed top_n to batch calls where possible.
    from collections import defaultdict

    groups: dict[int, list[Chunk]] = defaultdict(list)
    top_n_map: dict[int, int] = {}
    for chunk in chunks:
        n = _compute_adaptive_top_n(chunk)
        groups[n].append(chunk)
        top_n_map[chunk.id] = n

    results_by_id: dict[int, ExtractionResult] = {}
    for top_n, group in groups.items():
        with _override_top_n(extractor, top_n):
            for result in extractor.extract(group):
                results_by_id[result.chunk_id] = result

    # Return in original chunk order.
    return [results_by_id[c.id] for c in chunks], top_n_map


# ---------------------------------------------------------------------------
# Per-chunk metrics
# ---------------------------------------------------------------------------


def _compute_chunk_metrics(
    chunk_text: str,
    keywords: list[str],
    chunk_emb: np.ndarray,
    joined_kw_emb: np.ndarray,
    kw_embs: list[np.ndarray],
    tfidf_vectorizer,
    # tfidf_matrix,
    # chunk_row: int,
) -> dict:
    # 1. Embedding cosine similarity
    cos_sim = _cos_sim(chunk_emb, joined_kw_emb) if keywords else 0.0

    # 2. Diversity
    if len(kw_embs) >= 2:
        from sklearn.metrics.pairwise import cosine_similarity

        mat = cosine_similarity(np.array(kw_embs))
        n = len(kw_embs)
        n_pairs = n * (n - 1) / 2
        upper_sum = (mat.sum() - n) / 2.0
        avg_sim = upper_sum / n_pairs if n_pairs > 0 else 0.0
        diversity = 1.0 - float(avg_sim)
    else:
        diversity = 1.0

    # 3. TF-IDF informativeness
    # vocab = tfidf_vectorizer.vocabulary_
    # chunk_vec = tfidf_matrix[chunk_row]
    # scores: list[float] = []
    # for kw in keywords:
    #     word_scores = []
    #     for word in kw.lower().split():
    #         col = vocab.get(word)
    #         if col is not None:
    #             word_scores.append(float(chunk_vec[0, col]))
    #     if word_scores:
    #         scores.append(sum(word_scores) / len(word_scores))
    # tfidf_mean = float(np.mean(scores)) if scores else 0.0
    tfidf_mean = calculate_tfidf_informativeness(
        keywords, chunk_text, tfidf_vectorizer
    )

    # 4. Length
    avg_chars = float(np.mean([len(kw)
                      for kw in keywords])) if keywords else 0.0
    avg_words = (
        float(np.mean([len(kw.split())
              for kw in keywords])) if keywords else 0.0
    )

    return {
        "keywords": keywords,
        "cos_sim": round(cos_sim, 4),
        "diversity": round(diversity, 4),
        "tfidf_mean": round(tfidf_mean, 4),
        "avg_chars": round(avg_chars, 2),
        "avg_words": round(avg_words, 2),
    }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _compute_clustering(
    chunks: list[Chunk],
    results: list[ExtractionResult],
    embed_model,
    n_clusters: int,
    chunk_embs: np.ndarray,
) -> float:
    """K-Means silhouette score using mean keyword embedding as doc vector.

    Falls back to chunk text embedding for chunks with no keywords.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    result_map: dict[int, list[str]] = {
        r.chunk_id: r.keywords for r in results}

    # Build per-chunk keyword text for encoding
    kw_texts: list[str] = []
    for c in chunks:
        kws = result_map.get(c.id, [])
        kw_texts.append(" ".join(kws) if kws else c.text)

    kw_embs: np.ndarray = embed_model.encode(
        kw_texts, show_progress_bar=False, normalize_embeddings=False
    )

    # For chunks with no keywords use the chunk text embedding
    X_list: list[np.ndarray] = []
    for i, c in enumerate(chunks):
        kws = result_map.get(c.id, [])
        if kws:
            X_list.append(kw_embs[i])
        else:
            X_list.append(chunk_embs[i])

    X = np.array(X_list)
    n_valid = len(X)

    k = min(n_clusters, n_valid - 1)
    if k < 2:
        logger.warning(
            "Not enough samples (%d) for clustering with k=%d — skipping.",
            n_valid,
            n_clusters,
        )
        return float("nan")

    try:
        labels = KMeans(
            n_clusters=k, random_state=42, n_init="auto"
        ).fit_predict(X)
        if len(set(labels)) < 2:
            return float("nan")
        return float(silhouette_score(X, labels))
    except ValueError as e:
        logger.warning("Silhouette score failed: %s", e)
        return float("nan")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_NUMERIC_FIELDS = ("cos_sim", "diversity", "tfidf_mean",
                   "avg_chars", "avg_words")


def _aggregate_metrics(records: list[dict]) -> dict:
    result = {}
    for field in _NUMERIC_FIELDS:
        vals = [r[field] for r in records if field in r]
        if vals:
            result[field] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }
        else:
            result[field] = {"mean": None, "std": None}
    return result


# ---------------------------------------------------------------------------
# Jaccard
# ---------------------------------------------------------------------------


def _compute_jaccard(predicted: list[str], reference: list[str]) -> float:
    pred = {k.strip().lower() for k in predicted if k.strip()}
    ref = {k.strip().lower() for k in reference if k.strip()}

    if not pred and not ref:
        return 1.0

    intersection = len(pred.intersection(ref))
    union = len(pred.union(ref))

    return intersection / union


def _load_annotations(path: str) -> dict[int, list[str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        int(entry["chunk_id"]): entry.get("reference_keywords", [])
        for entry in data
    }


def _compute_jaccard_scores(
    annotations: dict[int, list[str]],
    results_by_id: dict[str, dict[int, ExtractionResult]],
    method_names: list[str],
) -> dict:
    scores: dict[str, list[float]] = {m: [] for m in method_names}
    for chunk_id, ref_kws in annotations.items():
        if not ref_kws:
            continue
        for method in method_names:
            result = results_by_id.get(method, {}).get(chunk_id)
            if result is not None:
                scores[method].append(
                    _compute_jaccard(result.keywords, ref_kws))
    return {
        method: {
            "mean": round(float(np.mean(vals)), 4) if vals else None,
            "std": round(float(np.std(vals)), 4) if vals else None,
            "n_annotated": len(vals),
        }
        for method, vals in scores.items()
    }


# ---------------------------------------------------------------------------
# Annotation template
# ---------------------------------------------------------------------------


def _generate_annotation_template(
    chunks: list[Chunk], n: int, seed: int, output_path: str
) -> None:
    def _wrap(text: str, width: int = 100) -> list[str]:
        lines = []
        for para in text.splitlines():
            if not para:
                lines.append("")
                continue
            while len(para) > width:
                lines.append(para[:width])
                para = para[width:]
            lines.append(para)
        return lines

    sample = _sample_chunks(chunks, min(n, len(chunks)), seed)
    template = [
        {
            "chunk_id": c.id,
            "text": _wrap(c.text),
            "reference_keywords": [],
        }
        for c in sample
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    print(f"Annotation template written to {output_path}")
    print(
        f"Fill in 'reference_keywords' for each chunk, then re-run with "
        f"--annotations {output_path}"
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(
    aggregate: dict,
    clustering: dict,
    jaccard: dict | None,
    methods: list[str],
) -> None:
    col_method = max(10, max((len(m) for m in methods), default=0) + 2)
    col_metric = 16

    metrics = list(_NUMERIC_FIELDS)
    header1 = f"{'Method':<{col_method}}"
    header2 = " " * col_method
    for m in metrics:
        header1 += f"  {m[:col_metric].center(col_metric)}"
        header2 += f"  {'mean ± std'.center(col_metric)}"
    header1 += f"  {'silhouette':>{col_metric}}"
    header2 += f"  {'':>{col_metric}}"
    if jaccard:
        header1 += f"  {'jaccard':>{col_metric}}"
        header2 += f"  {'mean ± std':>{col_metric}}"

    sep = "-" * len(header1)
    print(f"\n{sep}")
    print(header1)
    print(header2)
    print(sep)

    for method in methods:
        if method not in aggregate:
            continue
        row = f"{method:<{col_method}}"
        for field in metrics:
            agg = aggregate[method].get(field, {})
            mean_v = agg.get("mean")
            std_v = agg.get("std")
            if mean_v is not None:
                cell = f"{mean_v:.3f}±{std_v:.3f}"
            else:
                cell = "—"
            row += f"  {cell:>{col_metric}}"

        clust = clustering.get(method)
        if clust is not None and not (
            isinstance(clust, float) and clust != clust  # nan check
        ):
            row += f"  {clust:>{col_metric}.4f}"
        else:
            row += f"  {'—':>{col_metric}}"

        if jaccard and method in jaccard:
            j = jaccard[method]
            jm, js = j.get("mean"), j.get("std")
            if jm is not None:
                jcell = f"{jm:.3f}±{js:.3f}"
            else:
                jcell = "—"
            row += f"  {jcell:>{col_metric}}"

        print(row)

    print(sep)


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> dict:
    all_chunks = load_chunks(args.chunks_path, args.meta_path)
    chunks = _sample_chunks(all_chunks, args.n_chunks, args.seed)
    logger.info("Sampled %d / %d chunks (seed=%d)",
                len(chunks), len(all_chunks), args.seed)

    # Build extractors (insertion order defines display order)
    extractors: dict[str, BaseExtractor] = {
        "yake": YakeExtractor(top_n=args.top_n),
        "keybert": KeyBERTExtractor(
            top_n=args.top_n, keyphrase_ngram_range=(1, 2)
        ),
    }
    models = args.models if not args.skip_llm else []
    if models:
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            logger.warning(
                "No API key found — skipping LLM extractors. "
                "Pass --api-key or set OPENROUTER_API_KEY."
            )
        else:
            for model_id in models:
                extractors[model_id] = OpenRouterExtractor(
                    api_key=api_key,
                    model=model_id,
                    top_n=args.top_n,
                    adaptive_top_n=False,
                )

    active_methods = list(extractors.keys())

    # Extract keywords (adaptive top_n computed once, shared across methods)
    raw_results: dict[str, list[ExtractionResult]] = {}
    top_n_per_chunk: dict[int, int] = {}
    for name, extractor in extractors.items():
        logger.info("Running %s extractor on %d chunks…", name, len(chunks))
        results, top_n_map = _extract_adaptive(
            extractor, chunks, args.top_n, args.adaptive_top_n
        )
        raw_results[name] = results
        if not top_n_per_chunk:
            top_n_per_chunk = top_n_map  # same for all methods; store once

    # results_by_id[method][chunk_id] = ExtractionResult
    results_by_id: dict[str, dict[int, ExtractionResult]] = {
        name: {r.chunk_id: r for r in results}
        for name, results in raw_results.items()
    }

    # Fit TF-IDF once on the sampled corpus
    logger.info("Fitting TF-IDF on %d chunks…", len(chunks))
    # tfidf_vec, tfidf_mat, id_to_row = _fit_tfidf(chunks)
    tfidf_vec, id_to_row = get_global_vectorizer(all_chunks)

    # Load embedding model once
    logger.info("Loading embedding model: %s", args.embed_model)
    embed_model = _load_embed_model(args.embed_model)

    # Per-method batch embeddings
    method_emb_cache: dict[str, tuple] = {}
    for name in active_methods:
        logger.info("Batch-encoding embeddings for method: %s", name)
        method_emb_cache[name] = _batch_embed_for_method(
            chunks, raw_results[name], embed_model
        )

    # Per-chunk metrics
    per_chunk_output: list[dict] = []
    for i, chunk in enumerate(chunks):
        row: int = id_to_row[chunk.id]
        entry: dict = {
            "chunk_id": chunk.id,
            "top_n": top_n_per_chunk.get(chunk.id, args.top_n),
            "text_preview": chunk.text[:200],
            "methods": {},
        }
        for name in active_methods:
            chunk_embs, joined_embs, kw_emb_lists = method_emb_cache[name]
            result = results_by_id[name].get(chunk.id)
            keywords = result.keywords if result else []
            entry["methods"][name] = _compute_chunk_metrics(
                chunk_text=chunk.text,
                keywords=keywords,
                chunk_emb=chunk_embs[i],
                joined_kw_emb=joined_embs[i],
                kw_embs=kw_emb_lists[i],
                tfidf_vectorizer=tfidf_vec,
                # tfidf_matrix=tfidf_mat,
                # chunk_row=row,
            )
        per_chunk_output.append(entry)

    # Aggregate
    aggregate: dict[str, dict] = {}
    for name in active_methods:
        records = [e["methods"][name]
                   for e in per_chunk_output if name in e["methods"]]
        aggregate[name] = _aggregate_metrics(records)

    # Clustering
    clustering: dict[str, float | None] = {}
    for name in active_methods:
        logger.info("Computing clustering silhouette for method: %s", name)
        chunk_embs_for_method = method_emb_cache[name][0]
        score = _compute_clustering(
            chunks,
            raw_results[name],
            embed_model,
            args.n_clusters,
            chunk_embs_for_method,
        )
        clustering[name] = None if (
            score != score) else round(score, 4)  # nan → None

    # Jaccard (optional)
    jaccard: dict | None = None
    if args.annotations:
        logger.info("Loading annotations from %s", args.annotations)
        annotations = _load_annotations(args.annotations)
        jaccard = _compute_jaccard_scores(
            annotations, results_by_id, active_methods)

    config = {
        "n_chunks": len(chunks),
        "top_n": args.top_n,
        "adaptive_top_n": args.adaptive_top_n,
        "seed": args.seed,
        "embed_model": args.embed_model,
        "n_clusters": args.n_clusters,
        "models": models,
        "extractors": {name: ext.get_config() for name, ext in extractors.items()},
    }

    output: dict = {
        "config": config,
        "per_chunk": per_chunk_output,
        "aggregate": aggregate,
        "clustering": clustering,
    }
    if jaccard is not None:
        output["jaccard"] = jaccard

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare YAKE, KeyBERT, and OpenRouter keyword extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-chunks", type=int, default=150)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["google/gemini-flash-1.5"],
        metavar="MODEL",
        help="One or more OpenRouter model IDs to run as LLM extractors.",
    )
    # all-MiniLM-L6-v2
    parser.add_argument("--embed-model", default="all-MiniLM-L12-v2")
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--generate-annotations",
        action="store_true",
        help="Write a N-chunk annotation template JSON and exit.",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        help="Path to filled annotation JSON for Jaccard scoring.",
    )
    parser.add_argument(
        "--adaptive-top-n",
        action="store_true",
        help="Scale top_n per chunk as int(sqrt(len(text))). Overrides --top-n.",
    )
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument(
        "--chunks-path", default=None, help="Path to chunks pickle file."
    )
    parser.add_argument(
        "--meta-path", default=None, help="Path to metadata pickle file."
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        default=False,
        help="Use the partial index (index/partial_sections/) instead of the full index",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    default_chunks, default_meta = get_index_paths(args.partial)
    if args.chunks_path is None:
        args.chunks_path = default_chunks
    if args.meta_path is None:
        args.meta_path = default_meta

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.generate_annotations:
        all_chunks = load_chunks(args.chunks_path, args.meta_path)
        template_path = args.annotations or "data/annotations_template.json"
        _generate_annotation_template(
            all_chunks, n=args.n_chunks, seed=args.seed, output_path=template_path
        )
        return

    results = run_experiment(args)
    active_methods = list(results["aggregate"].keys())
    print_summary(
        results["aggregate"],
        results["clustering"],
        results.get("jaccard"),
        active_methods,
    )

    out_path = args.output or os.path.join(
        "data", f"keyword_experiment_{strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda o: (
                float(o)
                if isinstance(o, np.floating)
                else int(o)
                if hasattr(o, "__index__")
                else str(o)
            ),
        )
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    load_dotenv()
    main()
