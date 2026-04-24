import json
import pytest
from pathlib import Path
from datetime import datetime
from tests.metrics import SimilarityScorer


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_tokensmith_benchmarks(benchmarks, config, results_dir, kg_artifacts):
    """
    Run all benchmarks through the TokenSmith system.

    Args:
        benchmarks: List of benchmark dictionaries from benchmarks.yaml
        config: Merged configuration from config.yaml and CLI args
        results_dir: Directory to save results
        kg_artifacts: Pre-loaded KG data (None if --kg-run-dir not set)
    """
    # Initialize scorer with configured metrics
    scorer = SimilarityScorer(enabled_metrics=config["metrics"])

    # Print test configuration
    print_test_config(config, scorer)

    # Run each benchmark
    passed = 0
    failed = 0

    for benchmark in benchmarks:
        result = run_benchmark(benchmark, config, results_dir, scorer, kg_artifacts)
        if result["passed"]:
            passed += 1
        else:
            failed += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")


def print_test_config(config, scorer):
    """Print the test configuration in a readable format."""
    active_metrics = list(scorer._get_active_metrics().keys())
    
    print(f"\n{'='*60}")
    print("  TokenSmith Benchmark Configuration")
    print(f"{'='*60}")
    print(f"  Generator Model:    {Path(config['model_path']).name}")
    print(f"  Embedding Model:    {Path(config['embed_model']).name if '/' in config['embed_model'] else config['embed_model']}")
    # print(f"  Retrieval Method:   {config['retrieval_method']}")
    
    # if config['retrieval_method'] == 'hybrid':
    #     print(f"    • FAISS weight:   {config['faiss_weight']:.2f}")
    #     print(f"    • BM25 weight:    {config['bm25_weight']:.2f}")
    #     print(f"    • Tag weight:     {config['tag_weight']:.2f}")
    
    print(f"  System Prompt:      {config['system_prompt_mode']}")
    print(f"  Chunks Enabled:     {not config['disable_chunks']}")
    print(f"  Golden Chunks:      {config['use_golden_chunks']}")
    print(f"  HyDE Enabled:       {config.get('use_hyde', False)}")
    print(f"  Output Mode:        {config['output_mode']}")
    print(f"  Metrics:            {', '.join(active_metrics)}")
    print(f"{'='*60}\n")


def run_benchmark(benchmark, config, results_dir, scorer, kg_artifacts=None):
    """
    Run a single benchmark test.

    Returns:
        dict: Result dictionary with test outcome and metrics
    """
    benchmark_id = benchmark.get("id", "unknown")
    question = benchmark["question"]
    expected_answer = benchmark["expected_answer"]
    keywords = benchmark.get("keywords", [])
    threshold = config["threshold_override"] or benchmark["similarity_threshold"] or 0.6 
    golden_chunks = benchmark.get("golden_chunks", None)
    ideal_retrieved_chunks = benchmark.get("ideal_retrieved_chunks", None)

    # Print header
    print(f"\n{'─'*60}")
    print(f"  Benchmark: {benchmark_id}")
    print(f"  Question: {question}")
    print(f"  Threshold: {threshold}")
    print(f"{'─'*60}")
    
    # Get answer from TokenSmith
    try:
        retrieved_answer, chunks_info, hyde_query = get_tokensmith_answer(
            question=question,
            config=config,
            golden_chunks=golden_chunks if config["use_golden_chunks"] else None,
            kg_artifacts=kg_artifacts,
        )
    except Exception as e:
        import logging, traceback
        error_msg = f"Error running TokenSmith: {e}"
        print(f"  ❌ FAILED: {error_msg}")
        log_failure(results_dir, benchmark_id, error_msg)
        traceback.print_exc()
        logging.exception("Error running TokenSmith")
        return {"passed": False}

    
    # Validate answer
    if not retrieved_answer or not retrieved_answer.strip():
        error_msg = f"No answer generated for benchmark '{benchmark_id}'"
        print(f"  ❌ FAILED: {error_msg}")
        log_failure(results_dir, benchmark_id, error_msg)
        return {"passed": False}
    
    # Calculate scores
    try:
        scores = scorer.calculate_scores(retrieved_answer, expected_answer, keywords, question=question, ideal_retrieved_chunks=ideal_retrieved_chunks, actual_retrieved_chunks=chunks_info)
    except Exception as e:
        error_msg = f"Scoring error: {e}"
        print(f"  ❌ FAILED: {error_msg}")
        log_failure(results_dir, benchmark_id, error_msg)
        return {"passed": False}
    
    # Check if test passed
    final_score = scores.get("final_score", 0)
    passed = final_score >= threshold
    
    # Print result
    print_result(benchmark_id, passed, final_score, threshold, scores, config["output_mode"], retrieved_answer)
    
    # Save detailed result
    result_data = {
        "test_id": benchmark_id,
        "question": question,
        "expected_answer": expected_answer,
        "retrieved_answer": retrieved_answer,
        "keywords": keywords,
        "threshold": threshold,
        "scores": scores,
        "passed": passed,
        "active_metrics": scores.get("active_metrics", []),
        "metric_weights": get_metric_weights(scorer, scores.get("active_metrics", [])),
        "chunks_info": chunks_info if chunks_info else [],
        "hyde_query": hyde_query if hyde_query else None,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_path": config["model_path"],
            "embed_model": config["embed_model"],
            # "retrieval_method": config["retrieval_method"],
            "system_prompt_mode": config["system_prompt_mode"],
            "disable_chunks": config["disable_chunks"],
            "use_golden_chunks": config["use_golden_chunks"],
        }
    }
    
    save_result(results_dir, result_data)
    
    # Log failures
    if not passed:
        log_failure(results_dir, benchmark_id, format_failure_message(
            question, expected_answer, retrieved_answer, final_score, threshold, scores
        ))
    
    return result_data


def _get_kg_retrieved_chunks(question, config, kg_artifacts):
    """Use KG retrievers to retrieve relevant chunk texts for a query."""
    from src.knowledge_graph.query import SectionSummaryRetriever
    from src.knowledge_graph.experimental_retriever import SectionKGRetriever

    graph = kg_artifacts["graph"]
    chunks = kg_artifacts["chunks"]
    tree = kg_artifacts["tree"]
    canonical_lookup = kg_artifacts["canonical_lookup"]
    index = kg_artifacts["summary_index"]
    entries = kg_artifacts["summary_entries"]
    top_k = config.get("top_k", 10)
    num_hops = config.get("kg_num_hops", 1)
    neighbor_weight = config.get("kg_neighbor_weight", 0.5)

    retrievers = []
    if index is not None:
        retrievers.append(SectionSummaryRetriever(index, entries))
    if index is not None and tree is not None:
        retrievers.append(
            SectionKGRetriever(
                summary_index=index,
                summary_entries=entries,
                summary_embed_model="sentence-transformers/all-MiniLM-L6-v2",
                section_tree=tree,
                graph=graph,
                kg_chunks=chunks,
                chunks=list(chunks.values()),
                canonical_lookup=canonical_lookup,
                kg_neighbor_weight=neighbor_weight,
                kg_num_hops=num_hops,
            )
        )

    if not retrievers:
        return []

    all_scores: dict = {}
    for retriever in retrievers:
        scores = retriever.get_scores(question, top_k, list(chunks.values()))
        for cid, score in scores.items():
            all_scores[cid] = max(all_scores.get(cid, 0.0), score)

    sorted_ids = sorted(all_scores, key=all_scores.get, reverse=True)[:top_k]
    return [chunks[cid] for cid in sorted_ids if cid in chunks]


def get_tokensmith_answer(question, config, golden_chunks=None, kg_artifacts=None):
    """
    Get answer from TokenSmith system.

    Args:
        question: Question text
        config: Configuration dict
        golden_chunks: Optional list of golden chunks to use instead of retrieval
        kg_artifacts: Optional pre-loaded KG data for KG-based retrieval

    Returns:
        tuple: (Generated answer, chunks_info list, hyde_query)
    """
    from src.main import get_answer
    from src.instrumentation.logging import get_logger
    from src.config import RAGConfig
    from src.retriever import BM25Retriever, FAISSRetriever, IndexKeywordRetriever, load_artifacts
    from src.ranking.ranker import EnsembleRanker
    import argparse
    
    # Create a mock args namespace with our config values
    args = argparse.Namespace(
        index_prefix=config["index_prefix"],
        model_path=config.get("model_path"),
        system_prompt_mode=config.get("system_prompt_mode"),
    )

    # Create RAGConfig from our test config
    cfg = RAGConfig(
        chunk_mode=config.get("chunk_mode", "recursive_sections"),
        top_k=config.get("top_k", 10),
        embed_model=config.get("embed_model"),
        ensemble_method=config.get("retrieval_method", "rrf"),
        rrf_k=60,
        ranker_weights=config.get("ranker_weights", {"faiss": 1, "bm25": 0}),
        rerank_mode=config.get("rerank_mode", "none"),
        rerank_top_k=config.get("rerank_top_k", 5),
        system_prompt_mode=config.get("system_prompt_mode", "baseline"),
        max_gen_tokens=config.get("max_gen_tokens", 400),
        disable_chunks=config.get("disable_chunks", False),
        use_golden_chunks=config.get("use_golden_chunks", False),
        output_mode=config.get("output_mode", "html"),
        metrics=config.get("metrics", ["all"]),
        use_hyde=config.get("use_hyde", False),
        hyde_max_tokens=config.get("hyde_max_tokens", 300),
        use_indexed_chunks=config.get("use_indexed_chunks", False),
        extracted_index_path=config.get("extracted_index_path", "data/extracted_index.json"),
        page_to_chunk_map_path=config.get("page_to_chunk_map_path", "index/sections/textbook_index_page_to_chunk_map.json"),
    )
    
    # KG retrieval overrides regular retrieval when kg_artifacts is available
    # and the benchmark doesn't already supply explicit golden chunks.
    if kg_artifacts and not golden_chunks:
        kg_chunks = _get_kg_retrieved_chunks(question, config, kg_artifacts)
        if kg_chunks:
            golden_chunks = kg_chunks
            cfg = RAGConfig(
                chunk_mode=cfg.chunk_mode,
                top_k=cfg.top_k,
                embed_model=cfg.embed_model,
                ensemble_method=cfg.ensemble_method,
                rrf_k=cfg.rrf_k,
                ranker_weights=cfg.ranker_weights,
                rerank_mode=cfg.rerank_mode,
                rerank_top_k=cfg.rerank_top_k,
                system_prompt_mode=cfg.system_prompt_mode,
                max_gen_tokens=cfg.max_gen_tokens,
                disable_chunks=False,
                use_golden_chunks=True,
                output_mode=cfg.output_mode,
                metrics=cfg.metrics,
                use_hyde=False,
                hyde_max_tokens=cfg.hyde_max_tokens,
                use_indexed_chunks=False,
                extracted_index_path=cfg.extracted_index_path,
                page_to_chunk_map_path=cfg.page_to_chunk_map_path,
            )

    # Print status
    if golden_chunks and cfg.use_golden_chunks:
        source = "KG" if kg_artifacts else "golden"
        print(f"  📌 Using {len(golden_chunks)} {source} chunks")
    elif config["disable_chunks"]:
        print(f"  📭 No chunks (baseline mode)")
    else:
        if config.get("use_hyde", False):
            print(f"  🔬 HyDE enabled - generating hypothetical document...")
        print(f"  🔍 Retrieving chunks...")
    
    logger = get_logger()

    # Run the query through the main pipeline
    artifacts_dir = cfg.get_artifacts_directory()
    faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(
        artifacts_dir=artifacts_dir, 
        index_prefix=config["index_prefix"]
    )

    retrievers = [
        FAISSRetriever(faiss_index, cfg.embed_model),
        BM25Retriever(bm25_index)
    ]
    
    # Add index keyword retriever if weight > 0
    if cfg.ranker_weights.get("index_keywords", 0) > 0:
        retrievers.append(
            IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path)
        )
    
    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k)
    )
    
    # Package artifacts for reuse
    artifacts = {
        "chunks": chunks,
        "sources": sources,
        "retrievers": retrievers,
        "ranker": ranker,
        "metadata": metadata,
    }

    result = get_answer(
        question=question,
        cfg=cfg,
        args=args,
        logger=logger,
        artifacts=artifacts,
        console=None,
        golden_chunks=golden_chunks,
        is_test_mode=True
    )
    
    # Handle return value (answer, chunks_info, hyde_query) or just answer
    if isinstance(result, tuple):
        generated, chunks_info, hyde_query = result
    else:
        generated, chunks_info, hyde_query = result, None, None
    
    # Clean answer - extract up to end token if present
    generated = clean_answer(generated)
    
    return generated, chunks_info, hyde_query


def clean_answer(text):
    """
    Extract answer up to end token if present.
    
    End tokens: [end of text], </s>, <|end|>, <|endoftext|>
    """
    end_tokens = [
        "[end of text]",
        "</s>",
        "<|end|>",
        "<|endoftext|>",
        "<|im_end|>",
    ]
    
    # Find the earliest end token
    earliest_pos = len(text)
    found_token = None
    
    for token in end_tokens:
        pos = text.find(token)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
            found_token = token
    
    # Extract up to end token or return full text
    if found_token:
        return text[:earliest_pos].strip()
    
    return text.strip()


def print_result(benchmark_id, passed, final_score, threshold, scores, output_mode, retrieved_answer=None):
    """Print test result based on output mode."""
    if output_mode == "terminal":
        # Detailed terminal output
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"\n  {status}")
        print(f"  Final Score: {final_score:.3f} (threshold: {threshold:.3f})")
        
        # Show metric breakdown
        active_metrics = scores.get("active_metrics", [])
        if len(active_metrics) > 1:
            print(f"  Metric Breakdown:")
            for metric in active_metrics:
                metric_score = scores.get(f"{metric}_similarity", 0)
                print(f"    • {metric:12} : {metric_score:.3f}")
        
        keywords_matched = scores.get("keywords_matched", 0)
        total_keywords = len(scores.get("keywords", []))
        if total_keywords > 0:
            print(f"    • keywords    : {keywords_matched}/{total_keywords}")
        
        # Display retrieved answer
        if retrieved_answer:
            print(f"\n  📝 Retrieved Answer:")
            print(f"  {'-'*58}")
            for line in retrieved_answer.split('\n'):
                print(f"  {line}")
            print(f"  {'-'*58}")
    else:
        # Compact output for HTML mode
        status = "✅" if passed else "❌"
        print(f"  {status} Score: {final_score:.3f} (threshold: {threshold:.3f})")


def get_metric_weights(scorer, active_metric_names):
    """Get weights for active metrics."""
    weights = {}
    for name in active_metric_names:
        metric = scorer.registry.get_metric(name)
        if metric:
            weights[name] = metric.weight
    return weights


def save_result(results_dir, result_data):
    """Save benchmark result to JSON file (one result per line)."""
    results_file = results_dir / "benchmark_results.json"
    with open(results_file, "a") as f:
        json.dump(result_data, f, indent=None, ensure_ascii=False, default=str)
        f.write("\n")


def log_failure(results_dir, benchmark_id, message):
    """Log benchmark failure to dedicated log file."""
    failed_log = results_dir / "failed_tests.log"
    with open(failed_log, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"BENCHMARK FAILURE: {benchmark_id}\n")
        f.write(f"{'='*60}\n")
        f.write(f"{message}\n")
        f.write(f"{'='*60}\n")


def format_failure_message(question, expected, retrieved, final_score, threshold, scores):
    """Create detailed failure message."""
    lines = [
        f"Question: {question}",
        f"",
        f"Expected Answer:",
        f"{expected}",
        f"",
        f"Retrieved Answer:",
        f"{retrieved}",
        f"",
        f"Final Score: {final_score:.3f} (threshold: {threshold:.3f})",
        f"Active Metrics: {', '.join(scores.get('active_metrics', []))}",
        f"",
        f"Individual Metric Scores:",
    ]
    
    for metric in scores.get("active_metrics", []):
        metric_score = scores.get(f"{metric}_similarity", 0)
        lines.append(f"  {metric}: {metric_score:.3f}")
    
    keywords_matched = scores.get("keywords_matched", 0)
    total_keywords = len(scores.get("keywords", []))
    if total_keywords > 0:
        lines.append(f"  keywords: {keywords_matched}/{total_keywords}")
    
    return "\n".join(lines)
