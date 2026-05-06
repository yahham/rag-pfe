#!/usr/bin/env python3
"""
Evaluate RAG on the BEIR SciFact benchmark.

Pipeline per query
──────────────────
1. Retrieval:  BM25 + BERT (RRF) → top RETRIEVAL_POOL documents
               (reuses the same index and embedding cache as evaluate_beir_hybrid_llm.py)
2. Context:    top CONTEXT_DOCS documents passed to the LLM
3. Generation: LLM produces an answer grounded in those documents

Metrics
───────
Standard IR metrics on the retrieval step (identical to the other evaluators):
  MRR, NDCG, Recall, Precision, F1  at k = 1, 5, 10, 20, 100

RAG-specific metrics:
  Context Precision@k
    What fraction of the k retrieved documents are relevant according to qrels?
    Measures the signal-to-noise ratio of the context given to the LLM.
    Identical to Precision@k but named to reflect its role in the RAG pipeline.

  Context Recall@k
    What fraction of all relevant documents appear in the top k?
    Measures whether the LLM has access to everything it needs to answer correctly.
    Identical to Recall@k but named to reflect its role in the RAG pipeline.

  Faithfulness  (requires LLM API calls)
    Are the factual claims in the generated answer supported by the retrieved
    context? Scored by an LLM judge on a per-query basis, then averaged.
    Range: 0.0 (all claims unsupported) to 1.0 (all claims supported).
    Computed on --faithfulness-sample queries (default: all 300) to allow
    limiting API usage during development.

Output
──────
Metrics table printed to stdout.
Generated answers saved to: <dataset>/rag_answers.json
  Format: [[query_id, query_text, [retrieved_doc_ids], generated_answer], ...]
  This file is the input needed to run RAGAS or other external evaluators.
"""

import os
import sys
import json
import math
import logging
import argparse
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from lib.beir_loader import load_beir_corpus, load_beir_queries, load_beir_qrels
from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from lib.hybrid_search import rrf_combine_search_results
from lib.llm import (
    answer_scientific,
    faithfulness_judge,
    generate_questions_from_answer,
)
from lib.search_utils import BEIR_CACHE_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

RETRIEVAL_POOL = 100  # docs retrieved by each sub-system before RRF
CONTEXT_DOCS = 5  # top-K docs passed to the LLM for answer generation

# ── Build / load retrieval components ─────────────────────────────────────────
# Intentionally identical to evaluate_beir_hybrid_llm.py so both scripts
# share the same on-disk cache.


def build_or_load_bm25_index(corpus: list[dict], cache_dir: Path) -> InvertedIndex:
    idx = InvertedIndex(cache_dir=cache_dir)
    all_cached = all(
        p.exists()
        for p in (
            idx.index_path,
            idx.docmap_path,
            idx.term_frequencies_path,
            idx.doc_lengths_path,
        )
    )
    if all_cached:
        logger.info("Loading BM25 index from cache (%s) ...", cache_dir)
        idx.load()
    else:
        logger.info(
            "Building BM25 index for %d documents (cache: %s) ...",
            len(corpus),
            cache_dir,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        idx.build(corpus)
        idx.save()
        logger.info("BM25 index saved.")
    logger.info(
        "  %d documents  |  %.1f avg tokens/doc",
        len(idx.docmap),
        idx._avg_doc_length,
    )
    return idx


def build_or_load_bert_embeddings(
    corpus: list[dict], cache_dir: Path
) -> ChunkedSemanticSearch:
    css = ChunkedSemanticSearch(cache_dir=cache_dir)
    if css.chunk_embeddings_path.exists() and css.chunk_metadata_path.exists():
        logger.info("Loading BERT chunk embeddings from cache (%s) ...", cache_dir)
    else:
        logger.info(
            "Building BERT chunk embeddings for %d documents (cache: %s) ...",
            len(corpus),
            cache_dir,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
    css.load_or_create_embeddings(corpus)
    logger.info(
        "  %d chunk vectors  |  %d-dim",
        len(css.chunk_embeddings),
        css.chunk_embeddings.shape[1],
    )
    return css


# ── Retrieval ──────────────────────────────────────────────────────────────────


def retrieve_rrf(
    query: str,
    idx: InvertedIndex,
    css: ChunkedSemanticSearch,
    pool: int,
) -> list[dict]:
    """Run BM25 + BERT and fuse with RRF. Returns up to pool results."""
    bm25_hits = idx.bm25_search(query, limit=pool)
    bert_hits = css.search_chunks(query, limit=pool)
    return rrf_combine_search_results(bm25_hits, bert_hits, k=60, limit=pool)


# ── Standard IR metrics ────────────────────────────────────────────────────────


def mrr_at_k(ranked: dict, qrels: dict, k: int) -> float:
    total, n = 0.0, 0
    for qid, doc_ids in ranked.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        n += 1
        for rank, did in enumerate(doc_ids[:k], start=1):
            if did in rel:
                total += 1.0 / rank
                break
    return total / n if n else 0.0


def ndcg_at_k(ranked: dict, qrels: dict, k: int) -> float:
    total, n = 0.0, 0
    for qid, doc_ids in ranked.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        n += 1
        dcg = sum(
            rel.get(did, 0) / math.log2(i + 2) for i, did in enumerate(doc_ids[:k])
        )
        idcg = sum(
            r / math.log2(i + 2)
            for i, r in enumerate(sorted(rel.values(), reverse=True)[:k])
        )
        if idcg > 0:
            total += dcg / idcg
    return total / n if n else 0.0


def recall_at_k(ranked: dict, qrels: dict, k: int) -> float:
    total, n = 0.0, 0
    for qid, doc_ids in ranked.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        n += 1
        hits = sum(1 for did in doc_ids[:k] if did in rel)
        total += hits / len(rel)
    return total / n if n else 0.0


def precision_at_k(ranked: dict, qrels: dict, k: int) -> float:
    total, n = 0.0, 0
    for qid, doc_ids in ranked.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        n += 1
        top_k = doc_ids[:k]
        if top_k:
            total += sum(1 for did in top_k if did in rel) / len(top_k)
    return total / n if n else 0.0


def f1_at_k(ranked: dict, qrels: dict, k: int) -> float:
    p = precision_at_k(ranked, qrels, k)
    r = recall_at_k(ranked, qrels, k)
    return 2 * p * r / (p + r) if (p + r) else 0.0


# ── RAG-specific metrics ───────────────────────────────────────────────────────


def context_precision_at_k(ranked: dict, qrels: dict, k: int) -> float:
    """Fraction of the k retrieved documents that are relevant.

    Identical to Precision@k but named to reflect its role in RAG evaluation:
    it measures how much noise is in the context window given to the LLM.
    A low score means the LLM receives mostly irrelevant documents, which
    increases the risk of hallucination or distracted generation.
    """
    return precision_at_k(ranked, qrels, k)


def context_recall_at_k(ranked: dict, qrels: dict, k: int) -> float:
    """Fraction of all relevant documents that appear in the top k.

    Identical to Recall@k but named to reflect its role in RAG evaluation:
    it measures whether the context window given to the LLM contains all the
    information needed to produce a faithful, complete answer.
    """
    return recall_at_k(ranked, qrels, k)


def compute_faithfulness(
    queries_to_eval: dict,
    ranked: dict,
    corpus_map: dict,
    context_k: int,
    sample: int | None,
) -> tuple[float, int]:
    """Compute average faithfulness over a sample of queries.

    For each query:
      1. Retrieve the top context_k documents from the ranked list.
      2. Generate an answer with the LLM.
      3. Ask the LLM judge whether the answer's claims are supported by
         the retrieved context.
      4. Record the faithfulness score (supported / total claims).

    Returns (mean_faithfulness, n_evaluated).
    """
    query_ids = list(queries_to_eval.keys())
    if sample is not None:
        query_ids = query_ids[:sample]

    scores = []
    skipped = 0

    for qid in tqdm(query_ids, desc="Faithfulness"):
        qtext = queries_to_eval[qid]
        doc_ids = ranked.get(qid, [])[:context_k]

        # Build document dicts from the corpus map for the LLM functions.
        context_docs = [corpus_map[did] for did in doc_ids if did in corpus_map]
        if not context_docs:
            skipped += 1
            continue

        # Generate an answer grounded in the retrieved context.
        try:
            answer = answer_scientific(qtext, context_docs)
        except RuntimeError as e:
            logger.warning("Generation failed for query %s: %s", qid, e)
            skipped += 1
            continue

        # Judge faithfulness of the generated answer.
        result = faithfulness_judge(qtext, context_docs, answer)
        if result is None:
            logger.warning(
                "Faithfulness judge returned None for query %s. Skipping.", qid
            )
            skipped += 1
            continue

        scores.append(result["faithfulness_score"])

    if skipped:
        logger.warning(
            "Faithfulness: skipped %d/%d queries (generation or parse failure).",
            skipped,
            len(query_ids),
        )

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return mean_score, len(scores)


def compute_answer_relevance(
    queries_to_eval: dict,
    answers_map: dict[str, tuple[list[str], str]],
    n_questions: int = 3,
    sample: int | None = None,
) -> tuple[float, int]:
    """Compute RAGAS-style answer relevance over a sample of queries.

    For each (question, answer) pair:
      1. Generate n_questions synthetic questions from the answer via LLM.
         The intuition: if the answer actually addresses the question, the
         questions it implies should be semantically close to the original.
      2. Embed the original question and the synthetic questions using the
         same sentence-transformers model used for retrieval.
      3. Compute the cosine similarity between the original embedding and
         each synthetic embedding, then average the similarities.

    A score near 1.0 means the answer is tightly on-topic.
    A score near 0.0 means the answer addresses a different question entirely.

    Returns (mean_relevance_score, n_evaluated).
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # Use the same model as the retrieval pipeline for consistency.
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    query_ids = list(queries_to_eval.keys())
    if sample is not None:
        query_ids = query_ids[:sample]

    scores = []
    skipped = 0

    for qid in tqdm(query_ids, desc="Answer relevance"):
        qtext = queries_to_eval[qid]
        _, answer = answers_map[qid]

        if not answer:
            skipped += 1
            continue

        # Step 1 — generate synthetic questions from the answer.
        synthetic_qs = generate_questions_from_answer(answer, n=n_questions)
        if not synthetic_qs:
            skipped += 1
            continue

        # Step 2 — embed original and synthetic questions.
        original_emb = embed_model.encode(qtext, normalize_embeddings=True)
        synthetic_emb = embed_model.encode(synthetic_qs, normalize_embeddings=True)

        # Step 3 — cosine similarity (dot product since vectors are unit-normed).
        similarities = np.dot(synthetic_emb, original_emb)
        scores.append(float(np.mean(similarities)))

    if skipped:
        logger.warning(
            "Answer relevance: skipped %d/%d queries "
            "(empty answer or LLM parse failure).",
            skipped,
            len(query_ids),
        )

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return mean_score, len(scores)


# ── Output helpers ─────────────────────────────────────────────────────────────


def print_ir_metrics(ranked: dict, qrels: dict) -> None:
    ks = [1, 5, 10, 20, 100]
    print(f"\n── Retrieval Metrics {'─' * 46}")
    print(f"\n{'Metric':<18}" + "".join(f"{'@'+str(k):>10}" for k in ks))
    print("-" * 68)
    for name, fn in [
        ("MRR", mrr_at_k),
        ("NDCG", ndcg_at_k),
        ("Recall", recall_at_k),
        ("Precision", precision_at_k),
        ("F1", f1_at_k),
    ]:
        print(f"{name:<18}" + "".join(f"{fn(ranked, qrels, k):>10.4f}" for k in ks))


def print_rag_metrics(ranked: dict, qrels: dict) -> None:
    ks = [1, 5, 10, 20, 100]
    print(f"\n── RAG-Specific Metrics {'─' * 43}")
    print(f"\n{'Metric':<25}" + "".join(f"{'@'+str(k):>10}" for k in ks))
    print("-" * 75)
    for name, fn in [
        ("Context Precision", context_precision_at_k),
        ("Context Recall", context_recall_at_k),
    ]:
        print(f"{name:<25}" + "".join(f"{fn(ranked, qrels, k):>10.4f}" for k in ks))


def save_answers(
    records: list[tuple[str, str, list[str], str]],
    output_path: Path,
) -> None:
    """Save generated answers to JSON.

    Each record is [query_id, query_text, [doc_id, ...], answer].
    This format is compatible with RAGAS and other external evaluators.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info("Answers saved to %s", output_path)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG on a BEIR dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv(
            "BEIR_DATASET_PATH",
            str(Path(__file__).parents[2] / "datasets" / "scifact"),
        ),
        help="Path to the BEIR dataset directory.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Qrels split to evaluate on.",
    )
    parser.add_argument(
        "--context-docs",
        type=int,
        default=CONTEXT_DOCS,
        help=(
            "Number of top-retrieved documents passed to the LLM for generation. "
            "Also the k used for Context Precision and Context Recall."
        ),
    )
    parser.add_argument(
        "--faithfulness-sample",
        type=int,
        default=None,
        help=(
            "Number of queries to evaluate for faithfulness. "
            "Each query makes two LLM API calls (generation + judge). "
            "Defaults to all queries. Use a smaller value (e.g. 50) during "
            "development to limit API usage."
        ),
    )
    parser.add_argument(
        "--skip-faithfulness",
        action="store_true",
        help=(
            "Skip faithfulness evaluation entirely. Useful when running without "
            "an API key or when only retrieval metrics are needed."
        ),
    )
    parser.add_argument(
        "--answer-relevance-sample",
        type=int,
        default=None,
        help=(
            "Number of queries to evaluate for answer relevance. "
            "Each query makes one LLM call (synthetic question generation) "
            "plus local embedding inference. "
            "Defaults to all queries. Use a smaller value during development."
        ),
    )
    parser.add_argument(
        "--skip-answer-relevance",
        action="store_true",
        help=(
            "Skip answer relevance evaluation entirely. "
            "Useful when running without an API key or when only retrieval "
            "and faithfulness metrics are needed."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(BEIR_CACHE_PATH),
        help="Directory for BM25 index and BERT embedding cache files.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    cache_dir = Path(args.cache_dir)

    # ── Load BEIR data ─────────────────────────────────────────────────────────
    logger.info("Loading BEIR corpus ...")
    corpus = load_beir_corpus(dataset_path / "corpus.jsonl")
    logger.info("  %d documents.", len(corpus))

    # Build a doc_id → doc dict for fast lookup during faithfulness evaluation.
    corpus_map = {doc["id"]: doc for doc in corpus}

    logger.info("Loading queries ...")
    queries = load_beir_queries(dataset_path / "queries.jsonl")
    logger.info("  %d queries.", len(queries))

    logger.info("Loading qrels (%s split) ...", args.split)
    qrels = load_beir_qrels(dataset_path / "qrels" / f"{args.split}.tsv")
    logger.info("  %d queries have relevance judgements.", len(qrels))

    # ── Build / load retrieval components ─────────────────────────────────────
    idx = build_or_load_bm25_index(corpus, cache_dir)
    css = build_or_load_bert_embeddings(corpus, cache_dir)

    # ── Retrieval pass ─────────────────────────────────────────────────────────
    queries_to_eval = {qid: t for qid, t in queries.items() if qid in qrels}
    logger.info("Running retrieval for %d queries ...", len(queries_to_eval))

    ranked_results: dict[str, list[str]] = {}

    for qid, qtext in tqdm(queries_to_eval.items(), desc="Retrieving"):
        hits = retrieve_rrf(qtext, idx, css, pool=RETRIEVAL_POOL)
        ranked_results[qid] = [hit["doc_id"] for hit in hits]

    # ── Generation pass ────────────────────────────────────────────────────────
    # Generate answers for all evaluated queries.
    # These are needed both for the output file and for faithfulness evaluation.
    logger.info(
        "Generating answers for %d queries (context: top %d docs) ...",
        len(queries_to_eval),
        args.context_docs,
    )
    answer_records: list[tuple[str, str, list[str], str]] = []

    for qid, qtext in tqdm(queries_to_eval.items(), desc="Generating"):
        doc_ids = ranked_results.get(qid, [])[: args.context_docs]
        context_docs = [corpus_map[did] for did in doc_ids if did in corpus_map]

        try:
            answer = answer_scientific(qtext, context_docs)
        except RuntimeError as e:
            logger.warning("Generation failed for query %s: %s", qid, e)
            answer = ""

        answer_records.append((qid, qtext, doc_ids, answer))

    # ── Faithfulness evaluation ────────────────────────────────────────────────
    faithfulness_score: float | None = None
    faithfulness_n: int = 0

    if not args.skip_faithfulness:
        n_calls = (
            args.faithfulness_sample
            if args.faithfulness_sample is not None
            else len(queries_to_eval)
        )
        logger.info(
            "Evaluating faithfulness on %d queries "
            "(%d LLM API calls per query: 1 generation already done + 1 judge) ...",
            n_calls,
            1,  # generation is done above; only the judge call is new
        )
        # Use the already-generated answers to avoid a second generation call.
        # Build a map from qid to (context_docs, answer) for the judge.
        answers_map = {rec[0]: (rec[2], rec[3]) for rec in answer_records}
        scores = []
        skipped = 0
        sample_ids = list(queries_to_eval.keys())
        if args.faithfulness_sample is not None:
            sample_ids = sample_ids[: args.faithfulness_sample]

        for qid in tqdm(sample_ids, desc="Faithfulness"):
            doc_ids, answer = answers_map[qid]
            if not answer:
                skipped += 1
                continue
            context_docs = [corpus_map[did] for did in doc_ids if did in corpus_map]
            result = faithfulness_judge(queries_to_eval[qid], context_docs, answer)
            if result is None:
                skipped += 1
                continue
            scores.append(result["faithfulness_score"])

        if skipped:
            logger.warning(
                "Faithfulness: skipped %d/%d queries.", skipped, len(sample_ids)
            )
        faithfulness_score = sum(scores) / len(scores) if scores else 0.0
        faithfulness_n = len(scores)

    # ── Answer relevance evaluation ────────────────────────────────────────────
    # Build answers_map here so both faithfulness and answer relevance can use it.
    # answers_map[qid] = (doc_ids, generated_answer)
    answers_map = {rec[0]: (rec[2], rec[3]) for rec in answer_records}

    answer_relevance_score: float | None = None
    answer_relevance_n: int = 0

    if not args.skip_answer_relevance:
        n_ar_calls = (
            args.answer_relevance_sample
            if args.answer_relevance_sample is not None
            else len(queries_to_eval)
        )
        logger.info(
            "Evaluating answer relevance on %d queries "
            "(%d LLM calls for synthetic question generation + local embedding) ...",
            n_ar_calls,
            n_ar_calls,
        )
        answer_relevance_score, answer_relevance_n = compute_answer_relevance(
            queries_to_eval=queries_to_eval,
            answers_map=answers_map,
            n_questions=3,
            sample=args.answer_relevance_sample,
        )

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  RAG Evaluation on BEIR SciFact")
    print(f"  Retrieval: BM25 + BERT (RRF, k=60)")
    print(f"  Generation: {args.context_docs} context docs → LLM")
    print(f"  Corpus: {len(corpus)} docs  |  Queries evaluated: {len(ranked_results)}")
    print(f"{'═' * 70}")

    print_ir_metrics(ranked_results, qrels)
    print_rag_metrics(ranked_results, qrels)

    print(f"\n── Faithfulness {'─' * 51}")
    if faithfulness_score is not None:
        print(
            f"\n  Faithfulness:  {faithfulness_score:.4f}"
            f"  (evaluated on {faithfulness_n} queries)"
        )
        if faithfulness_n < len(queries_to_eval):
            print(
                f"  Note: faithfulness was sampled. "
                f"Run without --faithfulness-sample to evaluate all "
                f"{len(queries_to_eval)} queries."
            )
    else:
        print("\n  Faithfulness: skipped (--skip-faithfulness was set).")

    if answer_relevance_score is not None:
        print(
            f"  Answer Relevance:  {answer_relevance_score:.4f}"
            f"  (evaluated on {answer_relevance_n} queries)"
        )
    else:
        print("  Answer Relevance:  skipped (--skip-answer-relevance was set).")

    print(f"\nTotal queries evaluated: {len(ranked_results)}")

    # ── Save generated answers ─────────────────────────────────────────────────
    output_path = dataset_path / "rag_answers.json"
    save_answers(answer_records, output_path)
    print(f"Generated answers saved to: {output_path}")


if __name__ == "__main__":
    main()
