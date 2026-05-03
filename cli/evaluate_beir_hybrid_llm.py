#!/usr/bin/env python3
"""
Evaluate BM25 + BERT + LLM on the BEIR SciFact benchmark.

Pipeline per query
──────────────────
1. BM25 search        → top RETRIEVAL_POOL candidates   (InvertedIndex)
2. BERT chunk search  → top RETRIEVAL_POOL candidates   (ChunkedSemanticSearch)
3. RRF fusion         → single ranked list              (rrf_combine_search_results)
4. Reranking          → final ordered list
   --rerank none           keep RRF order (BM25+BERT baseline for comparison)
   --rerank cross_encoder  ms-marco-MiniLM-L6-v2, fast, no API calls
   --rerank batch          one LLM call per query (300 calls, OPENROUTER_API_KEY required)
   --rerank individual     one LLM call per candidate (expensive, use small --rerank-pool)

After reranking, the top --rerank-pool documents are replaced by the reranker's
order and the remainder of the RETRIEVAL_POOL keeps its original RRF order.
This preserves list length so Recall@100 is still meaningful.

Metrics: MRR, NDCG, Recall, Precision, F1 at k = 1, 5, 10, 20, 100

Output
──────
Metrics printed to stdout.
Ranked lists saved to: <dataset>/bm25_bert_llm_results_<method>.json
  Same [[query_id, [doc_ids...]], ...] format as the other result files.
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

# Allow imports from the cli/ package when the script is run directly.
sys.path.insert(0, str(Path(__file__).parent))

from lib.beir_loader import load_beir_corpus, load_beir_queries, load_beir_qrels
from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from lib.hybrid_search import rrf_combine_search_results
from lib.rerank import individual_rerank, batch_rerank, cross_encoder_rerank
from lib.search_utils import BEIR_CACHE_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

RETRIEVAL_POOL = 100  # candidates retrieved by each sub-system before RRF
RERANK_POOL = 25  # top-RRF docs passed to the reranker
# remaining (RETRIEVAL_POOL - RERANK_POOL) keep RRF order

# ── Index / embedding helpers ─────────────────────────────────────────────────


def build_or_load_bm25_index(
    corpus: list[dict],
    cache_dir: Path,
) -> InvertedIndex:
    """Return a ready-to-use InvertedIndex, building and saving it if not cached."""
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
        "  %d documents  |  %.1f avg tokens/doc", len(idx.docmap), idx._avg_doc_length
    )
    return idx


def build_or_load_bert_embeddings(
    corpus: list[dict],
    cache_dir: Path,
) -> ChunkedSemanticSearch:
    """Return a ready-to-use ChunkedSemanticSearch, building embeddings if not cached."""
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


# ── Retrieval + reranking ─────────────────────────────────────────────────────


def retrieve_rrf(
    query: str,
    idx: InvertedIndex,
    css: ChunkedSemanticSearch,
    pool: int,
) -> list[dict]:
    """Run BM25 + BERT and fuse with RRF.  Returns up to pool results."""
    bm25_hits = idx.bm25_search(query, limit=pool)
    bert_hits = css.search_chunks(query, limit=pool)
    return rrf_combine_search_results(bm25_hits, bert_hits, k=60, limit=pool)


def apply_reranking(
    query: str,
    rrf_results: list[dict],
    method: str,
    rerank_pool: int,
    final_limit: int,
) -> list[dict]:
    """
    Rerank the top rerank_pool documents, then append the remainder in their
    original RRF order. This keeps the full list length intact for Recall@100.

    Documents inside the rerank pool get an extra score field:
      cross_encoder → "cross_encoder_score"
      batch / individual → "rerank_score"
    Documents outside the pool have no extra field.
    """
    top = rrf_results[:rerank_pool]
    rest = rrf_results[rerank_pool:]

    if method == "cross_encoder":
        reranked_top = cross_encoder_rerank(query, top, limit=None)
    elif method == "batch":
        reranked_top = batch_rerank(query, top, limit=None)
    elif method == "individual":
        reranked_top = individual_rerank(query, top, limit=None)
    else:
        raise ValueError(f"Unknown reranking method: '{method}'")

    return (reranked_top + rest)[:final_limit]


# ── Metric functions ───────────────────────────────────────────────────────────
# Identical to those in the BERT and BM25+BERT evaluate.py files so results
# are directly comparable.


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


def print_metrics_table(
    ranked: dict,
    qrels: dict,
    label: str,
) -> None:
    ks = [1, 5, 10, 20, 100]
    print(f"\n{'═' * 70}")
    print(f"  {label}")
    print(f"  Corpus: 5183 docs  |  Queries evaluated: {len(ranked)}")
    print(f"{'═' * 70}")
    print(f"\n{'Metric':<15}" + "".join(f"{'@'+str(k):>10}" for k in ks))
    print("-" * 65)
    for name, fn in [
        ("MRR", mrr_at_k),
        ("NDCG", ndcg_at_k),
        ("Recall", recall_at_k),
        ("Precision", precision_at_k),
        ("F1", f1_at_k),
    ]:
        print(f"{name:<15}" + "".join(f"{fn(ranked, qrels, k):>10.4f}" for k in ks))
    print(f"\nTotal queries evaluated: {len(ranked)}")


# ── Save results ───────────────────────────────────────────────────────────────


def save_results(ranked: dict, output_path: Path) -> None:
    """Save ranked results in the same format as the other result files.

    Format: [[query_id, [doc_id, doc_id, ...]], ...]
    Compatible with tfidf_results.json, bm25_results.json, bert_results.json,
    bm25_bert_results.json.
    """
    results_list = [[qid, doc_ids] for qid, doc_ids in ranked.items()]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)
    logger.info("Results saved to %s", output_path)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 + BERT + LLM on a BEIR dataset.",
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
        "--rerank",
        choices=["none", "cross_encoder", "batch", "individual"],
        default="cross_encoder",
        help=(
            "Reranking method applied after RRF fusion. "
            "'none' gives the BM25+BERT baseline. "
            "'cross_encoder' is fastest (no API calls). "
            "'batch' makes one LLM call per query (300 calls). "
            "'individual' makes one LLM call per candidate per query (expensive)."
        ),
    )
    parser.add_argument(
        "--rerank-pool",
        type=int,
        default=RERANK_POOL,
        help=(
            "Number of top-RRF documents passed to the reranker. "
            "The rest keep their RRF order. "
            "Reduce this for LLM methods to limit API calls."
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

    logger.info("Loading queries ...")
    queries = load_beir_queries(dataset_path / "queries.jsonl")
    logger.info("  %d queries.", len(queries))

    logger.info("Loading qrels (%s split) ...", args.split)
    qrels = load_beir_qrels(dataset_path / "qrels" / f"{args.split}.tsv")
    logger.info("  %d queries have relevance judgements.", len(qrels))

    # ── Build / load retrieval components ─────────────────────────────────────
    idx = build_or_load_bm25_index(corpus, cache_dir)
    css = build_or_load_bert_embeddings(corpus, cache_dir)

    # ── Warn about LLM API cost ────────────────────────────────────────────────
    queries_to_eval = {qid: t for qid, t in queries.items() if qid in qrels}
    n_queries = len(queries_to_eval)

    if args.rerank == "batch":
        logger.warning(
            "batch reranking will make %d LLM API calls (one per query). "
            "OPENROUTER_API_KEY must be set.",
            n_queries,
        )
    elif args.rerank == "individual":
        logger.warning(
            "individual reranking will make %d LLM API calls "
            "(%d queries × %d candidates). "
            "Consider --rerank batch or --rerank cross_encoder instead.",
            n_queries * args.rerank_pool,
            n_queries,
            args.rerank_pool,
        )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    logger.info(
        "Evaluating %d queries  |  rerank=%s  |  rerank_pool=%d ...",
        n_queries,
        args.rerank,
        args.rerank_pool,
    )

    ranked_results: dict[str, list[str]] = {}

    for qid, qtext in tqdm(queries_to_eval.items(), desc="Evaluating"):
        rrf_hits = retrieve_rrf(qtext, idx, css, pool=RETRIEVAL_POOL)

        if args.rerank == "none":
            final_hits = rrf_hits
        else:
            final_hits = apply_reranking(
                query=qtext,
                rrf_results=rrf_hits,
                method=args.rerank,
                rerank_pool=args.rerank_pool,
                final_limit=RETRIEVAL_POOL,
            )

        # Extract doc IDs in final rank order.
        # rrf_combine_search_results stores the id under "doc_id".
        ranked_results[qid] = [hit["doc_id"] for hit in final_hits]

    # ── Print metrics table ────────────────────────────────────────────────────
    method_labels = {
        "none": "BM25 + BERT (RRF, no reranking)          — BEIR SciFact",
        "cross_encoder": "BM25 + BERT + Cross-Encoder              — BEIR SciFact",
        "batch": "BM25 + BERT + LLM (batch reranking)      — BEIR SciFact",
        "individual": "BM25 + BERT + LLM (individual reranking) — BEIR SciFact",
    }
    print_metrics_table(ranked_results, qrels, label=method_labels[args.rerank])

    # ── Save results ───────────────────────────────────────────────────────────
    output_path = dataset_path / f"bm25_bert_llm_results_{args.rerank}.json"
    save_results(ranked_results, output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
