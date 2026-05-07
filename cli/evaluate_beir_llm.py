#!/usr/bin/env python3
"""
Evaluate a standalone LLM (no retrieval) on the BEIR SciFact benchmark.

Pipeline per query
──────────────────
Query → LLM → Answer

No retrieval step. The LLM answers from parametric knowledge alone.

Metrics
───────
Standard IR metrics (MRR, NDCG, Recall, Precision, F1) are NOT computed
because there is no ranked document list to compare against qrels.

Answer Relevance  (requires LLM API calls)
  Measures whether the generated answer addresses the question.
  Uses the same RAGAS-style approach as the RAG evaluator:
    1. Generate N synthetic questions from the answer via LLM.
    2. Embed the original and synthetic questions.
    3. Average cosine similarities.
  Computed on --sample queries (default: all 300).

This gives a direct comparison point for RAG:
  LLM alone   → Answer Relevance only (no retrieval quality)
  RAG         → Retrieval metrics + Faithfulness + Answer Relevance

Output
──────
Metrics printed to stdout.
Generated answers saved to: <dataset>/llm_only_answers.json
  Format: [[query_id, query_text, answer], ...]
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from lib.beir_loader import load_beir_queries, load_beir_qrels
from lib.llm import answer_llm_only, generate_questions_from_answer
from lib.search_utils import BEIR_DATASET_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Answer Relevance ───────────────────────────────────────────────────────────


def compute_answer_relevance(
    queries_to_eval: dict,
    answers_map: dict[str, str],
    n_questions: int = 3,
    sample: int | None = None,
) -> tuple[float, int]:
    """Compute RAGAS-style answer relevance for LLM-only answers.

    Identical algorithm to the RAG evaluator, but answers_map here maps
    qid → answer string directly (no retrieved docs tuple).
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    query_ids = list(queries_to_eval.keys())
    if sample is not None:
        query_ids = query_ids[:sample]

    scores = []
    skipped = 0

    for qid in tqdm(query_ids, desc="Answer relevance"):
        qtext = queries_to_eval[qid]
        answer = answers_map.get(qid, "")

        if not answer:
            skipped += 1
            continue

        synthetic_qs = generate_questions_from_answer(answer, n=n_questions)
        if not synthetic_qs:
            skipped += 1
            continue

        original_emb = embed_model.encode(qtext, normalize_embeddings=True)
        synthetic_emb = embed_model.encode(synthetic_qs, normalize_embeddings=True)
        similarities = np.dot(synthetic_emb, original_emb)
        scores.append(float(np.mean(similarities)))

    if skipped:
        logger.warning(
            "Answer relevance: skipped %d/%d queries.",
            skipped,
            len(query_ids),
        )

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return mean_score, len(scores)


# ── Save answers ───────────────────────────────────────────────────────────────


def save_answers(
    records: list[tuple[str, str, str]],
    output_path: Path,
) -> None:
    """Save generated answers to JSON.

    Each record is [query_id, query_text, answer].
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info("Answers saved to %s", output_path)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a standalone LLM (no retrieval) on a BEIR dataset.",
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
        help="Qrels split to identify which queries to evaluate on.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Number of queries to generate answers for. "
            "Defaults to all queries that have qrels (300 for SciFact). "
            "Each query makes one generation call and one synthetic "
            "question generation call (two API calls total)."
        ),
    )
    parser.add_argument(
        "--skip-answer-relevance",
        action="store_true",
        help="Skip answer relevance evaluation. Only generate and save answers.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    # ── Load queries and qrels ─────────────────────────────────────────────────
    logger.info("Loading queries ...")
    queries = load_beir_queries(dataset_path / "queries.jsonl")
    logger.info("  %d queries loaded.", len(queries))

    logger.info("Loading qrels (%s split) ...", args.split)
    qrels = load_beir_qrels(dataset_path / "qrels" / f"{args.split}.tsv")
    logger.info("  %d queries have relevance judgements.", len(qrels))

    queries_to_eval = {qid: t for qid, t in queries.items() if qid in qrels}
    if args.sample is not None:
        queries_to_eval = dict(list(queries_to_eval.items())[: args.sample])

    logger.info(
        "Generating answers for %d queries (no retrieval, LLM only) ...",
        len(queries_to_eval),
    )

    # ── Generation pass ────────────────────────────────────────────────────────
    answer_records: list[tuple[str, str, str]] = []
    answers_map: dict[str, str] = {}

    for qid, qtext in tqdm(queries_to_eval.items(), desc="Generating"):
        try:
            answer = answer_llm_only(qtext)
        except RuntimeError as e:
            logger.warning("Generation failed for query %s: %s", qid, e)
            answer = ""
        answer_records.append((qid, qtext, answer))
        answers_map[qid] = answer

    # ── Answer relevance ───────────────────────────────────────────────────────
    answer_relevance_score: float | None = None
    answer_relevance_n: int = 0

    if not args.skip_answer_relevance:
        logger.info(
            "Evaluating answer relevance on %d queries ...",
            len(queries_to_eval),
        )
        answer_relevance_score, answer_relevance_n = compute_answer_relevance(
            queries_to_eval=queries_to_eval,
            answers_map=answers_map,
            n_questions=3,
            sample=None,  # already sliced by --sample above
        )

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  LLM-Only Evaluation on BEIR SciFact  (no retrieval)")
    print(f"  Model: parametric knowledge, no context provided")
    print(f"  Queries evaluated: {len(queries_to_eval)}")
    print(f"{'═' * 70}")
    print(
        "\n  Note: IR metrics (MRR, NDCG, Recall, Precision, F1) are not\n"
        "  computed because there is no ranked document list to compare\n"
        "  against the relevance judgements.\n"
        "\n"
        "  Faithfulness is not computed because no retrieved context\n"
        "  exists to verify the answer against.\n"
    )
    print(f"── LLM Generation Metric {'─' * 43}")

    if answer_relevance_score is not None:
        print(
            f"\n  Answer Relevance:  {answer_relevance_score:.4f}"
            f"  (evaluated on {answer_relevance_n} queries)"
        )
    else:
        print("\n  Answer Relevance:  skipped (--skip-answer-relevance was set).")

    print(f"\nTotal queries evaluated: {len(queries_to_eval)}")

    # ── Save answers ───────────────────────────────────────────────────────────
    output_path = dataset_path / "llm_only_answers.json"
    save_answers(answer_records, output_path)
    print(f"Generated answers saved to: {output_path}")


if __name__ == "__main__":
    main()
