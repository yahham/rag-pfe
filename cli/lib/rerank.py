import json
import logging
from .search_utils import PROMPT_PATH
from .llm import call_llm
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_MOVIE_TEMPLATE = "<movie id={idx}>\nTitle: {title}\n{description}\n</movie>"

# Loaded once at import time; reused across all cross-encoder calls.
_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def individual_rerank(
    query: str, documents: list[dict], limit: int | None = None
) -> list[dict]:
    """Re-rank documents by asking the LLM to score each one against the query.

    Each document receives a relevance score from 0–10. Results are returned
    in descending score order, optionally trimmed to limit.
    """
    with open(PROMPT_PATH / "individual_rerank.md", "r") as f:
        prompt_template = f.read()

    results = []
    for doc in documents:
        formatted = prompt_template.format(
            query=query,
            title=doc["title"],
            description=doc["description"],
        )
        raw = call_llm(formatted).strip()
        try:
            score = int("".join(filter(str.isdigit, raw.split()[0])))
            score = max(0, min(10, score))
        except (ValueError, IndexError):
            score = 0
        results.append({**doc, "rerank_score": score})

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:limit] if limit is not None else results


def batch_rerank(
    query: str, documents: list[dict], limit: int | None = None
) -> list[dict]:
    """Re-rank documents in a single LLM call by presenting them all at once.

    The model returns a JSON list of 0-based indices ordered from most to least
    relevant. Documents are reordered accordingly. Results are optionally trimmed
    to limit.
    """
    with open(PROMPT_PATH / "batch_rerank.md", "r") as f:
        prompt_template = f.read()

    doc_list_str = "\n".join(
        _MOVIE_TEMPLATE.format(
            idx=i,
            title=doc["title"],
            description=doc["description"],
        )
        for i, doc in enumerate(documents)
    )

    formatted = prompt_template.format(query=query, doc_list_str=doc_list_str)
    raw = call_llm(formatted).strip()

    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines() if not line.startswith("```")
        ).strip()

    try:
        ranked_indices: list[int] = json.loads(raw)
        if not isinstance(ranked_indices, list):
            raise ValueError("Expected a JSON list.")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Batch rerank: could not parse LLM response (%s). Returning original order.",
            exc,
        )
        return documents[:limit] if limit is not None else documents

    seen: set[int] = set()
    reranked: list[dict] = []
    for idx in ranked_indices:
        if isinstance(idx, int) and 0 <= idx < len(documents) and idx not in seen:
            reranked.append(documents[idx])
            seen.add(idx)

    for i, doc in enumerate(documents):
        if i not in seen:
            reranked.append(doc)

    return reranked[:limit] if limit is not None else reranked


def cross_encoder_rerank(
    query: str, documents: list[dict], limit: int | None = None
) -> list[dict]:
    """Re-rank documents using a cross-encoder model for precise relevance scoring.

    Pairs the query with each document's title and description and scores them
    jointly. Results are returned in descending score order, optionally trimmed
    to limit.
    """
    pairs = [
        [query, f"{doc.get('title', '')} - {doc.get('description', '')}"]
        for doc in documents
    ]
    scores = _cross_encoder.predict(pairs)
    results = [
        {**doc, "cross_encoder_score": float(scores[i])}
        for i, doc in enumerate(documents)
    ]
    results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
    return results[:limit] if limit is not None else results
