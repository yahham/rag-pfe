from .search_utils import PROMPT_PATH
from .llm import call_llm


def individual_rerank(
    query: str, documents: list[dict], limit: int | None = None
) -> list[dict]:
    """Re-rank documents by asking the LLM to score each one against the query.

    Each document receives a relevance score from 0-10. Results are returned
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
