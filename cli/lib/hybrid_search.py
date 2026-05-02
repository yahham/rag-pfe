from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies
from .llm import augment_query
from .rerank import individual_rerank, batch_rerank, cross_encoder_rerank


class HybridSearch:
    """Combines BM25 keyword search and chunked semantic search via score fusion."""

    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_embeddings(documents)

        self.idx = InvertedIndex()
        try:
            self.idx.load()
        except FileNotFoundError:
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        """Return BM25-ranked results for query, drawing from a pool of up to limit documents."""
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self, query: str, alpha: float = 0.5, limit: int = 5
    ) -> list[dict]:
        """Return the top results by combining normalized BM25 and semantic scores.

        alpha controls the blend: 1.0 is pure BM25, 0.0 is pure semantic.
        """
        pool = max(limit * 10, 50)
        bm25_results = self._bm25_search(query, pool)
        semantic_results = self.semantic_search.search_chunks(query, pool)
        return combine_search_results(bm25_results, semantic_results, alpha, limit)

    def rrf_search(self, query: str, k: int = 60, limit: int = 5) -> list[dict]:
        """Return the top results using Reciprocal Rank Fusion over BM25 and semantic rankings."""
        pool = max(limit * 10, 50)
        bm25_results = self._bm25_search(query, pool)
        semantic_results = self.semantic_search.search_chunks(query, pool)
        return rrf_combine_search_results(bm25_results, semantic_results, k, limit)


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    """Return the weighted combination of a BM25 and a semantic score."""
    return (alpha * bm25_score) + ((1 - alpha) * semantic_score)


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to the [0, 1] range.

    Returns a list of zeros when all scores are equal (including all-zero inputs),
    rather than inflating them to 1.0.
    """
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    if score_range == 0:
        return [0.0] * len(scores)
    return [(score - min_score) / score_range for score in scores]


def normalize_search_results(results: list[dict]) -> list[dict]:
    """Add a 'normalized_score' field to each result dict, in place."""
    scores = [r["score"] for r in results]
    for result, norm in zip(results, normalize_scores(scores)):
        result["normalized_score"] = norm
    return results


def combine_search_results(
    bm25_results: list[dict],
    semantic_results: list[dict],
    alpha: float = 0.5,
    limit: int = 5,
) -> list[dict]:
    """Merge and re-rank BM25 and semantic result lists using weighted score fusion."""
    bm25_norm = normalize_search_results(bm25_results)
    semantic_norm = normalize_search_results(semantic_results)

    combined: dict[int, dict] = {}

    for result in bm25_norm:
        doc_id = result["doc_id"]
        combined[doc_id] = {
            "doc_id": doc_id,
            "title": result["title"],
            "description": result["description"],
            "bm25_score": result["normalized_score"],
            "semantic_score": 0.0,
        }

    for result in semantic_norm:
        doc_id = result["id"]
        if doc_id not in combined:
            combined[doc_id] = {
                "doc_id": doc_id,
                "title": result["title"],
                "description": result["description"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        combined[doc_id]["semantic_score"] = result["normalized_score"]

    for entry in combined.values():
        entry["hybrid_score"] = hybrid_score(
            entry["bm25_score"], entry["semantic_score"], alpha
        )

    return sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)[
        :limit
    ]


def rrf_score(rank: int, k: int = 60) -> float:
    """Return the RRF contribution for a single ranked result."""
    return 1.0 / (k + rank)


def rrf_final_score(
    bm25_rank: int | None, semantic_rank: int | None, k: int = 60
) -> float:
    """Return the combined RRF score from both ranking lists.

    Each list contributes independently: a document appearing in only one list
    still receives that list's RRF contribution rather than being zeroed out.
    """
    score = 0.0
    if bm25_rank is not None:
        score += rrf_score(bm25_rank, k)
    if semantic_rank is not None:
        score += rrf_score(semantic_rank, k)
    return score


def rrf_combine_search_results(
    bm25_results: list[dict],
    semantic_results: list[dict],
    k: int = 60,
    limit: int = 5,
) -> list[dict]:
    """Merge and re-rank BM25 and semantic result lists using Reciprocal Rank Fusion."""
    scores: dict[int, dict] = {}

    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["doc_id"]
        scores[doc_id] = {
            "doc_id": doc_id,
            "title": result["title"],
            "description": result["description"],
            "bm25_rank": rank,
            "bm25_rrf_score": rrf_score(rank, k),
            "semantic_rank": None,
            "semantic_rrf_score": 0.0,
        }

    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        if doc_id not in scores:
            scores[doc_id] = {
                "doc_id": doc_id,
                "title": result["title"],
                "description": result["description"],
                "bm25_rank": None,
                "bm25_rrf_score": 0.0,
                "semantic_rank": None,
                "semantic_rrf_score": 0.0,
            }
        scores[doc_id]["semantic_rank"] = rank
        scores[doc_id]["semantic_rrf_score"] = rrf_score(rank, k)

    for entry in scores.values():
        entry["rrf_score"] = rrf_final_score(
            entry["bm25_rank"], entry["semantic_rank"], k
        )

    return sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)[:limit]


# ---------------------------------------------------------------------------
# CLI command handlers
# ---------------------------------------------------------------------------


def weighted_search(query: str, alpha: float = 0.5, limit: int = 5) -> None:
    """Run a hybrid weighted search and print ranked results."""
    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.weighted_search(query, alpha, limit)
    if not results:
        print(f"No results found for '{query}'.")
        return
    print(f"Hybrid results for '{query}' (alpha={alpha}):")
    for i, result in enumerate(results, start=1):
        print(f"  {i}. {result['title']}")
        print(
            f"     Hybrid: {result['hybrid_score']:.4f}  "
            f"BM25: {result['bm25_score']:.4f}  "
            f"Semantic: {result['semantic_score']:.4f}"
        )
        print(f"     {result['description'][:100]}")


def _fmt_rank(rank: int | None) -> str:
    """Format a rank for display, substituting a dash when the rank is absent."""
    return str(rank) if rank is not None else "-"


def rrf_search(
    query: str,
    k: int = 60,
    limit: int = 5,
    enhance: str | None = None,
    rerank_method: str | None = None,
) -> None:
    """Run a Reciprocal Rank Fusion hybrid search and print ranked results."""
    movies = load_movies()
    hs = HybridSearch(movies)

    if enhance:
        enhanced_query = augment_query(query, enhance)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'")
        query = enhanced_query

    pool = limit * 5 if rerank_method else limit
    results = hs.rrf_search(query, k, pool)

    if rerank_method == "individual":
        print(f"Reranking {len(results)} candidates with the individual method...")
        results = individual_rerank(query, results, limit=limit)
    elif rerank_method == "batch":
        print(f"Reranking {len(results)} candidates with the batch method...")
        results = batch_rerank(query, results, limit=limit)
    elif rerank_method == "cross_encoder":
        print(f"Reranking {len(results)} candidates with the cross_encoder method...")
        results = cross_encoder_rerank(query, results, limit=limit)
    if not results:
        print(f"No results found for '{query}'.")
        return

    print(f"RRF results for '{query}' (k={k}):")
    for i, result in enumerate(results[:limit], start=1):
        print(f"  {i}. {result['title']}")
        score_parts = []
        if "rerank_score" in result:
            score_parts.append(f"Rerank: {result['rerank_score']}/10")
        if "cross_encoder_score" in result:
            score_parts.append(f"Cross Encoder: {result['cross_encoder_score']:.4f}")
        score_parts.append(f"RRF Score: {result['rrf_score']:.4f}")
        score_parts.append(f"BM25 Rank: {_fmt_rank(result['bm25_rank'])}")
        score_parts.append(f"Semantic Rank: {_fmt_rank(result['semantic_rank'])}")
        print(f"     {'  '.join(score_parts)}")
        print(f"     {result['description'][:100]}")
