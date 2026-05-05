from .llm import (
    answer_question,
    summarize_documents,
    citations_documents,
    answer_question_detailed as _llm_answer_detailed,
)
from .hybrid_search import HybridSearch
from .search_utils import load_movies


def _rrf_search(query: str, limit: int) -> list[dict]:
    """Run an RRF hybrid search and return results."""
    movies = load_movies()
    hs = HybridSearch(movies)
    return hs.rrf_search(query, k=60, limit=limit)


def query_answering(query: str, limit: int = 5) -> None:
    """Retrieve relevant movies and answer the query using their descriptions."""
    results = _rrf_search(query, limit)
    print("Search results:")
    for result in results:
        print(f"  - {result['title']}")
    print("\nRAG Response:")
    print(answer_question(query, results))


def document_summarization(query: str, limit: int = 5) -> None:
    """Retrieve relevant movies and produce a synthesized summary."""
    results = _rrf_search(query, limit)
    print("Search results:")
    for result in results:
        print(f"  - {result['title']}")
    print("\nLLM Summary:")
    print(summarize_documents(query, results))


def document_citations(query: str, limit: int = 5) -> None:
    """Retrieve relevant movies and answer the query with inline citations."""
    results = _rrf_search(query, limit)
    print("Search results:")
    for result in results:
        print(f"  - {result['title']}")
    print("\nLLM Answer:")
    print(citations_documents(query, results))


def question_answering_detailed(query: str, limit: int = 5) -> None:
    """Retrieve relevant movies and produce a detailed structured answer."""
    results = _rrf_search(query, limit)
    print("Search results:")
    for result in results:
        print(f"  - {result['title']}")
    print("\nLLM Answer:")
    print(_llm_answer_detailed(query, results))
