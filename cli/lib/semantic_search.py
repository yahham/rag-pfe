import os
import numpy as np
from pathlib import Path
from .search_utils import load_movies, CACHE_PATH
from sentence_transformers import SentenceTransformer


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Return the cosine similarity between two vectors, or 0.0 if either is a zero vector."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class SemanticSearch:
    """Semantic search over a document collection using sentence embeddings."""

    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings: np.ndarray | None = None
        self.documents: list[dict] | None = None
        self.document_map: dict[int, dict] = {}
        self.embeddings_path: Path = CACHE_PATH / "movie_embeddings.npy"

    def generate_embedding(self, text: str) -> np.ndarray:
        """Encode a single string into an embedding vector."""
        if not text or not text.strip():
            raise ValueError("Cannot create an embedding from empty text.")
        return self.model.encode(text)

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Encode all documents and save the embeddings to disk."""
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        movie_strings = [
            f"{doc['title']}: {doc['description']}" for doc in documents
        ]
        os.makedirs(CACHE_PATH, exist_ok=True)
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Load cached embeddings from disk, rebuilding them if missing or stale."""
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        if self.embeddings_path.exists():
            cached = np.load(self.embeddings_path)
            if len(cached) == len(documents):
                self.embeddings = cached
                return self.embeddings
            print(
                f"Cache contains {len(cached)} embeddings but dataset has "
                f"{len(documents)} documents. Rebuilding."
            )
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Return the top-scoring documents for query using cosine similarity."""
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings are loaded. Call 'load_or_create_embeddings' first."
            )
        query_embedding = self.generate_embedding(query)
        similarities = [
            (cosine_similarity(query_embedding, doc_emb), doc)
            for doc_emb, doc in zip(self.embeddings, self.documents)
        ]
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            }
            for score, doc in similarities[:limit]
        ]


# ---------------------------------------------------------------------------
# CLI command handlers
# ---------------------------------------------------------------------------

def verify_model() -> None:
    """Print basic information about the loaded sentence transformer model."""
    ss = SemanticSearch()
    print(f"Model: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text: str) -> None:
    """Print the embedding dimensions and first few values for the given text."""
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text:                 {text}")
    print(f"Dimensions:           {embedding.shape[0]}")
    print(f"First 5 dimensions:   {embedding[:5]}")


def verify_embeddings() -> None:
    """Load or build embeddings for the movie dataset and print summary information."""
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Documents: {len(documents)}")
    print(f"Embedding matrix shape: {embeddings.shape}  "
          f"({embeddings.shape[0]} vectors × {embeddings.shape[1]} dimensions)")


def search_command(query: str, limit: int = 5) -> None:
    """Run a semantic search and print ranked results."""
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    results = ss.search(query, limit)
    if not results:
        print(f"No results found for '{query}'.")
        return
    print(f"Results for '{query}':")
    for i, result in enumerate(results, start=1):
        print(f"  {i}. {result['title']} (score: {result['score']:.4f})")
        print(f"     {result['description'][:100]}")
