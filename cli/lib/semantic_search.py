import os
import re
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from .search_utils import load_movies, CACHE_PATH


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Return the cosine similarity between two vectors, or 0.0 if either is a zero vector."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class SemanticSearch:
    """Semantic search over a document collection using sentence embeddings."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings: np.ndarray | None = None
        self.documents: list[dict] | None = None
        self.document_map: dict[int, dict] = {}
        _cache = cache_dir if cache_dir is not None else CACHE_PATH
        self.embeddings_path: Path = _cache / "movie_embeddings.npy"

    def generate_embedding(self, text: str) -> np.ndarray:
        """Encode a single string into an embedding vector."""
        if not text or not text.strip():
            raise ValueError("Cannot create an embedding from empty text.")
        return self.model.encode(text)

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Encode all documents and save the embeddings to disk."""
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
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


class ChunkedSemanticSearch(SemanticSearch):
    """Semantic search that splits document descriptions into overlapping chunks
    before encoding, enabling finer-grained retrieval over long texts."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        super().__init__(cache_dir=cache_dir)
        _cache = cache_dir if cache_dir is not None else CACHE_PATH
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_embeddings_path: Path = _cache / "chunk_embeddings.npy"
        self.chunk_metadata: list[dict] | None = None
        self.chunk_metadata_path: Path = _cache / "chunk_metadata.json"

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Chunk all document descriptions, encode the chunks, and save to disk."""
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []
        for movie_idx, document in enumerate(documents):
            if not document["description"].strip():
                continue
            chunks = semantic_chunking(
                document["description"], max_chunk_size=4, overlap=1
            )
            for chunk_idx, _ in enumerate(chunks):
                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunks),
                    }
                )
            all_chunks.extend(chunks)
        os.makedirs(CACHE_PATH, exist_ok=True)
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )
        return self.chunk_embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Load cached chunk embeddings from disk, rebuilding them if missing."""
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        if self.chunk_embeddings_path.exists() and self.chunk_metadata_path.exists():
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 5) -> list[dict]:
        """Score each movie by the maximum cosine similarity across its chunks and
        return the top-scoring documents.
        """
        if (
            self.chunk_embeddings is None
            or self.chunk_metadata is None
            or self.documents is None
        ):
            raise ValueError(
                "No chunk embeddings are loaded. Call 'load_or_create_embeddings' first."
            )
        query_embedding = self.generate_embedding(query)
        movie_scores: dict[int, float] = defaultdict(float)
        for chunk_embedding, metadata in zip(
            self.chunk_embeddings, self.chunk_metadata
        ):
            movie_idx = metadata["movie_idx"]
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            movie_scores[movie_idx] = max(movie_scores[movie_idx], similarity)
        top = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [
            {
                "id": self.documents[movie_idx]["id"],
                "title": self.documents[movie_idx]["title"],
                "description": self.documents[movie_idx]["description"][:100],
                "score": round(score, 4),
            }
            for movie_idx, score in top
        ]


# ---------------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------------


def fixed_sized_chunking(
    text: str, overlap: int = 0, chunk_size: int = 200
) -> list[str]:
    """Split text into overlapping fixed-size word chunks."""
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})."
        )
    words = text.split()
    chunks = []
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
    return chunks


def semantic_chunking(
    text: str, max_chunk_size: int = 4, overlap: int = 0
) -> list[str]:
    """Split text into overlapping sentence-boundary chunks."""
    if overlap >= max_chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than max_chunk_size ({max_chunk_size})."
        )
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks = []
    step_size = max_chunk_size - overlap
    for i in range(0, len(sentences), step_size):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
    return chunks


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
    print(f"Text:               {text}")
    print(f"Dimensions:         {embedding.shape[0]}")
    print(f"First 5 dimensions: {embedding[:5]}")


def verify_embeddings() -> None:
    """Load or build embeddings for the movie dataset and print summary information."""
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Documents: {len(documents)}")
    print(
        f"Embedding matrix shape: {embeddings.shape}  "
        f"({embeddings.shape[0]} vectors × {embeddings.shape[1]} dimensions)"
    )


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


def chunk_text(text: str, overlap: int = 0, chunk_size: int = 200) -> None:
    """Split text into fixed-size word chunks and print each one."""
    chunks = fixed_sized_chunking(text, overlap, chunk_size)
    print(
        f"Chunking {len(text.split())} words into chunks of {chunk_size} (overlap: {overlap}):"
    )
    for i, chunk in enumerate(chunks, start=1):
        print(f"  {i}. {chunk}")


def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 0) -> None:
    """Split text into sentence-boundary chunks and print each one."""
    chunks = semantic_chunking(text, max_chunk_size, overlap)
    print(
        f"Semantically chunking text into groups of {max_chunk_size} sentences (overlap: {overlap}):"
    )
    for i, chunk in enumerate(chunks, start=1):
        print(f"  {i}. {chunk}")


def embed_chunks() -> None:
    """Build or load chunk embeddings for the movie dataset and print a summary."""
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_embeddings(movies)
    print(f"Chunk embeddings ready: {len(embeddings)} vectors.")


def search_chunked(query: str, limit: int = 5) -> None:
    """Run a chunked semantic search and print ranked results."""
    movies = load_movies()
    css = ChunkedSemanticSearch()
    css.load_or_create_embeddings(movies)
    results = css.search_chunks(query, limit)
    if not results:
        print(f"No results found for '{query}'.")
        return
    print(f"Chunked results for '{query}':")
    for i, result in enumerate(results, start=1):
        description = result["description"]
        suffix = "..." if len(result["description"]) >= 100 else ""
        print(f"  {i}. {result['title']} (score: {result['score']:.4f})")
        print(f"     {description}{suffix}")
