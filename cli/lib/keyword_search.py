import os
import math
import string
import pickle
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from .search_utils import load_movies, load_stopwords, CACHE_PATH, BM25_K1, BM25_B

stemmer = PorterStemmer()

class InvertedIndex:
    """In-memory inverted index with TF, IDF, TF-IDF, and BM25 support."""

    def __init__(self):
        self.index                 = defaultdict(set)
        self.docmap                = {}
        self.term_frequencies      = defaultdict(Counter)
        self.doc_lengths           = {}
        self._avg_doc_length: float | None = None  # Cached on build(); invalidated on load()
        self.index_path            = CACHE_PATH / "index.pkl"
        self.docmap_path           = CACHE_PATH / "docmap.pkl"
        self.term_frequencies_path = CACHE_PATH / "term_frequencies.pkl"
        self.doc_lengths_path      = CACHE_PATH / "doc_lengths.pkl"

    def _add_document(self, doc_id: int, text: str) -> None:
        """Tokenize text and add the document to the index."""
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def _get_avg_doc_length(self) -> float:
        """Return the average document length, computing it once and caching the result."""
        if self._avg_doc_length is None:
            lengths = self.doc_lengths.values()
            self._avg_doc_length = sum(lengths) / len(lengths) if lengths else 0.0
        return self._avg_doc_length

    def get_documents(self, term: str) -> list[int]:
        """Return a sorted list of document IDs that contain the given term."""
        return sorted(self.index[term])

    def get_tf(self, doc_id: int, term: str) -> int:
        """Return the raw term frequency of a single term in the given document."""
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError(f"Expected exactly 1 token, got {len(tokens)}.")
        if doc_id not in self.docmap:
            raise KeyError(f"Document ID {doc_id} not found in the index.")
        return self.term_frequencies[doc_id][tokens[0]]

    def get_idf(self, term: str) -> float:
        """Return the smoothed IDF score for a single term."""
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError(f"Expected exactly 1 token, got {len(tokens)}.")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        """Return the TF-IDF score of a term in the given document."""
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        """Return the BM25-weighted term frequency for a term in the given document.

        k1 controls term frequency saturation: higher values reduce the ceiling
        effect, giving more weight to repeated terms. Typical range: 1.2–2.0.
        b controls document length normalization: 0 disables it, 1 applies it fully.
        """
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self._get_avg_doc_length()
        length_norm = (1 - b + b * (doc_length / avg_doc_length)) if avg_doc_length > 0 else 1.0
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_bm25_idf(self, term: str) -> float:
        """Return the BM25 IDF score for a single term.

        Uses the Robertson-Sparck Jones variant with a +1 floor to prevent
        negative scores for very common terms.
        """
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError(f"Expected exactly 1 token, got {len(tokens)}.")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25(self, doc_id: int, term: str) -> float:
        """Return the full BM25 score (TF component × IDF component) for a term in a document."""
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = 5) -> list[dict]:
        """Return the top-scoring documents for query using BM25 ranking.

        Only documents that appear in the inverted index for at least one query
        token are considered, keeping scoring efficient.
        """
        query_tokens = tokenize_text(query)
        scores: dict[int, float] = defaultdict(float)
        for token in query_tokens:
            for doc_id in self.get_documents(token):
                scores[doc_id] += self.get_bm25(doc_id, token)
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [
            {"doc_id": doc_id, "title": self.docmap[doc_id]["title"], "score": score}
            for doc_id, score in top
        ]

    def build(self) -> None:
        """Load the movies dataset and build the index from scratch."""
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self._add_document(doc_id, text)
            self.docmap[doc_id] = movie
        self._avg_doc_length = None  # Reset cache so it is recomputed from fresh data

    def save(self) -> None:
        """Persist the index, docmap, term frequencies, and document lengths to disk."""
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        """Load a previously saved index from disk."""
        paths = (
            self.index_path,
            self.docmap_path,
            self.term_frequencies_path,
            self.doc_lengths_path,
        )
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Cache files not found. Run 'build' first to index the dataset.\n"
                f"Missing: {', '.join(str(p) for p in missing)}"
            )
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
        self._avg_doc_length = None  # Invalidate cache after fresh load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase and strip punctuation from text."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    """Clean, filter stopwords, and stem the tokens in text."""
    stopwords = load_stopwords()
    tokens = []
    for token in clean_text(text).split():
        token = token.strip("\n")
        if token and token not in stopwords:
            tokens.append(stemmer.stem(token))
    return tokens


# ---------------------------------------------------------------------------
# CLI command handlers
# ---------------------------------------------------------------------------

def build_command() -> None:
    """Build and save the inverted index from the movies dataset."""
    idx = InvertedIndex()
    idx.build()
    idx.save()
    print(f"Index built and saved to '{CACHE_PATH}'.")

def tf_command(doc_id: int, term: str) -> None:
    idx = InvertedIndex()
    idx.load()
    print(f"Term frequency of '{term}' in document {doc_id}: {idx.get_tf(doc_id, term)}")

def idf_command(term: str) -> None:
    idx = InvertedIndex()
    idx.load()
    print(f"Inverse document frequency of '{term}': {idx.get_idf(term):.4f}")

def tfidf_command(doc_id: int, term: str) -> None:
    idx = InvertedIndex()
    idx.load()
    print(f"TF-IDF score of '{term}' in document {doc_id}: {idx.get_tfidf(doc_id, term):.4f}")

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    """Load the index and return the BM25 TF score for the given document and term."""
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25_idf_command(term: str) -> float:
    """Load the index and return the BM25 IDF score for the given term."""
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_search(query: str, n_results: int = 5) -> list[dict]:
    """Load the index and return BM25-ranked results for the given query."""
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit=n_results)

def search_command(query: str, n_results: int = 5) -> list[dict]:
    """Return up to n_results movies whose index entries match the query tokens."""
    idx = InvertedIndex()
    idx.load()
    seen, results = set(), []
    for token in tokenize_text(query):
        for doc_id in idx.get_documents(token):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(idx.docmap[doc_id])
            if len(results) >= n_results:
                return results
    return results
