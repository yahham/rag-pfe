import os
import math
import string
import pickle
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from .search_utils import load_movies, load_stopwords, CACHE_PATH

stemmer = PorterStemmer()

class InvertedIndex:
    """In-memory inverted index with TF, IDF, and TF-IDF support."""
    def __init__(self):
        self.index                 = defaultdict(set)
        self.docmap                = {}
        self.term_frequencies      = defaultdict(Counter)
        self.index_path            = CACHE_PATH / "index.pkl"
        self.docmap_path           = CACHE_PATH / "docmap.pkl"
        self.term_frequencies_path = CACHE_PATH / "term_frequencies.pkl"

    def _add_document(self, doc_id: int, text: str) -> None:
        """Tokenize text and add the document to the index."""
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

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

    def build(self) -> None:
        """Load the movies dataset and build the index from scratch."""
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self._add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self) -> None:
        """Persist the index, docmap, and term frequencies to disk."""
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        """Load a previously saved index from disk."""
        missing = [
            p for p in (self.index_path, self.docmap_path, self.term_frequencies_path)
            if not p.exists()
        ]
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
    print(
        f"TF-IDF score of '{term}' in document {doc_id}: "
        f"{idx.get_tfidf(doc_id, term):.4f}"
    )

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
