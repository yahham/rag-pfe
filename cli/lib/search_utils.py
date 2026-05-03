import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"
MOVIES_PATH = DATA_PATH / "movies.json"
STOPWORDS_PATH = DATA_PATH / "stopwords.txt"
CACHE_PATH = PROJECT_ROOT / "cache"
PROMPT_PATH = PROJECT_ROOT / "cli" / "lib" / "prompts"

BEIR_DATASET_PATH = Path(
    os.getenv("BEIR_DATASET_PATH", str(PROJECT_ROOT / "datasets" / "scifact"))
)
BEIR_CACHE_PATH = PROJECT_ROOT / "cache" / "beir"

BM25_K1 = 1.5  # Term frequency saturation parameter. Typical range: 1.2–2.0.
BM25_B = 0.75  # Document length normalization factor. Range: 0 (none) to 1 (full).


def load_movies() -> list[dict]:
    """Load and return the list of movies from the dataset."""
    with open(MOVIES_PATH, "r") as f:
        return json.load(f)["movies"]


def load_stopwords() -> list[str]:
    """Load and return the list of stopwords, one per line."""
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
