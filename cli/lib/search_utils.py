import json
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
DATA_PATH      = PROJECT_ROOT / "data"
MOVIES_PATH    = DATA_PATH / "movies.json"
STOPWORDS_PATH = DATA_PATH / "stopwords.txt"
CACHE_PATH     = PROJECT_ROOT / "cache"

def load_movies() -> list[dict]:
    """Load and return the list of movies from the dataset."""
    with open(MOVIES_PATH, "r") as f:
        return json.load(f)["movies"]

def load_stopwords() -> list[str]:
    """Load and return the list of stopwords, one per line."""
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
