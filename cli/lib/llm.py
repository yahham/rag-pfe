import os
from .search_utils import PROMPT_PATH
from dotenv import load_dotenv
from google import genai

load_dotenv()

_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. Add it to your .env file or environment."
    )

MODEL = "gemini-3-flash-preview"
client = genai.Client(api_key=_api_key)


def generate_content(prompt: str, query: str) -> str:
    """Format prompt with query and return the model's text response."""
    formatted = prompt.format(query=query)
    response = client.models.generate_content(model=MODEL, contents=formatted)
    return response.text


def _augment_query(query: str, prompt_type: str) -> str:
    """Load the prompt file for prompt_type and return the LLM-enhanced query."""
    prompt_file = PROMPT_PATH / f"{prompt_type}.md"
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file not found: '{prompt_file}'. "
            f"Create 'cli/lib/prompts/{prompt_type}.md' to enable this enhancement."
        )
    with open(prompt_file, "r") as f:
        prompt = f.read()
    result = generate_content(prompt, query).strip()
    if not result:
        raise ValueError(
            f"The LLM returned an empty response for prompt type '{prompt_type}'."
        )
    return result


def correct_spelling(query: str) -> str:
    """Return a spelling-corrected version of query."""
    return _augment_query(query, "spelling")


def rewrite_query(query: str) -> str:
    """Return a rewritten version of query for improved retrieval."""
    return _augment_query(query, "rewrite")


def expand_query(query: str) -> str:
    """Return an expanded version of query with additional relevant terms."""
    return _augment_query(query, "expand")
