import os
import json
import time
import logging
from .search_utils import PROMPT_PATH
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

_api_key = os.environ.get("OPENROUTER_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "OPENROUTER_API_KEY is not set. Add it to your .env file or environment."
    )

MODEL = "openai/gpt-oss-120b:free"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=_api_key,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 2.0  # seconds; doubles on each attempt


def call_llm(content: str) -> str:
    """Send a pre-formatted message to the LLM and return the text response.

    Retries up to _MAX_RETRIES times with exponential backoff when the provider
    is rate-limited, either via a RateLimitError exception or a None choices
    payload (OpenRouter's behaviour for upstream 429s).
    """
    delay = _RETRY_BASE_DELAY
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": content}],
                extra_body={"reasoning": {"enabled": True}},
            )
        except RateLimitError:
            response = None

        if response is not None and response.choices:
            text = response.choices[0].message.content
            if text is not None:
                return text

        if attempt == _MAX_RETRIES:
            raise RuntimeError(
                f"LLM did not return a valid response after {_MAX_RETRIES} attempts. "
                "The upstream provider may be rate-limited — wait a moment and retry, "
                "or add your own API key at https://openrouter.ai/settings/integrations."
            )

        logger.warning(
            "No valid response from provider (attempt %d/%d). Retrying in %.0fs.",
            attempt,
            _MAX_RETRIES,
            delay,
        )
        time.sleep(delay)
        delay *= 2

    raise RuntimeError("Unreachable")


def generate_content(prompt: str, query: str, **kwargs) -> str:
    """Format prompt with query and any additional keyword arguments, then return
    the model's text response.
    """
    return call_llm(prompt.format(query=query, **kwargs))


def augment_query(query: str, prompt_type: str) -> str:
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
    return augment_query(query, "spelling")


def rewrite_query(query: str) -> str:
    """Return a rewritten version of query for improved retrieval."""
    return augment_query(query, "rewrite")


def expand_query(query: str) -> str:
    """Return an expanded version of query with additional relevant terms."""
    return augment_query(query, "expand")


def llm_judge(query: str, formatted_results: str) -> list[int] | None:
    """Score each search result for relevance to query on a 0–3 scale.

    Returns a list of integer scores in the same order as the results,
    or None if the LLM response cannot be parsed.
    """
    with open(PROMPT_PATH / "llm_judge.md", "r") as f:
        prompt = f.read()
    raw = generate_content(prompt, query, formatted_results=formatted_results).strip()

    # Strip markdown code fences if the model wraps its output.
    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines() if not line.startswith("```")
        ).strip()

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON list, got {type(parsed).__name__}.")
        return [max(0, min(3, int(s))) for s in parsed]
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("LLM judge: could not parse response (%s). Raw: %r", exc, raw)
        return None


def _format_documents(documents: list[dict]) -> str:
    """Serialize a list of search result dicts into a readable string for prompts."""
    parts = []
    for i, doc in enumerate(documents, start=1):
        title = doc.get("title", "Unknown")
        description = doc.get("description", "")
        parts.append(f"[{i}] {title}\n{description}")
    return "\n\n".join(parts)


def answer_question(query: str, documents: list[dict]) -> str:
    """Answer query using the provided documents as context."""
    with open(PROMPT_PATH / "answer_question.md", "r") as f:
        prompt = f.read()
    return generate_content(prompt, query=query, docs=_format_documents(documents))


def summarize_documents(query: str, documents: list[dict]) -> str:
    """Summarize the provided documents in the context of query."""
    with open(PROMPT_PATH / "summarization.md", "r") as f:
        prompt = f.read()
    return generate_content(prompt, query=query, docs=_format_documents(documents))


def citations_documents(query: str, documents: list[dict]) -> str:
    """Answer query with inline citations drawn from the provided documents."""
    with open(PROMPT_PATH / "answer_with_citations.md", "r") as f:
        prompt = f.read()
    return generate_content(prompt, query=query, docs=_format_documents(documents))


def answer_question_detailed(query: str, documents: list[dict]) -> str:
    """Answer query in detail, following structured guidance from the prompt."""
    with open(PROMPT_PATH / "answer_question_detailed.md", "r") as f:
        prompt = f.read()
    return generate_content(prompt, query=query, docs=_format_documents(documents))
