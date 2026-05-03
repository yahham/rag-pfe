"""
BEIR data loaders.

Converts the BEIR wire format to the internal document format used by
InvertedIndex and SemanticSearch:
  BEIR:     {"_id": str, "title": str, "text": str}
  Internal: {"id":  str, "title": str, "description": str}
"""

import json
from pathlib import Path


def load_beir_corpus(path: Path) -> list[dict]:
    """Load corpus.jsonl and return documents in the internal format.

    Combines title + text into the 'description' field, which is what
    InvertedIndex and ChunkedSemanticSearch both index.
    """
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            title = raw.get("title", "") or ""
            text = raw.get("text", "") or ""
            docs.append(
                {
                    "id": raw["_id"],
                    "title": title,
                    "description": text,
                }
            )
    return docs


def load_beir_queries(path: Path) -> dict[str, str]:
    """Load queries.jsonl and return {query_id: query_text}."""
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries


def load_beir_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load qrels TSV and return {query_id: {doc_id: relevance_score}}.

    Only relevance scores > 0 are stored, consistent with the Rust evaluator.
    """
    qrels: dict[str, dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # skip header line: "query-id\tcorpus-id\tscore"
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qrels.setdefault(qid, {})[did] = score
    return qrels
