#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    search_command,
    chunk_text,
    semantic_chunk_text,
    embed_chunks,
    search_chunked,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic search over a movie dataset."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Verify model command
    subparsers.add_parser("verify", help="Print model information")

    # Verify embeddings command
    subparsers.add_parser(
        "verify_embeddings", help="Load or build embeddings and print a summary"
    )

    # Embed text command
    embed_parser = subparsers.add_parser(
        "embed_text", help="Encode text and print the resulting embedding dimensions"
    )
    embed_parser.add_argument("text", type=str, help="The text to encode")

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Search the movie dataset using semantic similarity"
    )
    search_parser.add_argument("query", type=str, help="One or more search terms")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to return (default: 5)",
    )

    # Fixed-size chunk command
    chunk_parser = subparsers.add_parser(
        "chunk", help="Split text into fixed-size word chunks and print them"
    )
    chunk_parser.add_argument("text", type=str, help="The text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk (default: 200)",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of words to overlap between consecutive chunks (default: 0)",
    )

    # Semantic chunk command
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text into sentence-boundary chunks and print them"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="The text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per chunk (default: 4)",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences to overlap between consecutive chunks (default: 0)",
    )

    # Embed chunks command
    subparsers.add_parser(
        "embed_chunks", help="Build or load chunk embeddings for the movie dataset"
    )

    # Search chunked command
    search_chunked_parser = subparsers.add_parser(
        "search_chunked",
        help="Search the movie dataset using chunk-level semantic similarity",
    )
    search_chunked_parser.add_argument(
        "query", type=str, help="One or more search terms"
    )
    search_chunked_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to return (default: 5)",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.overlap, args.chunk_size)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
