#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over a movie dataset.")
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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
