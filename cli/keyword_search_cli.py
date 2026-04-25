#!/usr/bin/env python3

import argparse
from lib.keyword_search import (
    search_command,
    build_command,
    tf_command,
    idf_command,
    tfidf_command,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword search over a movie dataset.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Search movies using indexed keywords"
    )
    search_parser.add_argument("query", type=str, help="One or more search terms")

    # Build command
    subparsers.add_parser(
        "build", help="Index the movies.json dataset and write cache files"
    )

    # TF command
    tf_parser = subparsers.add_parser("tf", help="Calculate Term Frequency")
    tf_parser.add_argument("doc_id", type=int, help="The ID of the movie to inspect")
    tf_parser.add_argument("term", type=str, help="The term to count")

    # IDF command
    idf_parser = subparsers.add_parser(
        "idf", help="Calculate Inverse Document Frequency"
    )
    idf_parser.add_argument("term", type=str, help="The term to analyze")

    # TF-IDF command
    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF score")
    tfidf_parser.add_argument("doc_id", type=int, help="The ID of the movie to inspect")
    tfidf_parser.add_argument("term", type=str, help="The term to score")

    args = parser.parse_args()

    match args.command:
        case "search":
            results = search_command(args.query, n_results=5)
            if not results:
                print(f"No results found for '{args.query}'.")
            else:
                print(f"Results for '{args.query}':")
                for i, result in enumerate(results, start=1):
                    print(f"  {i}. {result['title']}")
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
