#!/usr/bin/env python3

import argparse
from lib.keyword_search import (
    search_command,
    build_command,
    tf_command,
    idf_command,
    tfidf_command,
    bm25_idf_command,
    bm25_tf_command,
    bm25_search
)
from lib.search_utils import BM25_K1, BM25_B


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

    # BM25 TF command
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Calculate the BM25 term frequency score for a document and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="The ID of the movie to inspect")
    bm25_tf_parser.add_argument("term", type=str, help="The term to score")
    bm25_tf_parser.add_argument(
        "k1",
        type=float,
        nargs="?",
        default=BM25_K1,
        help=f"Saturation parameter controlling term frequency scaling (default: {BM25_K1})",
    )
    bm25_tf_parser.add_argument(
        "b",
        type=float,
        nargs="?",
        default=BM25_B,
        help=f"Document length normalization factor: 0 disables normalization, 1 applies it fully (default: {BM25_B})",
    )

    # BM25 IDF command
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Calculate the BM25 inverse document frequency score for a term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="The term to analyze")

    # BM25 Search command
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movie using BM25")
    bm25search_parser.add_argument("query", type=str, help="One or more search terms")

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
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF  score of '{args.term}' in document {args.doc_id}: {bm25tf:.4f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.4f}")
        case "bm25search":
            bm25_results = bm25_search(args.query)
            if not bm25_results:
                print(f"No results found for '{args.query}'.")
            else:
                print(f"BM25 results for '{args.query}':")
                for i, result in enumerate(bm25_results, start=1):
                    print(f"  {i}. {result['title']} (doc {result['doc_id']}) — score: {result['score']:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
    