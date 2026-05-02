#!/usr/bin/env python3

import argparse
from lib.hybrid_search import normalize_scores, weighted_search, rrf_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid search over a movie dataset.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Normalize scores command
    normalize_parser = subparsers.add_parser(
        "normalize", help="Min-max normalize a list of scores and print the results"
    )
    normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="+",
        help="One or more numeric scores to normalize",
    )

    # Weighted search command
    weighted_search_parser = subparsers.add_parser(
        "weighted_search",
        help="Search the movie dataset by blending BM25 and semantic scores",
    )
    weighted_search_parser.add_argument(
        "query", type=str, help="One or more search terms"
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Blend factor: 1.0 is pure BM25, 0.0 is pure semantic (default: 0.5)",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to return (default: 5)",
    )

    # RRF search command
    rrf_search_parser = subparsers.add_parser(
        "rrf_search",
        help="Search the movie dataset using Reciprocal Rank Fusion over BM25 and semantic rankings",
    )
    rrf_search_parser.add_argument("query", type=str, help="One or more search terms")
    rrf_search_parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="RRF smoothing constant: higher values reduce the influence of top ranks (default: 60)",
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to return (default: 5)",
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual"],
        help="Rerank method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            for norm_score in normalize_scores(args.scores):
                print(f"  {norm_score:.4f}")
        case "weighted_search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf_search":
            rrf_search(args.query, args.k, args.limit, args.enhance, args.rerank_method)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
