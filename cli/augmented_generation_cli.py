#!/usr/bin/env python3
import argparse
from lib.rag import (
    query_answering,
    document_summarization,
    document_citations,
    question_answering_detailed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    answer_parser = subparsers.add_parser(
        "answer", help="Search for relevant movies and answer the query"
    )
    answer_parser.add_argument("query", type=str, help="The question or search query")
    answer_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)",
    )

    summarize_parser = subparsers.add_parser(
        "summarize", help="Search for relevant movies and summarize the results"
    )
    summarize_parser.add_argument(
        "query", type=str, help="The question or search query"
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)",
    )

    cite_parser = subparsers.add_parser(
        "cite", help="Search for relevant movies and answer with inline citations"
    )
    cite_parser.add_argument("query", type=str, help="The question or search query")
    cite_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)",
    )

    detailed_parser = subparsers.add_parser(
        "detailed_answer",
        help="Search for relevant movies and provide a detailed answer",
    )
    detailed_parser.add_argument("query", type=str, help="The question or search query")
    detailed_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)",
    )

    args = parser.parse_args()

    match args.command:
        case "answer":
            query_answering(args.query, args.limit)
        case "summarize":
            document_summarization(args.query, args.limit)
        case "cite":
            document_citations(args.query, args.limit)
        case "detailed_answer":
            question_answering_detailed(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
