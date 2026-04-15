import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.retrieval.dense_retriever import load_dense_retriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_formatter import format_sources


DENSE_PATH = "data/processed/dense_retriever.pkl"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FIRST_STAGE_K = 20
FINAL_EVIDENCE_K = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--first_stage_k", type=int, default=FIRST_STAGE_K)
    parser.add_argument("--final_evidence_k", type=int, default=FINAL_EVIDENCE_K)
    parser.add_argument("--show_evidence", action="store_true")
    args = parser.parse_args()

    dense = load_dense_retriever(DENSE_PATH)
    reranker = CrossEncoderReranker(RERANKER_NAME)
    generator = AnswerGenerator()

    dense_results = dense.search(args.query, top_k=args.first_stage_k)
    reranked = reranker.rerank(args.query, dense_results, top_k=args.final_evidence_k)

    result = generator.generate(args.query, reranked)

    print("\n=== QUERY ===")
    print(args.query)

    print("\n=== GROUNDED ANSWER ===")
    print(result["answer"])

    print("\n=== SOURCES ===")
    print(format_sources(reranked))

    if args.show_evidence:
        print("\n=== EVIDENCE TEXT ===")
        for i, chunk in enumerate(reranked, start=1):
            print(f"\n[{i}] {chunk['chunk_id']}")
            print(chunk["text"])
            print("-" * 80)


if __name__ == "__main__":
    main()