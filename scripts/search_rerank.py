import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.retrieval.dense_retriever import load_dense_retriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker


DENSE_PATH = "data/processed/dense_retriever.pkl"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    dense = load_dense_retriever(DENSE_PATH)
    reranker = CrossEncoderReranker(RERANKER_NAME)

    dense_results = dense.search(args.query, top_k=20)
    reranked = reranker.rerank(args.query, dense_results, top_k=args.top_k)

    print(f"Query: {args.query}\n")
    for item in reranked:
        print(
            f"Rank {item['rank']} | Rerank Score {item['rerank_score']:.4f} | {item['chunk_id']}"
        )
        print(item["text"][:700])
        print("-" * 80)


if __name__ == "__main__":
    main()