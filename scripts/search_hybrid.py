import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pickle

from src.retrieval.dense_retriever import load_dense_retriever
from src.retrieval.hybrid_fusion import reciprocal_rank_fusion


BM25_PATH = "data/processed/bm25_corpus.pkl"
DENSE_PATH = "data/processed/dense_retriever.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)

    dense = load_dense_retriever(DENSE_PATH)

    bm25_results = bm25.search(args.query, top_k=20)
    dense_results = dense.search(args.query, top_k=20)

    fused = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
        k=60,
        top_k=args.top_k,
    )

    print(f"Query: {args.query}\n")
    for item in fused:
        print(
            f"Rank {item['rank']} | Score {item['score']:.6f} | {item['chunk_id']} | Sources: {item.get('sources')}"
        )
        print(item["text"][:700])
        print("-" * 80)


if __name__ == "__main__":
    main()