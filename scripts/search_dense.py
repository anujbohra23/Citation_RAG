import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.retrieval.dense_retriever import load_dense_retriever


DENSE_PATH = "data/processed/dense_retriever.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    retriever = load_dense_retriever(DENSE_PATH)
    results = retriever.search(args.query, top_k=args.top_k)

    print(f"Query: {args.query}\n")
    for item in results:
        print(
            f"Rank {item['rank']} | Score {item['score']:.4f} | {item['chunk_id']}"
        )
        print(item["text"][:700])
        print("-" * 80)


if __name__ == "__main__":
    main()