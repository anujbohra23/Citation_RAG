import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from src.indexing.bm25_index import load_bm25_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--index_path', type=str, default='data/processed/bm25_corpus.pkl')
    args = parser.parse_args()

    retriever = load_bm25_index(args.index_path)
    results = retriever.search(args.query, top_k=args.top_k)

    print(f"Query: {args.query}\n")
    for r in results:
        print(f"Rank {r['rank']} | Score {r['score']:.4f} | {r['chunk_id']}")
        print(r['text'][:500])
        print('-' * 80)


if __name__ == '__main__':
    main()
