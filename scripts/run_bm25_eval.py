import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pickle
from collections import defaultdict

from src.evaluation.retrieval_metrics import evaluate_run


BM25_PATH = "data/processed/bm25_corpus.pkl"
QUERIES_PATH = "data/eval/queries_1000_seed42.jsonl"
QRELS_PATH = "data/eval/qrels_1000_seed42.jsonl"


def load_queries(path: str):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def load_qrels(path: str):
    qrels_binary = defaultdict(set)
    qrels_graded = defaultdict(dict)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row["qid"]
            chunk_id = row["chunk_id"]
            relevance = float(row["relevance"])

            if relevance > 0:
                qrels_binary[qid].add(chunk_id)
            qrels_graded[qid][chunk_id] = relevance

    return dict(qrels_binary), dict(qrels_graded)


def main():
    with open(BM25_PATH, "rb") as f:
        retriever = pickle.load(f)

    queries = load_queries(QUERIES_PATH)
    print(f"Running evaluation on {len(queries)} queries...")

    qrels_binary, qrels_graded = load_qrels(QRELS_PATH)

    run = {}
    for i, item in enumerate(queries, start=1):
        if i % 25 == 0:
            print(f"Processed {i}/{len(queries)} queries...")

        qid = item["qid"]
        query = item["query"]
        results = retriever.search(query, top_k=10)
        run[qid] = [r["chunk_id"] for r in results]

    metrics = evaluate_run(
        run=run,
        qrels_binary=qrels_binary,
        qrels_graded=qrels_graded,
        ks=[1, 3, 5, 10],
    )

    print("\nBM25 baseline metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()