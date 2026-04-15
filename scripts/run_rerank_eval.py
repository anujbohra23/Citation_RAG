import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from collections import defaultdict

from src.evaluation.retrieval_metrics import evaluate_run
from src.retrieval.dense_retriever import load_dense_retriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker


DENSE_PATH = "data/processed/dense_retriever.pkl"
QUERIES_PATH = "data/eval/queries_1000_seed42.jsonl"
QRELS_PATH = "data/eval/qrels_1000_seed42.jsonl"
RESULTS_PATH = "data/eval/results/dense_rerank_medquad_1000_seed42.json"

FIRST_STAGE_K = 20
FINAL_TOP_K = 10
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


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
    dense = load_dense_retriever(DENSE_PATH)
    reranker = CrossEncoderReranker(RERANKER_NAME)

    queries = load_queries(QUERIES_PATH)
    print(f"Running dense+rerank evaluation on {len(queries)} queries...")

    qrels_binary, qrels_graded = load_qrels(QRELS_PATH)

    run = {}
    for i, item in enumerate(queries, start=1):
        if i % 25 == 0:
            print(f"Processed {i}/{len(queries)} queries...")

        qid = item["qid"]
        query = item["query"]

        dense_results = dense.search(query, top_k=FIRST_STAGE_K)
        reranked = reranker.rerank(query, dense_results, top_k=FINAL_TOP_K)

        run[qid] = [r["chunk_id"] for r in reranked]

    metrics = evaluate_run(
        run=run,
        qrels_binary=qrels_binary,
        qrels_graded=qrels_graded,
        ks=[1, 3, 5, 10],
    )

    print("\nDense + Cross-Encoder Reranker metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    output = {
        "model": "dense_rerank",
        "first_stage": "dense",
        "reranker": RERANKER_NAME,
        "dataset": "MedQuAD",
        "eval_split": "queries_1000_seed42",
        "num_queries": len(queries),
        "first_stage_k": FIRST_STAGE_K,
        "final_top_k": FINAL_TOP_K,
        "metrics": metrics,
    }

    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()