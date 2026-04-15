import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import random

INPUT_QUERIES = "data/interim/queries.jsonl"
INPUT_QRELS = "data/eval/qrels.jsonl"

OUTPUT_QUERIES = "data/eval/queries_1000_seed42.jsonl"
OUTPUT_QRELS = "data/eval/qrels_1000_seed42.jsonl"

SAMPLE_SIZE = 1000
SEED = 42


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    queries = load_jsonl(INPUT_QUERIES)
    qrels = load_jsonl(INPUT_QRELS)

    random.seed(SEED)
    sampled_queries = random.sample(queries, SAMPLE_SIZE)
    sampled_qids = {q["qid"] for q in sampled_queries}

    sampled_qrels = [row for row in qrels if row["qid"] in sampled_qids]

    write_jsonl(OUTPUT_QUERIES, sampled_queries)
    write_jsonl(OUTPUT_QRELS, sampled_qrels)

    print(f"Saved {len(sampled_queries)} frozen queries to {OUTPUT_QUERIES}")
    print(f"Saved {len(sampled_qrels)} frozen qrels to {OUTPUT_QRELS}")


if __name__ == "__main__":
    main()