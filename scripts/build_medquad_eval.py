import json
from pathlib import Path


CHUNKS_PATH = "data/interim/chunks.jsonl"
QUERIES_PATH = "data/interim/queries.jsonl"
QRELS_PATH = "data/eval/qrels.jsonl"


def main() -> None:
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    Path("data/eval").mkdir(parents=True, exist_ok=True)

    queries = []
    qrels = []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            metadata = row.get("metadata") or {}
            question = metadata.get("question", "").strip()
            chunk_id = row["chunk_id"]
            doc_id = row["doc_id"]

            if not question:
                continue

            qid = doc_id

            queries.append({
                "qid": qid,
                "query": question
            })

            qrels.append({
                "qid": qid,
                "chunk_id": chunk_id,
                "relevance": 2
            })

    with open(QUERIES_PATH, "w", encoding="utf-8") as f:
        for item in queries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(QRELS_PATH, "w", encoding="utf-8") as f:
        for item in qrels:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(queries)} queries to {QUERIES_PATH}")
    print(f"Saved {len(qrels)} qrels to {QRELS_PATH}")


if __name__ == "__main__":
    main()