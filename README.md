# CS535 Hybrid RAG Starter Project

A runnable starter implementation for **Hybrid Retrieval-Augmented Generation with Cross-Encoder Reranking for Technical Question Answering**.

This repo is structured to match a staged build:

1. **Weeks 5-6:** corpus onboarding, chunking, BM25 baseline, retrieval evaluation
2. **Weeks 7-8:** dense retrieval with sentence-transformers + FAISS
3. **Weeks 9-10:** hybrid fusion (RRF) + cross-encoder reranking
4. **Weeks 11-12:** grounded answer generation with citations
5. **Weeks 13-14:** faithfulness evaluation and demo polish

## What's implemented now

- Conservative text cleaning for technical corpora
- Paragraph-aware chunking with overlap fallback for oversized paragraphs
- BM25 baseline retrieval using `rank_bm25`
- Retrieval metrics: Recall@K, MRR, NDCG@K
- Scripts to build chunks, index BM25, run evaluation, and inspect search results
- Sample corpus, sample queries, and sample qrels
- Stubs for dense retrieval, hybrid fusion, reranking, generation, and demo

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Build chunks

```bash
python scripts/build_chunks.py
```

### Build BM25 index

```bash
python scripts/build_bm25.py
```

### Evaluate BM25

```bash
python scripts/run_bm25_eval.py
```

### Run a sample search

```bash
python scripts/search_bm25.py --query "How does BM25 handle document length normalization?" --top_k 5
```

## Data format

### Raw docs
Place `.txt` or `.md` files under:

```text
data/raw/docs/
```

### Queries
`data/interim/queries.jsonl`

```json
{"qid": "q1", "query": "How does BM25 handle document length normalization?"}
```

### Qrels
`data/eval/qrels.jsonl`

```json
{"qid": "q1", "chunk_id": "bm25_basics_chunk_0", "relevance": 2}
```

## Suggested next milestone

- Freeze the chunk schema
- Expand the evaluation set to 25-50 labeled queries
- Keep the same chunks/qrels when adding dense retrieval, hybrid fusion, and reranking

## Notes

- This is intentionally **framework-light** for the baseline.
- `faiss-cpu`, `sentence-transformers`, and `streamlit` are included for later phases.






# Locked BM25 Baseline

Dataset: MedQuAD  
Corpus size: 21,646 chunks  
Original documents: 16,423 QA documents  
Evaluation subset: 1,000 queries  
Sampling: random.sample with seed=42  
Retriever: BM25  
Chunking:
- target_words = 220
- overlap_words = 40
- min_chunk_words = 40
- max_chunk_words = 300

Metrics:
- Recall@1 = 0.2000
- Recall@3 = 0.5272
- Recall@5 = 0.6528
- Recall@10 = 0.7711
- MRR = 0.4022
- NDCG@1 = 0.2123
- NDCG@3 = 0.3942
- NDCG@5 = 0.4480
- NDCG@10 = 0.4890