````markdown
# CS535 Hybrid RAG Starter Project

A runnable implementation for **Hybrid Retrieval-Augmented Generation with Cross-Encoder Reranking** for **medical question answering** using **MedQuAD**.

This repo follows the staged build plan:

1. **Weeks 5-6:** corpus onboarding, chunking, BM25 baseline, retrieval evaluation
2. **Weeks 7-8:** dense retrieval with sentence-transformers
3. **Weeks 9-10:** hybrid fusion (RRF) + cross-encoder reranking
4. **Weeks 11-12:** grounded answer generation with citations
5. **Weeks 13-14:** faithfulness evaluation and demo polish

---

## What's implemented now

- MedQuAD ingestion pipeline
- Conservative text cleaning for medical QA data
- Paragraph-aware chunking with overlap fallback for oversized paragraphs
- BM25 sparse retrieval baseline using `rank_bm25`
- Dense retrieval baseline using `sentence-transformers/all-MiniLM-L6-v2`
- Retrieval metrics: Recall@K, MRR, NDCG@K
- Scripts to:
  - build chunks
  - build BM25 index
  - build dense retrieval artifact
  - run BM25 evaluation
  - run dense evaluation
  - inspect search results
- Frozen 1,000-query evaluation split for reproducible comparison

---

## Project status

### Completed
- MedQuAD corpus onboarding
- chunking pipeline
- BM25 baseline
- dense retrieval baseline
- reproducible evaluation split

### Next
- Hybrid fusion with Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Grounded answer generation with citations
- Faithfulness evaluation

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers faiss-cpu
````

---

## Dataset

This project currently uses **MedQuAD**.

Clone the dataset into the project like this:

```bash
git clone https://github.com/abachaa/MedQuAD.git data/raw/medquad_repo
```

The raw dataset repo is treated as local data and should not be committed into this repository.

---

## Build pipeline

### 1. Build chunks

```bash
python -m scripts.build_chunks
```

### 2. Build evaluation files

```bash
python -m scripts.build_medquad_eval
python -m scripts.freeze_eval_subset
```

### 3. Build BM25 index

```bash
python -m scripts.build_bm25
```

### 4. Evaluate BM25

```bash
python -m scripts.run_bm25_eval
```

### 5. Build dense retrieval artifact

```bash
python -m scripts.build_dense_index
```

### 6. Evaluate dense retrieval

```bash
python -m scripts.run_dense_eval
```

---

## Search examples

### BM25 search

```bash
python -m scripts.search_bm25 --query "What causes asthma?" --top_k 5
```

### Dense search

```bash
python -m scripts.search_dense --query "What causes asthma?" --top_k 5
```

---

## Data layout

### Raw data

```text
data/raw/medquad_repo/
```

### Chunked corpus

```text
data/interim/chunks.jsonl
```

### Queries

```text
data/interim/queries.jsonl
data/eval/queries_1000_seed42.jsonl
```

Example:

```json
{"qid": "3_GHR_QA_0", "query": "What is (are) keratoderma with woolly hair ?"}
```

### Qrels

```text
data/eval/qrels.jsonl
data/eval/qrels_1000_seed42.jsonl
```

Example:

```json
{"qid": "3_GHR_QA_0", "chunk_id": "3_GHR_QA_0_chunk_0", "relevance": 2}
```

---

## Locked evaluation setup

* **Dataset:** MedQuAD
* **Original QA documents loaded:** 16,423
* **Chunks created:** 21,646
* **Frozen evaluation split:** 1,000 queries
* **Sampling:** fixed subset with `seed = 42`
* **Chunking parameters:**

  * `target_words = 220`
  * `overlap_words = 40`
  * `min_chunk_words = 40`
  * `max_chunk_words = 300`

This benchmark setup is now fixed for fair comparison across:

* BM25
* Dense retrieval
* Hybrid retrieval
* Reranking

---

## Results

### BM25 baseline

| Metric    |   BM25 |
| --------- | -----: |
| Recall@1  | 0.2000 |
| Recall@3  | 0.5272 |
| Recall@5  | 0.6528 |
| Recall@10 | 0.7711 |
| MRR       | 0.4022 |
| NDCG@1    | 0.2123 |
| NDCG@3    | 0.3942 |
| NDCG@5    | 0.4480 |
| NDCG@10   | 0.4890 |

### Dense retrieval baseline

Encoder: `sentence-transformers/all-MiniLM-L6-v2`

| Metric    |  Dense |
| --------- | -----: |
| Recall@1  | 0.3725 |
| Recall@3  | 0.5388 |
| Recall@5  | 0.6006 |
| Recall@10 | 0.6695 |
| MRR       | 0.6322 |
| NDCG@1    | 0.5176 |
| NDCG@3    | 0.5397 |
| NDCG@5    | 0.5584 |
| NDCG@10   | 0.5760 |

### BM25 vs Dense comparison

| Metric    |   BM25 |  Dense | Better |
| --------- | -----: | -----: | ------ |
| Recall@1  | 0.2000 | 0.3725 | Dense  |
| Recall@3  | 0.5272 | 0.5388 | Dense  |
| Recall@5  | 0.6528 | 0.6006 | BM25   |
| Recall@10 | 0.7711 | 0.6695 | BM25   |
| MRR       | 0.4022 | 0.6322 | Dense  |
| NDCG@1    | 0.2123 | 0.5176 | Dense  |
| NDCG@3    | 0.3942 | 0.5397 | Dense  |
| NDCG@5    | 0.4480 | 0.5584 | Dense  |
| NDCG@10   | 0.4890 | 0.5760 | Dense  |

---

## Key takeaways

* **BM25** provides stronger broader recall at larger cutoffs, especially `Recall@5` and `Recall@10`
* **Dense retrieval** gives much stronger top-rank quality, with large gains in:

  * `Recall@1`
  * `MRR`
  * `NDCG`
* This complementary behavior motivates the next stage:

  * **Hybrid fusion with RRF**
  * **Cross-encoder reranking**

---

## Example observation

For the query **"What causes asthma?"**, the dense retriever returned a highly relevant top result directly explaining that asthma arises from interacting genetic and environmental factors, followed by multiple semantically related answers. This suggests that dense retrieval is especially strong for medical semantic matching.

---

## Suggested next milestone

* Implement hybrid retrieval with Reciprocal Rank Fusion (RRF)
* Compare hybrid vs BM25 vs dense on the same frozen split
* Add cross-encoder reranking on the hybrid top-k candidate set

---

## Notes

* The baseline is intentionally framework-light
* Dense retrieval currently uses a NumPy-based similarity search fallback for stability on macOS
* FAISS can be reintroduced later if the environment supports it cleanly
* Keep the corpus, chunks, eval split, and qrels fixed while comparing retrieval methods

```
```
