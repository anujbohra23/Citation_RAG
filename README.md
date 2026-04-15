````markdown
# CS535 Medical QA with Hybrid RAG

An end-to-end medical question answering system built on **MedQuAD** with:

- **BM25** sparse retrieval
- **Dense retrieval** using `sentence-transformers/all-MiniLM-L6-v2`
- **Hybrid retrieval** with Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Grounded answer generation** with citations

The current best pipeline is:

**Dense Retriever → Cross-Encoder Reranker → Top-3 Evidence Chunks → Grounded Answer**

---

## Features

- MedQuAD ingestion and preprocessing
- Paragraph-aware chunking
- Reproducible 1,000-query evaluation split
- BM25, dense, hybrid, and reranked retrieval benchmarks
- Grounded QA demo with citations
- Streamlit demo interface

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers faiss-cpu transformers torch sentencepiece streamlit
````

---

## Dataset

Clone MedQuAD into the project:

```bash
git clone https://github.com/abachaa/MedQuAD.git data/raw/medquad_repo
```

Do not commit the raw dataset repo into this repository.

---

## Run the pipeline

### Build data and indexes

```bash
python -m scripts.build_chunks
python -m scripts.build_medquad_eval
python -m scripts.freeze_eval_subset
python -m scripts.build_bm25
python -m scripts.build_dense_index
```

### Run retrieval evaluations

```bash
python -m scripts.run_bm25_eval
python -m scripts.run_dense_eval
python -m scripts.run_hybrid_eval
python -m scripts.run_rerank_eval
```

### Run search examples

```bash
python -m scripts.search_bm25 --query "What causes asthma?" --top_k 5
python -m scripts.search_dense --query "What causes asthma?" --top_k 5
python -m scripts.search_hybrid --query "What causes asthma?" --top_k 5
python -m scripts.search_rerank --query "What causes asthma?" --top_k 5
```

### Run grounded QA demo

```bash
python -m scripts.run_answer_demo --query "What causes asthma?" --show_evidence
```

### Run Streamlit app

```bash
python -m streamlit run src/demo/streamlit_app.py
```

---

## Data

```text
data/raw/medquad_repo/
data/interim/chunks.jsonl
data/eval/queries_1000_seed42.jsonl
data/eval/qrels_1000_seed42.jsonl
data/eval/results/
```

---

## Evaluation setup

* **Dataset:** MedQuAD
* **Documents loaded:** 16,423
* **Chunks created:** 21,646
* **Evaluation split:** 1,000 queries
* **Sampling:** fixed subset with `seed = 42`

### Chunking parameters

* `target_words = 220`
* `overlap_words = 40`
* `min_chunk_words = 40`
* `max_chunk_words = 300`

---

## Retrieval results

| Metric    |       BM25 |  Dense | Hybrid RRF | Dense + Rerank |
| --------- | ---------: | -----: | ---------: | -------------: |
| Recall@1  |     0.2000 | 0.3725 |     0.2997 |     **0.4469** |
| Recall@3  |     0.5272 | 0.5388 |     0.5086 |     **0.5997** |
| Recall@5  | **0.6528** | 0.6006 |     0.5866 |         0.6378 |
| Recall@10 | **0.7711** | 0.6695 |     0.6730 |         0.6834 |
| MRR       |     0.4022 | 0.6322 |     0.5786 |     **0.7133** |
| NDCG@1    |     0.2123 | 0.5176 |     0.4284 |     **0.6141** |
| NDCG@3    |     0.3942 | 0.5397 |     0.4908 |     **0.6207** |
| NDCG@5    |     0.4480 | 0.5584 |     0.5167 |     **0.6257** |
| NDCG@10   |     0.4890 | 0.5760 |     0.5388 |     **0.6317** |

### Summary

* **BM25** gives the best broad recall at larger cutoffs
* **Dense retrieval** improves semantic ranking quality
* **Hybrid RRF** did not outperform dense on this benchmark
* **Dense + reranker** is the strongest overall retrieval setup

---

## Grounded answer generation

The current generation module uses:

1. Dense retrieval
2. Cross-encoder reranking
3. Top-3 evidence chunk selection
4. Deterministic grounded answer composition with citations

### Example

**Query:**
`What causes asthma?`

**Answer:**
`The exact cause is not known, but the retrieved evidence indicates that it likely results from a combination of genetic and environmental factors, often early in life [3]. The evidence mentions examples such as inherited tendency toward allergies, family history, certain childhood respiratory infections, and exposure to allergens or irritants like tobacco smoke [2][3].`

This design is intentionally deterministic to improve grounding, citation consistency, and local reproducibility.

---

## Demo

The Streamlit app provides:

* question input
* grounded answer
* cited sources
* evidence inspection

Run it with:

```bash
python -m streamlit run src/demo/streamlit_app.py
```

---

## Project structure

```text
src/
├── chunking/
├── core/
├── demo/
├── evaluation/
├── generation/
├── indexing/
├── ingestion/
├── reranking/
└── retrieval/

scripts/
data/
```

---

## Notes

* The benchmark setup is fixed for fair comparison
* Dense retrieval currently uses a NumPy similarity fallback for stability on macOS
* FAISS can be reintroduced later if the environment supports it cleanly
* The primary retrieval stack for generation is **Dense + Rerank**

```
```
