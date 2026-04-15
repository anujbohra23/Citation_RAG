from src.core.types import Chunk
from src.retrieval.bm25_retriever import BM25Retriever


def test_bm25_returns_ranked_results():
    chunks = [
        Chunk(chunk_id='c1', doc_id='d1', title='bm25', text='BM25 uses lexical matching and document length normalization.'),
        Chunk(chunk_id='c2', doc_id='d2', title='rrf', text='Reciprocal rank fusion combines ranked lists.'),
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.search('document length normalization', top_k=1)
    assert results[0]['chunk_id'] == 'c1'
