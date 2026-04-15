from typing import Dict, Any


class RetrievalPipeline:
    def __init__(self, sparse_retriever=None, dense_retriever=None, reranker=None):
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.reranker = reranker

    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        raise NotImplementedError('Compose BM25 + dense + RRF + reranker here in later phases.')
