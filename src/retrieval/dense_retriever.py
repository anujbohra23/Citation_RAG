from typing import List, Dict, Any


class DenseRetriever:
    """Placeholder for Weeks 7-8.

    Expected responsibilities:
    - load sentence-transformers bi-encoder
    - encode chunks
    - query FAISS index
    - return top-k scored chunk results
    """

    def __init__(self, *args, **kwargs):
        self.initialized = False

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError('DenseRetriever will be implemented in Weeks 7-8.')
