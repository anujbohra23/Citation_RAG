import numpy as np
import faiss


class FaissIndex:
    def __init__(self, dim: int, normalize: bool = True):
        self.dim = dim
        self.normalize = normalize
        self.index = faiss.IndexFlatIP(dim)

    def add(self, embeddings: np.ndarray) -> None:
        vectors = embeddings.astype("float32")
        if self.normalize:
            faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def search(self, query_embeddings: np.ndarray, top_k: int = 10):
        queries = query_embeddings.astype("float32")
        if self.normalize:
            faiss.normalize_L2(queries)
        scores, indices = self.index.search(queries, top_k)
        return scores, indices