import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.types import Chunk


class DenseRetriever:
    def __init__(
        self,
        model_name: str,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunks = chunks
        self.normalize = normalize

        self.embeddings = embeddings.astype("float32")
        if self.normalize:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12
            self.embeddings = self.embeddings / norms

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype("float32")

        if self.normalize:
            qnorm = np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-12
            query_embedding = query_embedding / qnorm

        scores = self.embeddings @ query_embedding[0]
        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[idx]),
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "source_path": chunk.source_path,
                }
            )

        return results


def save_dense_artifact(
    out_path: str,
    model_name: str,
    chunks: List[Chunk],
    embeddings: np.ndarray,
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": model_name,
        "chunks": chunks,
        "embeddings": embeddings.astype("float32"),
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)


def load_dense_retriever(path: str) -> DenseRetriever:
    with open(path, "rb") as f:
        payload = pickle.load(f)

    return DenseRetriever(
        model_name=payload["model_name"],
        chunks=payload["chunks"],
        embeddings=payload["embeddings"],
        normalize=True,
    )