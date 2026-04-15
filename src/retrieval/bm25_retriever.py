import re
from typing import List, Dict, Any
from src.retrieval.simple_bm25 import BM25Okapi
from src.core.types import Chunk


def tokenize_for_bm25(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9_+\-./#]+", text)


class BM25Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.tokenized_corpus = [tokenize_for_bm25(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        tokenized_query = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokenized_query)

        scored = []
        for idx, score in enumerate(scores):
            scored.append(
                {
                    "rank": None,
                    "score": float(score),
                    "chunk_id": self.chunks[idx].chunk_id,
                    "doc_id": self.chunks[idx].doc_id,
                    "title": self.chunks[idx].title,
                    "text": self.chunks[idx].text,
                    "source_path": self.chunks[idx].source_path,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_results = scored[:top_k]
        for rank, item in enumerate(top_results, start=1):
            item["rank"] = rank
        return top_results
