from typing import Any, Dict, List, Tuple

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs: List[Tuple[str, str]] = []
        for item in candidates:
            pairs.append((query, item["text"]))

        scores = self.model.predict(pairs, show_progress_bar=False)

        rescored = []
        for item, score in zip(candidates, scores):
            new_item = dict(item)
            new_item["rerank_score"] = float(score)
            rescored.append(new_item)

        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)

        for rank, item in enumerate(rescored[:top_k], start=1):
            item["rank"] = rank

        return rescored[:top_k]