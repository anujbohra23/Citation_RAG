from typing import Dict, List, Any


def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    k: int = 60,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    scores = {}
    payload = {}

    for results, source in [(bm25_results, "bm25"), (dense_results, "dense")]:
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            rrf_score = 1.0 / (k + rank)

            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score

            if chunk_id not in payload:
                payload[chunk_id] = dict(item)
                payload[chunk_id]["sources"] = [source]
            else:
                if source not in payload[chunk_id]["sources"]:
                    payload[chunk_id]["sources"].append(source)

    fused = []
    for chunk_id, score in scores.items():
        item = dict(payload[chunk_id])
        item["score"] = float(score)
        fused.append(item)

    fused.sort(key=lambda x: x["score"], reverse=True)

    for rank, item in enumerate(fused[:top_k], start=1):
        item["rank"] = rank

    return fused[:top_k]