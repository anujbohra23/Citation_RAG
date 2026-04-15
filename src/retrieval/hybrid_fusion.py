from typing import Dict, List


def reciprocal_rank_fusion(runs: Dict[str, List[str]], k: int = 60) -> List[str]:
    scores: Dict[str, float] = {}
    for _, ranked_ids in runs.items():
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return [item_id for item_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
