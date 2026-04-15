from typing import Dict, List, Set
import math


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    hits = len(retrieved_k.intersection(relevant_ids))
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for idx, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / idx
    return 0.0


def dcg_at_k(retrieved_ids: List[str], relevant_grades: Dict[str, float], k: int) -> float:
    score = 0.0
    for i, item_id in enumerate(retrieved_ids[:k], start=1):
        rel = relevant_grades.get(item_id, 0.0)
        if rel > 0:
            score += rel / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved_ids: List[str], relevant_grades: Dict[str, float], k: int) -> float:
    ideal = sorted(relevant_grades.values(), reverse=True)[:k]
    if not ideal:
        return 0.0

    idcg = 0.0
    for i, rel in enumerate(ideal, start=1):
        idcg += rel / math.log2(i + 1)

    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved_ids, relevant_grades, k) / idcg


def evaluate_run(
    run: Dict[str, List[str]],
    qrels_binary: Dict[str, Set[str]],
    qrels_graded: Dict[str, Dict[str, float]],
    ks: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    n = len(run)
    metrics = {f'Recall@{k}': 0.0 for k in ks}
    metrics['MRR'] = 0.0
    for k in ks:
        metrics[f'NDCG@{k}'] = 0.0

    if n == 0:
        return metrics

    for qid, retrieved in run.items():
        relevant_binary = qrels_binary.get(qid, set())
        relevant_graded = qrels_graded.get(qid, {})
        for k in ks:
            metrics[f'Recall@{k}'] += recall_at_k(retrieved, relevant_binary, k)
            metrics[f'NDCG@{k}'] += ndcg_at_k(retrieved, relevant_graded, k)
        metrics['MRR'] += reciprocal_rank(retrieved, relevant_binary)

    for key in metrics:
        metrics[key] /= n
    return metrics
