from src.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank, ndcg_at_k


def test_recall_at_k():
    assert recall_at_k(['a', 'b', 'c'], {'b', 'd'}, 2) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank(['x', 'y', 'z'], {'y'}) == 0.5


def test_ndcg_at_k_positive():
    score = ndcg_at_k(['a', 'b'], {'a': 2, 'b': 1}, 2)
    assert 0.0 < score <= 1.0
