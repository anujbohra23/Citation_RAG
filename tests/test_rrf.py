from src.retrieval.hybrid_fusion import reciprocal_rank_fusion


def test_rrf_combines_runs():
    fused = reciprocal_rank_fusion({
        'bm25': ['a', 'b', 'c'],
        'dense': ['b', 'a', 'd'],
    })
    assert fused[0] in {'a', 'b'}
