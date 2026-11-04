import pytest
from stratakb.rag.retriever import _build_hybrid_sql, _fuse_rrf


def test_rrf_fusion_combines_scores():
    """RRF should combine semantic and keyword ranks into a single score."""
    semantic = [("id1", 1), ("id2", 2), ("id3", 3)]
    keyword = [("id2", 1), ("id4", 2), ("id1", 3)]

    fused = _fuse_rrf(semantic, keyword, k=60, sem_weight=0.7, kw_weight=0.3)

    ids = [f[0] for f in fused]
    # id1: sem rank 1 (0.7/61) + kw rank 3 (0.3/63) = highest combined
    # id2: sem rank 2 (0.7/62) + kw rank 1 (0.3/61) = close second
    assert ids[0] == "id1", f"Expected id1 first, got {ids[0]}"
    assert ids[1] == "id2", f"Expected id2 second, got {ids[1]}"
    assert set(ids) == {"id1", "id2", "id3", "id4"}


def test_rrf_fusion_semantic_only():
    """When keyword returns nothing, semantic results should still work."""
    semantic = [("id1", 1), ("id2", 2)]
    keyword = []
    fused = _fuse_rrf(semantic, keyword, k=60, sem_weight=0.7, kw_weight=0.3)
    assert len(fused) == 2
    assert fused[0][0] == "id1"


def test_build_hybrid_sql_includes_fulltext():
    """Hybrid SQL should contain both tsv and embedding search CTEs."""
    sql, params = _build_hybrid_sql(
        query_vector="[1.0,2.0]",
        query_text="SEPA rejection AC01",
        layers=["sepa"],
        top_k=5,
    )
    assert "tsv" in sql
    assert "embedding" in sql
    assert "rrf" in sql.lower() or "UNION ALL" in sql
