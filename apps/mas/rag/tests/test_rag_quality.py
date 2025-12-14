"""Tests for RAG quality detection and query expansion."""
from __future__ import annotations


def test_detect_rag_quality_good():
    from apps.mas.rag.evidence import detect_rag_quality
    from apps.mas.rag.retriever import RetrievedChunk

    chunks = [
        RetrievedChunk(
            id="c1", doc_id="d1", title="Relevant Page", text="content",
            url="http://wiki/page", chunk_idx=0, score=0.017, retrieval_method="fusion"
        ),
        RetrievedChunk(
            id="c2", doc_id="d2", title="Another Page", text="content",
            url="http://wiki/page2", chunk_idx=0, score=0.012, retrieval_method="fusion"
        ),
    ]
    
    quality, max_score = detect_rag_quality(chunks)
    assert quality == "good"
    assert max_score == 0.017


def test_detect_rag_quality_poor():
    from apps.mas.rag.evidence import detect_rag_quality
    from apps.mas.rag.retriever import RetrievedChunk

    chunks = [
        RetrievedChunk(
            id="c1", doc_id="d1", title="Irrelevant", text="content",
            url="http://wiki/page", chunk_idx=0, score=0.008, retrieval_method="fusion"
        ),
        RetrievedChunk(
            id="c2", doc_id="d2", title="Also Irrelevant", text="content",
            url="http://wiki/page2", chunk_idx=0, score=0.008, retrieval_method="fusion"
        ),
    ]
    
    quality, max_score = detect_rag_quality(chunks)
    assert quality == "poor"
    assert max_score == 0.008


def test_detect_rag_quality_empty():
    from apps.mas.rag.evidence import detect_rag_quality
    
    quality, max_score = detect_rag_quality([])
    assert quality == "poor"
    assert max_score == 0.0


def test_expand_person_query_height():
    from apps.mas.rag.evidence import expand_person_query
    
    expanded = expand_person_query("What is the height of Sonny Cabatu?")
    assert len(expanded) > 0
    assert any("basketball" in q.lower() for q in expanded)
    assert any("Sonny Cabatu" in q for q in expanded)


def test_expand_person_query_who():
    from apps.mas.rag.evidence import expand_person_query
    
    expanded = expand_person_query("Who is John Smith?")
    assert len(expanded) > 0
    assert any("John Smith" in q for q in expanded)


def test_expand_person_query_non_person():
    from apps.mas.rag.evidence import expand_person_query
    
    # Non-person query should return empty
    expanded = expand_person_query("What is the capital of France?")
    assert expanded == []


def test_should_suggest_web_fallback_poor_quality():
    from apps.mas.rag.evidence import should_suggest_web_fallback
    from apps.mas.rag.retriever import RetrievedChunk

    chunks = [
        RetrievedChunk(
            id="c1", doc_id="d1", title="Irrelevant", text="content",
            url="http://wiki/page", chunk_idx=0, score=0.008, retrieval_method="fusion"
        ),
    ]
    
    should_suggest, msg = should_suggest_web_fallback(chunks, web_enabled=False)
    assert should_suggest is True
    assert "web search" in msg.lower()


def test_should_suggest_web_fallback_good_quality():
    from apps.mas.rag.evidence import should_suggest_web_fallback
    from apps.mas.rag.retriever import RetrievedChunk

    chunks = [
        RetrievedChunk(
            id="c1", doc_id="d1", title="Relevant", text="content",
            url="http://wiki/page", chunk_idx=0, score=0.020, retrieval_method="fusion"
        ),
    ]
    
    should_suggest, msg = should_suggest_web_fallback(chunks, web_enabled=False)
    assert should_suggest is False
    assert msg == ""


def test_should_suggest_web_fallback_web_enabled():
    from apps.mas.rag.evidence import should_suggest_web_fallback
    from apps.mas.rag.retriever import RetrievedChunk

    # Even with poor quality, don't suggest if web is already enabled
    chunks = [
        RetrievedChunk(
            id="c1", doc_id="d1", title="Irrelevant", text="content",
            url="http://wiki/page", chunk_idx=0, score=0.008, retrieval_method="fusion"
        ),
    ]
    
    should_suggest, msg = should_suggest_web_fallback(chunks, web_enabled=True)
    assert should_suggest is False

