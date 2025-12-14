from __future__ import annotations

from typing import List, Tuple


class MockRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def fusion_search(self, query: str, k: int = 5):
        return list(self._chunks)[:k]


def test_retrieve_rag_evidence_for_subtask_emits_events():
    from apps.mas.agents.supervisor import SubTask
    from apps.mas.graph.plan_graph import _retrieve_rag_evidence_for_subtask
    from apps.mas.rag.retriever import RetrievedChunk

    chunks = [
        RetrievedChunk(
            id="d1_chunk_0",
            doc_id="d1",
            title="Doc One",
            text="Some relevant content about hobbits.",
            url="http://wiki/doc1",
            chunk_idx=0,
            score=0.9,
            retrieval_method="fusion",
        )
    ]

    retriever = MockRetriever(chunks)
    st = SubTask(id="step1", role="qa", instruction="Who wrote The Hobbit?", depends_on=[])
    events: List[Tuple[str, str]] = []

    def time_left() -> float:
        return 100.0

    def emit(stage: str, content: str) -> None:
        events.append((stage, content))

    pack = _retrieve_rag_evidence_for_subtask(
        retriever=retriever,
        subtask=st,
        original_problem="Who wrote The Hobbit?",
        rag_top_k=5,
        time_left=time_left,
        emit_thinking=emit,
    )

    assert pack is not None
    assert pack.chunks and pack.chunks[0].url == "http://wiki/doc1"

    stages = [s for s, _ in events]
    assert "rag_step1_rag_query" in stages
    assert "rag_step1_rag_chunks" in stages


def test_retrieve_rag_evidence_reuses_seed_chunks():
    """When seed_chunks is provided, reuse them instead of calling fusion_search."""
    from apps.mas.agents.supervisor import SubTask
    from apps.mas.graph.plan_graph import _retrieve_rag_evidence_for_subtask
    from apps.mas.rag.retriever import RetrievedChunk

    seed = [
        RetrievedChunk(
            id="seed_chunk_0",
            doc_id="seed_doc",
            title="1995 Ginebra San Miguel season",
            text="Darryl Prue was sent home after July 9, 1995.",
            url="https://en.wikipedia.org/wiki/1995_Ginebra_San_Miguel_season",
            chunk_idx=0,
            score=0.016,  # High score
            retrieval_method="fusion",
        )
    ]

    # Mock retriever that should NOT be called if seed_chunks is provided
    class NoCallRetriever:
        def fusion_search(self, query: str, k: int = 5):
            raise AssertionError("fusion_search should not be called when seed_chunks provided")

    retriever = NoCallRetriever()
    st = SubTask(id="step1", role="qa", instruction="```json malformed instruction", depends_on=[])
    events: List[Tuple[str, str]] = []

    def time_left() -> float:
        return 100.0

    def emit(stage: str, content: str) -> None:
        events.append((stage, content))

    pack = _retrieve_rag_evidence_for_subtask(
        retriever=retriever,
        subtask=st,
        original_problem="When was Darryl Prue sent home?",
        rag_top_k=5,
        time_left=time_left,
        emit_thinking=emit,
        seed_chunks=seed,
    )

    assert pack is not None
    assert pack.chunks and pack.chunks[0].title == "1995 Ginebra San Miguel season"
    assert pack.chunks[0].score == 0.016

    # Should emit reuse event
    stages = [s for s, _ in events]
    assert any("reuse" in s for s in stages), f"Expected reuse event in {stages}"


