from __future__ import annotations


class FakeResult:
    def __init__(self, text: str):
        self.text = text


class FakeClient:
    def __init__(self):
        self.last_messages = None
        self.last_model = None
        self.last_temperature = None

    def complete_chat(self, messages, model=None, temperature=None, **kwargs):
        self.last_messages = messages
        self.last_model = model
        self.last_temperature = temperature
        return FakeResult("OK")


def test_supervisor_includes_rag_for_factual_and_not_for_numeric():
    from apps.mas.agents.supervisor import SupervisorAgent, SubTask

    client = FakeClient()
    sup = SupervisorAgent(client=client)

    results = [(SubTask(role="qa", instruction="Who wrote The Hobbit?"), [("m", "J.R.R. Tolkien")])]

    rag_evidence = "RAG Evidence (retrieved chunks):\n[R1] The Hobbit\n..."
    rag_citations = "[R1] The Hobbit — http://wiki/the_hobbit"

    # Factual: should include RAG blocks
    _ = sup.synthesize(
        "Who wrote The Hobbit?",
        results,
        rag_evidence=rag_evidence,
        rag_citations=rag_citations,
    )
    user_content = client.last_messages[1]["content"]
    assert "RAG Evidence" in user_content
    assert "Citations (" in user_content
    assert "[R1]" in user_content

    # Numeric: should NOT include RAG blocks (strict numeric output policy)
    _ = sup.synthesize(
        "Compute 2+2",
        [(SubTask(role="math", instruction="Compute 2+2"), [("m", "#### 4")])],
        rag_evidence=rag_evidence,
        rag_citations=rag_citations,
    )
    user_content2 = client.last_messages[1]["content"]
    assert "RAG Evidence" not in user_content2
    assert "Citations (" not in user_content2


