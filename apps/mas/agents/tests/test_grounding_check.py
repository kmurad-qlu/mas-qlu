from __future__ import annotations


def test_needs_grounding_repair_detects_missing_extracted_answer():
    from apps.mas.graph.plan_graph import _needs_grounding_repair

    assert _needs_grounding_repair("The answer is Foo", "Bar") is True
    assert _needs_grounding_repair("The answer is Bar", "Bar") is False


def test_format_grounding_fallback_includes_sources():
    from apps.mas.graph.plan_graph import _format_grounding_fallback

    out = _format_grounding_fallback("The Life of a Showgirl", ["https://example.com/a", "https://example.com/b"])
    assert "The Life of a Showgirl" in out
    assert "https://example.com/a" in out


