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


def test_ok_critique_normalization():
    from apps.mas.graph.plan_graph import _is_ok_critique

    assert _is_ok_critique("OK") is True
    assert _is_ok_critique("OK.") is True
    assert _is_ok_critique("OK. Looks good and complete.") is True
    assert _is_ok_critique("OK, but missing the ban detail.") is False


def test_json_repair_guard_helpers():
    from apps.mas.graph.plan_graph import _json_allowed, _looks_like_json_output

    assert _looks_like_json_output('```json\\n[{\"a\": 1}]\\n```') is True
    assert _looks_like_json_output('[{\"a\": 1}]') is True
    assert _looks_like_json_output('Answer: hello') is False

    assert _json_allowed("Return JSON") is True
    assert _json_allowed("List the planets") is True  # multi_quantity heuristic
    assert _json_allowed("Which is Ranveer Singh's latest movie?") is False


