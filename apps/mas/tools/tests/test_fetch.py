from __future__ import annotations

from apps.mas.tools.fetch import _validate_url, html_to_text, select_relevant_passages


def test_validate_url_rejects_non_http():
    ok, err = _validate_url("file:///etc/passwd")
    assert not ok
    assert "unsupported scheme" in err


def test_html_to_text_strips_tags_and_scripts():
    html = "<html><head><script>var x=1;</script></head><body><h1>Title</h1><p>Hello <b>world</b></p></body></html>"
    text = html_to_text(html)
    assert "var x" not in text
    assert "Title" in text
    assert "Hello world" in text


def test_select_relevant_passages_finds_keyword_window():
    long = ("noise " * 200) + "Darryl Prue played July 1–9. " + ("noise " * 200) + "Alexander Coles played July 16."
    out = select_relevant_passages(long, "When was Darryl Prue sent home?", max_chars=300)
    assert "Darryl Prue" in out


