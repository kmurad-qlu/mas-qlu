"""
Web Search Tool using DuckDuckGo.

Provides real-time web search for current events and recent information.
Uses news search for better current events coverage.
"""
from duckduckgo_search import DDGS
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union
import re


@dataclass
class WebResult:
    """
    Structured web search result.

    `source` is one of: NEWS, WEB
    `date` is best-effort (YYYY-MM-DD) when available (usually from NEWS results).
    """
    title: str
    body: str
    url: str
    source: str
    date: Optional[str] = None


def _format_results(results: List[WebResult], max_results: int) -> str:
    if not results:
        return "No results found."

    formatted: List[str] = []
    for i, r in enumerate(results[:max_results], 1):
        date = f" ({r.date})" if r.date else ""
        formatted.append(
            f"[{i}] [{r.source}]{date} {r.title}\n"
            f"    {r.body}\n"
            f"    URL: {r.url}"
        )
    return "\n\n".join(formatted)


def search_web(
    query: str,
    max_results: int = 5,
    return_format: Literal["text", "results", "both"] = "text",
) -> Union[str, List[WebResult], Tuple[str, List[WebResult]]]:
    """
    Search the web for a query using DuckDuckGo.
    
    Uses news search first for current events queries, then falls back to text search.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return
        return_format:
            - "text": return formatted string (backwards compatible default)
            - "results": return List[WebResult]
            - "both": return (formatted_string, List[WebResult])
    
    Returns a formatted string of search results.
    """
    try:
        ddgs = DDGS()
        out: List[WebResult] = []
        
        # Try news search first - better for current events
        try:
            news_results = list(ddgs.news(query, max_results=max_results))
            if news_results:
                for r in news_results:
                    date_raw = r.get("date", "") or ""
                    date_norm: Optional[str] = None
                    if date_raw:
                        # Best-effort: ddgs often returns an ISO-ish string; keep YYYY-MM-DD if present
                        date_norm = date_raw[:10]

                    out.append(
                        WebResult(
                            title=r.get("title", "No Title"),
                            body=r.get("body", "No description."),
                            url=r.get("url", r.get("href", "#")),
                            source="NEWS",
                            date=date_norm,
                        )
                    )
        except Exception:
            pass  # News search might fail, continue to text search
        
        # If not enough results from news, supplement with text search
        if len(out) < max_results:
            try:
                text_results = list(ddgs.text(
                    query,
                    max_results=max_results - len(out),
                    backend="lite",  # Lite backend often works better
                ))
                for r in text_results:
                    # Skip non-English results (check for common Chinese/non-Latin chars)
                    title = r.get("title", "")
                    if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', title):
                        continue
                    out.append(
                        WebResult(
                            title=title or "No Title",
                            body=r.get("body", "No description."),
                            url=r.get("href", "#"),
                            source="WEB",
                            date=None,
                        )
                    )
            except Exception:
                pass
        
        if return_format == "results":
            return out[:max_results]
        text = _format_results(out, max_results=max_results)
        if return_format == "both":
            return text, out[:max_results]
        return text
    except Exception as e:
        if return_format == "results":
            return []
        if return_format == "both":
            return f"[Error] Search failed: {str(e)}", []
        return f"[Error] Search failed: {str(e)}"

