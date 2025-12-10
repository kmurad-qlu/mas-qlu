"""
Web Search Tool using DuckDuckGo.
"""
from duckduckgo_search import DDGS
from typing import List, Dict

def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web for a query.
    Returns a formatted string of results.
    """
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No results found."
        
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No Title")
            body = r.get("body", "No description.")
            href = r.get("href", "#")
            formatted.append(f"[{i}] {title}\n    {body}\n    Source: {href}")
            
        return "\n\n".join(formatted)
    except Exception as e:
        return f"[Error] Search failed: {str(e)}"

