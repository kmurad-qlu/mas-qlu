"""
RAG evidence utilities.

Defines a first-class artifact for retrieved chunks so we can:
- inject retrieved context into agent prompts
- render retrieved results in the UI thinking panel
- (optionally) include citations in final answers
- detect low-quality RAG results and suggest fallbacks
- expand queries for better retrieval (person names, entities)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .retriever import RetrievedChunk


# Threshold for "low quality" RAG results - all scores below this indicate no relevant content
RAG_LOW_QUALITY_THRESHOLD = 0.012

# Threshold for "acceptable" RAG results - at least one result should be above this
RAG_ACCEPTABLE_THRESHOLD = 0.015


def detect_rag_quality(chunks: List[RetrievedChunk]) -> Tuple[str, float]:
    """
    Detect the quality of RAG retrieval results.
    
    Returns:
        Tuple of (quality_level, max_score) where quality_level is:
        - "good": At least one result with score >= 0.015
        - "marginal": All results between 0.012 and 0.015
        - "poor": All results below 0.012 (likely no relevant content indexed)
    """
    if not chunks:
        return "poor", 0.0
    
    max_score = max(ch.score for ch in chunks)
    
    if max_score >= RAG_ACCEPTABLE_THRESHOLD:
        return "good", max_score
    elif max_score >= RAG_LOW_QUALITY_THRESHOLD:
        return "marginal", max_score
    else:
        return "poor", max_score


def expand_person_query(query: str) -> List[str]:
    """
    Expand a query about a person to try finding related pages.
    
    For questions like "What is the height of Sonny Cabatu?", generate
    alternative queries that might find the person in team rosters,
    player lists, or biographical pages.
    
    Returns:
        List of expanded query strings to try
    """
    q = query.strip()
    q_lower = q.lower()
    expanded = []
    
    # Detect if this is a question about a person
    person_indicators = [
        "height of", "age of", "born", "died", "nationality of",
        "who is", "who was", "biography", "career of",
        "position of", "team of", "played for",
    ]
    is_person_query = any(ind in q_lower for ind in person_indicators)
    
    # Try to extract the person's name
    # Pattern: "height of X", "who is X", etc.
    name_patterns = [
        r"height of\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
        r"age of\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
        r"who is\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
        r"who was\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
        r"biography of\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
        # Also try to find capitalized names directly
        r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
    ]
    
    extracted_name = None
    for pattern in name_patterns:
        match = re.search(pattern, q)
        if match:
            extracted_name = match.group(1).strip()
            break
    
    if extracted_name and is_person_query:
        # Generate expanded queries
        name = extracted_name
        
        # Sports-related expansions (common for height/position queries)
        if "height" in q_lower or "position" in q_lower or "team" in q_lower:
            expanded.extend([
                f"{name} basketball player",
                f"{name} PBA roster",
                f"{name} basketball roster",
                f"{name} player profile",
                f"{name} sports",
            ])
        
        # General biographical expansions
        expanded.extend([
            f"{name} biography",
            f"{name} career",
            name,  # Just the name itself
        ])
    
    return expanded


def should_suggest_web_fallback(
    chunks: List[RetrievedChunk],
    web_enabled: bool,
) -> Tuple[bool, str]:
    """
    Determine if we should suggest enabling web search as a fallback.
    
    Returns:
        Tuple of (should_suggest, reason_message)
    """
    quality, max_score = detect_rag_quality(chunks)
    
    if quality == "poor" and not web_enabled:
        return True, (
            f"RAG retrieval quality is low (max_score={max_score:.4f}). "
            "The requested information may not be in the indexed knowledge base. "
            "Enable web search for better results."
        )
    
    if quality == "marginal" and not web_enabled:
        return True, (
            f"RAG retrieval quality is marginal (max_score={max_score:.4f}). "
            "Consider enabling web search for more comprehensive results."
        )
    
    return False, ""


@dataclass
class RAGEvidencePack:
    query: str
    method: str
    top_k: int
    chunks: List[RetrievedChunk]

    def format_context(
        self,
        max_length: int = 3500,
        include_titles: bool = True,
    ) -> str:
        """
        Format retrieved chunks for model context injection.
        """
        if not self.chunks:
            return ""
        parts: List[str] = []
        total = 0
        for i, ch in enumerate(self.chunks, 1):
            header = f"[R{i}] {ch.title}".strip() if (include_titles and ch.title) else f"[R{i}]"
            remaining = max_length - total - len(header) - 10
            if remaining <= 0:
                break
            body = (ch.text or "").strip()
            if len(body) > remaining:
                body = body[:remaining].rstrip()
            entry = f"{header}\n{body}"
            parts.append(entry)
            total += len(entry)
        return "\n\n".join(parts)

    def format_citations(self, max_items: int = 6) -> str:
        """
        Format citations mapping [R#] -> URL for final answer use.
        """
        if not self.chunks:
            return ""
        lines: List[str] = []
        for i, ch in enumerate(self.chunks[:max_items], 1):
            url = (ch.url or "").strip()
            title = (ch.title or "").strip()
            if url and title:
                lines.append(f"[R{i}] {title} — {url}")
            elif url:
                lines.append(f"[R{i}] {url}")
            elif title:
                lines.append(f"[R{i}] {title}")
        return "\n".join(lines).strip()

    def emit_thinking_summary(
        self,
        emit: Callable[[str, str], None],
        stage_prefix: str = "rag",
        snippet_chars: int = 220,
        max_items: int = 8,
    ) -> None:
        """
        Emit a compact markdown summary of retrieved chunks to the thinking log.
        """
        q = (self.query or "").strip()
        emit(f"{stage_prefix}_query", f"query='{q[:200]}' method={self.method} k={self.top_k} returned={len(self.chunks)}")
        if not self.chunks:
            emit(f"{stage_prefix}_chunks", "_(no retrieved chunks)_")
            return
        lines: List[str] = []
        for i, ch in enumerate(self.chunks[:max_items], 1):
            title = (ch.title or "").strip()[:80]
            url = (ch.url or "").strip()
            score = ch.score
            snippet = " ".join((ch.text or "").strip().split())
            if len(snippet) > snippet_chars:
                snippet = snippet[:snippet_chars].rstrip() + "…"
            if title and url:
                lines.append(f"- [R{i}] **{title}** (score={score:.3f}) — {url}\n  - {snippet}")
            elif url:
                lines.append(f"- [R{i}] (score={score:.3f}) — {url}\n  - {snippet}")
            else:
                lines.append(f"- [R{i}] (score={score:.3f})\n  - {snippet}")
        emit(f"{stage_prefix}_chunks", "\n".join(lines))


def build_rag_evidence_pack(
    *,
    query: str,
    method: str,
    top_k: int,
    chunks: List[RetrievedChunk],
    emit: Optional[Callable[[str, str], None]] = None,
) -> RAGEvidencePack:
    pack = RAGEvidencePack(query=query, method=method, top_k=top_k, chunks=chunks)
    if emit is not None:
        pack.emit_thinking_summary(emit)
    return pack


