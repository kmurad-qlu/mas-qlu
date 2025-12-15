"""
Safe web page fetching + lightweight HTML-to-text extraction.

Used to augment DDG snippets with in-page content (tables/sections) so the
reasoning system can do multi-hop / constraint-based inference on the page.

Supports concurrent URL fetching for improved performance.
"""

from __future__ import annotations

import concurrent.futures
import html
import re
import socket
import threading
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests


# Configuration for concurrent fetching
MAX_CONCURRENT_FETCHES = 4


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: int
    content_type: str
    text: str
    bytes_read: int
    error: Optional[str] = None


def _is_public_ip(hostname: str) -> bool:
    """
    SSRF guard: reject localhost/private/reserved IPs resolved from hostname.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except Exception:
        return False

    for family, _, _, _, sockaddr in infos:
        try:
            if family == socket.AF_INET:
                ip = sockaddr[0]
            elif family == socket.AF_INET6:
                ip = sockaddr[0]
            else:
                continue
            ipa = ip_address(ip)
            if (
                ipa.is_private
                or ipa.is_loopback
                or ipa.is_link_local
                or ipa.is_multicast
                or ipa.is_reserved
                or ipa.is_unspecified
            ):
                return False
        except Exception:
            return False
    return True


def _validate_url(url: str) -> Tuple[bool, str]:
    if not url or not url.strip():
        return False, "empty url"
    p = urlparse(url.strip())
    if p.scheme not in ("http", "https"):
        return False, f"unsupported scheme: {p.scheme}"
    if not p.netloc:
        return False, "missing hostname"
    host = p.hostname or ""
    if not host:
        return False, "missing hostname"
    # Block obvious local targets early (still do DNS guard below).
    if host in {"localhost"} or host.endswith(".local"):
        return False, "blocked hostname"
    if not _is_public_ip(host):
        return False, "blocked ip range / unresolved host"
    return True, ""


_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>")
_TAG_RE = re.compile(r"(?is)<[^>]+>")


def html_to_text(html_str: str) -> str:
    """
    Very lightweight HTML -> text extraction.
    Keeps table text reasonably (because tags are stripped but cell contents remain).
    """
    if not html_str:
        return ""
    s = _SCRIPT_STYLE_RE.sub(" ", html_str)
    # Add line breaks around common block boundaries before stripping tags.
    s = re.sub(r"(?is)</(p|div|br|tr|li|h1|h2|h3|h4|h5|h6|table|ul|ol)>", "\n", s)
    s = re.sub(r"(?is)<br\\s*/?>", "\n", s)
    s = _TAG_RE.sub(" ", s)
    s = html.unescape(s)
    # Normalize whitespace
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _keywords_from_query(query: str) -> List[str]:
    q = (query or "").strip()
    if not q:
        return []
    # Prefer proper nouns / capitalized sequences.
    caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\b", q)
    kws: List[str] = []
    for c in sorted(caps, key=len, reverse=True):
        if c not in kws:
            kws.append(c)
    # Add informative tokens (basic stopword removal).
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "from", "by",
        "is", "was", "were", "are", "be", "been", "being",
        "what", "when", "where", "who", "why", "how",
    }
    toks = re.findall(r"[A-Za-z0-9]{3,}", q.lower())
    for t in toks:
        if t in stop:
            continue
        if t not in (k.lower() for k in kws):
            kws.append(t)
    return kws[:12]


def select_relevant_passages(
    text: str,
    query: str,
    max_chars: int = 4500,
    window_before: int = 400,
    window_after: int = 900,
) -> str:
    """
    Select a small set of windows around query keywords so we keep the most relevant
    parts of a long page (e.g., Wikipedia tables/sections).
    """
    t = (text or "").strip()
    if not t:
        return ""
    if len(t) <= max_chars:
        return t

    kws = _keywords_from_query(query)
    if not kws:
        return t[:max_chars].rstrip()

    lower = t.lower()
    hits: List[Tuple[int, int, int]] = []  # (score, start, end)
    for kw in kws:
        if not kw:
            continue
        kw_l = kw.lower()
        idx = 0
        while True:
            pos = lower.find(kw_l, idx)
            if pos < 0:
                break
            start = max(0, pos - window_before)
            end = min(len(t), pos + window_after)
            window = lower[start:end]
            score = sum(1 for k in kws if k.lower() in window)
            hits.append((score, start, end))
            idx = pos + max(1, len(kw_l))

    if not hits:
        return t[:max_chars].rstrip()

    # Merge windows, preferring higher-scoring ones.
    hits.sort(key=lambda x: (x[0], -(x[2] - x[1])), reverse=True)
    chosen: List[Tuple[int, int]] = []
    for score, s, e in hits:
        # Merge if overlaps existing chosen
        overlapped = False
        for i, (cs, ce) in enumerate(chosen):
            if s <= ce and e >= cs:
                chosen[i] = (min(cs, s), max(ce, e))
                overlapped = True
                break
        if not overlapped:
            chosen.append((s, e))
        # Stop once we have enough raw coverage
        if sum(ce - cs for cs, ce in chosen) > max_chars * 1.8:
            break

    chosen.sort(key=lambda x: x[0])
    out_parts: List[str] = []
    used = 0
    for s, e in chosen:
        chunk = t[s:e].strip()
        if not chunk:
            continue
        # Add separator if not contiguous
        if out_parts:
            out_parts.append("\n...\n")
            used += 5
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            # Try to keep a keyword occurrence inside the truncated excerpt.
            cl = chunk.lower()
            first_hit = -1
            for kw in kws:
                pos = cl.find(kw.lower())
                if pos >= 0:
                    first_hit = pos if first_hit < 0 else min(first_hit, pos)
            if first_hit >= 0 and remaining > 80:
                start2 = max(0, first_hit - remaining // 3)
                chunk = chunk[start2 : start2 + remaining].strip()
            else:
                chunk = chunk[:remaining].rstrip()
            chunk = chunk.rstrip() + "…"
        out_parts.append(chunk)
        used += len(chunk)
        if used >= max_chars:
            break
    return "".join(out_parts).strip()


def fetch_url_text(
    url: str,
    *,
    timeout_s: float = 10.0,
    max_bytes: int = 2_000_000,
    max_redirects: int = 3,
    user_agent: str = "MAS-RAG-Fetcher/1.0",
    cache: Optional[Dict[str, FetchResult]] = None,
) -> FetchResult:
    """
    Fetch a URL safely with SSRF guardrails and size/time limits.
    Returns best-effort text (HTML converted to text).
    """
    cache = cache if cache is not None else {}
    u0 = (url or "").strip()
    if u0 in cache:
        return cache[u0]

    ok, err = _validate_url(u0)
    if not ok:
        res = FetchResult(url=u0, final_url=u0, status_code=0, content_type="", text="", bytes_read=0, error=err)
        cache[u0] = res
        return res

    current = u0
    session = requests.Session()
    headers = {"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5"}

    for _ in range(max_redirects + 1):
        ok, err = _validate_url(current)
        if not ok:
            res = FetchResult(url=u0, final_url=current, status_code=0, content_type="", text="", bytes_read=0, error=err)
            cache[u0] = res
            return res

        try:
            r = session.get(
                current,
                headers=headers,
                timeout=(min(timeout_s, 6.0), timeout_s),
                stream=True,
                allow_redirects=False,
            )
        except Exception as e:
            res = FetchResult(url=u0, final_url=current, status_code=0, content_type="", text="", bytes_read=0, error=str(e)[:200])
            cache[u0] = res
            return res

        # Handle redirects manually so we can re-validate host/IP.
        if 300 <= r.status_code < 400:
            loc = r.headers.get("Location") or ""
            if not loc:
                break
            current = urljoin(current, loc)
            continue

        content_type = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        bytes_read = 0
        chunks: List[bytes] = []
        try:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                bytes_read += len(chunk)
                if bytes_read > max_bytes:
                    break
                chunks.append(chunk)
        finally:
            try:
                r.close()
            except Exception:
                pass

        raw = b"".join(chunks)
        # requests may not expose encoding reliably when streaming; fallback to utf-8.
        try:
            text = raw.decode(r.encoding or "utf-8", errors="replace")
        except Exception:
            text = raw.decode("utf-8", errors="replace")

        if content_type in {"text/plain"}:
            out_text = re.sub(r"\n{3,}", "\n\n", text).strip()
        else:
            out_text = html_to_text(text)

        res = FetchResult(
            url=u0,
            final_url=str(r.url) if getattr(r, "url", None) else current,
            status_code=int(getattr(r, "status_code", 0) or 0),
            content_type=content_type,
            text=out_text,
            bytes_read=bytes_read,
            error=None if 200 <= r.status_code < 300 else f"http {r.status_code}",
        )
        cache[u0] = res
        return res

    res = FetchResult(url=u0, final_url=current, status_code=0, content_type="", text="", bytes_read=0, error="too many redirects")
    cache[u0] = res
    return res


def fetch_urls_concurrent(
    urls: List[str],
    *,
    timeout_s: float = 10.0,
    max_bytes: int = 2_000_000,
    max_workers: int = MAX_CONCURRENT_FETCHES,
    user_agent: str = "MAS-RAG-Fetcher/1.0",
    cache: Optional[Dict[str, FetchResult]] = None,
    emit_thinking: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, FetchResult]:
    """
    Fetch multiple URLs concurrently for improved performance.
    
    Args:
        urls: List of URLs to fetch
        timeout_s: Timeout per request in seconds
        max_bytes: Maximum bytes to read per URL
        max_workers: Maximum concurrent requests
        user_agent: User agent string
        cache: Optional shared cache for results
        emit_thinking: Optional callback for progress updates
    
    Returns:
        Dict mapping URL -> FetchResult
    """
    if not urls:
        return {}
    
    cache = cache if cache is not None else {}
    emit = emit_thinking or (lambda s, c: None)
    
    # Filter out already cached URLs
    to_fetch: List[str] = []
    results: Dict[str, FetchResult] = {}
    for url in urls:
        u = (url or "").strip()
        if not u:
            continue
        if u in cache:
            results[u] = cache[u]
        elif u not in to_fetch:
            to_fetch.append(u)
    
    if not to_fetch:
        return results
    
    # Limit concurrent requests
    actual_workers = min(len(to_fetch), max_workers)
    emit("concurrent_fetch_start", f"Fetching {len(to_fetch)} URLs with {actual_workers} workers")
    
    # Use a thread-safe lock for cache updates
    cache_lock = threading.Lock()
    
    def fetch_single(url: str) -> Tuple[str, FetchResult]:
        """Fetch a single URL and return (url, result)."""
        result = fetch_url_text(
            url,
            timeout_s=timeout_s,
            max_bytes=max_bytes,
            user_agent=user_agent,
            cache={},  # Don't use shared cache in thread (we'll update after)
        )
        return (url, result)
    
    # Execute fetches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
        future_to_url = {executor.submit(fetch_single, url): url for url in to_fetch}
        
        try:
            for future in concurrent.futures.as_completed(future_to_url, timeout=timeout_s * 2):
                try:
                    url, result = future.result(timeout=2.0)
                    results[url] = result
                    
                    # Update shared cache
                    with cache_lock:
                        cache[url] = result
                    
                    # Emit progress
                    status = "OK" if not result.error else f"Error: {result.error[:50]}"
                    emit("concurrent_fetch_result", f"{url[:60]}... -> {status}")
                    
                except Exception as e:
                    url = future_to_url.get(future, "unknown")
                    error_result = FetchResult(
                        url=url,
                        final_url=url,
                        status_code=0,
                        content_type="",
                        text="",
                        bytes_read=0,
                        error=f"fetch error: {str(e)[:100]}",
                    )
                    results[url] = error_result
                    with cache_lock:
                        cache[url] = error_result
                    emit("concurrent_fetch_error", f"{url[:60]}... -> {str(e)[:50]}")
                    
        except concurrent.futures.TimeoutError:
            emit("concurrent_fetch_timeout", f"Overall timeout after {timeout_s * 2}s, got {len(results)} results")
    
    emit("concurrent_fetch_complete", f"Fetched {len(results)}/{len(urls)} URLs")
    return results


def fetch_and_extract_relevant(
    urls: List[str],
    query: str,
    *,
    timeout_s: float = 10.0,
    max_bytes: int = 2_000_000,
    max_chars_per_url: int = 4500,
    max_workers: int = MAX_CONCURRENT_FETCHES,
    cache: Optional[Dict[str, FetchResult]] = None,
    emit_thinking: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, str]:
    """
    Fetch multiple URLs concurrently and extract relevant passages from each.
    
    Args:
        urls: List of URLs to fetch
        query: Query to use for extracting relevant passages
        timeout_s: Timeout per request in seconds
        max_bytes: Maximum bytes to read per URL
        max_chars_per_url: Maximum chars to extract per URL
        max_workers: Maximum concurrent requests
        cache: Optional shared cache for results
        emit_thinking: Optional callback for progress updates
    
    Returns:
        Dict mapping URL -> extracted relevant text (empty string if failed)
    """
    if not urls:
        return {}
    
    # Fetch all URLs concurrently
    fetch_results = fetch_urls_concurrent(
        urls=urls,
        timeout_s=timeout_s,
        max_bytes=max_bytes,
        max_workers=max_workers,
        cache=cache,
        emit_thinking=emit_thinking,
    )
    
    # Extract relevant passages from each result
    extracted: Dict[str, str] = {}
    for url, result in fetch_results.items():
        if result.error or not result.text:
            extracted[url] = ""
            continue
        
        try:
            passage = select_relevant_passages(
                result.text,
                query,
                max_chars=max_chars_per_url,
            )
            extracted[url] = passage
        except Exception:
            extracted[url] = result.text[:max_chars_per_url] if result.text else ""
    
    return extracted


