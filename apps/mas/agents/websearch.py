"""
WebSearchAgent - Dedicated agent for real-time web search.

This agent is optimized for answering questions that require current/real-time
information by searching the web and synthesizing results into accurate answers.

Unlike the ScientistAgent, this agent focuses solely on web search without
code execution, making it faster and more reliable for current events queries.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable, Tuple
import re
from datetime import datetime
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ..infra.openrouter.client import OpenRouterClient
from ..tools.search import WebResult, search_web
from ..tools.fetch import fetch_url_text, select_relevant_passages, FetchResult


# Get current date for search queries
def _get_current_date_str() -> str:
    """Get current date string for search queries."""
    now = datetime.now()
    return f"{now.strftime('%B')} {now.year}"


# System prompt optimized for web search result synthesis
SYSTEM_WEBSEARCH = """You are a Web Search Analyst Agent. Your ONLY job is to:
1. Analyze web search results provided to you
2. Extract the most accurate and up-to-date information
3. Synthesize a DETAILED, comprehensive answer

CRITICAL RULES:
- Base your answer ONLY on the search results provided - DO NOT use your training data
- Your training data is OUTDATED (cutoff June 2024). The search results are CURRENT.
- If search results say someone died, is dead, was killed - REPORT THAT. Do NOT claim they are alive.
- If search results show a new person in a position - REPORT THE NEW PERSON. 
- TRUST THE SEARCH RESULTS over your prior knowledge
- Prefer the MOST RECENT dated articles (look for dates in the text)
- If results conflict, prefer major news outlets (MSN, CNN, BBC, Reuters, AP News, etc.)
- If no relevant results, say "No current information found"
- Include dates when available to show recency

OUTPUT FORMAT - BE DETAILED:
1. Start with a clear, direct answer to the question
2. Add supporting details and context from the search results
3. Include key facts, dates, statistics mentioned in the results
4. Mention source attribution for credibility
5. Use markdown formatting (bold, headers) for readability

Example structure:
**Direct Answer**: [The main answer]

**Key Details**:
- [Supporting fact 1]
- [Supporting fact 2]

**Source**: [Attribution to news sources]

Do NOT second-guess or contradict the search results with your training data.
"""

SEARCH_QUERY_PROMPT = """Generate 3-4 effective web search queries to find information about this question.

Question: {question}

Current date: {current_date}

Output ONLY the search queries, one per line. No explanations or numbering.

CRITICAL TIPS for effective queries:
- Include the current year ({year}) ONLY if the question is about current events / latest status.
- For MUSICIANS/ARTISTS: add "discography", "Spotify", "new album", "latest release", "singer", "artist"
- For SPORTS: add the sport name and "champion", "winner", year
- For PEOPLE: add their profession/role to disambiguate (e.g., "singer", "politician", "CEO")
- For products/software: add "latest version", "new release"
- For positions/roles: add "current" and the year
- Be SPECIFIC with full names AND context to avoid generic results
- If asking about an artist's work, include platform names like "Spotify", "Apple Music"

Example for musician queries:
- "Hassan Raheem latest album Spotify 2025"
- "Hassan Raheem discography new release"
- "Hassan Raheem singer Pakistan new album"
"""

SYNTHESIS_PROMPT = """Based on the web search results below, provide a DETAILED answer to the user's question.

QUESTION: {question}

SEARCH RESULTS:
{search_results}

CRITICAL INSTRUCTIONS:
- Your training data is OUTDATED (June 2024). These search results are from {current_date} and are CURRENT.
- If the search results say someone died, was killed, etc. - REPORT THAT as fact.
- If the search results show a new person in a position - REPORT THAT, not who you think it is.
- Do NOT contradict the search results with your prior knowledge.
- Look for dates in the articles to determine the most recent information.

REQUIRED OUTPUT FORMAT:
1. **Direct Answer**: Start with a clear, bold answer to the main question
2. **Key Details**: Extract ALL relevant facts, statistics, dates from the search results
3. **Context**: Provide background information if available in the results
4. **Sources**: Cite the news sources for credibility

Be COMPREHENSIVE - include all useful information from the search results, not just the bare minimum.
Use markdown formatting for readability (bold, bullet points, headers).

If the results don't contain relevant information, state that clearly.
"""

@dataclass
class WebEvidencePack:
    """
    Container for web evidence used by the rest of the pipeline.
    This is intentionally compatible with earlier text-first wiring, but now also includes
    structured results and (when possible) a deterministic extracted answer.
    """
    question: str
    intent: str
    entity: Optional[str]
    queries: List[str]
    results: List[WebResult]
    combined_results: str

    # Deterministic extraction (for lookup intents). If None, downstream should not guess.
    extracted_answer: Optional[str] = None
    extracted_confidence: float = 0.0
    extracted_sources: List[str] = field(default_factory=list)

    # Debugging: candidate list (highest score first)
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    # Optional: fetched page extracts (for deeper reasoning beyond snippets)
    fetched_pages: List[Dict[str, Any]] = field(default_factory=list)
    fetched_combined: str = ""


class WebSearchAgent:
    """
    Dedicated agent for real-time web search queries.
    
    Optimized for:
    - Current events (deaths, elections, appointments)
    - Latest versions/releases of products
    - Recent news and developments
    - Current status of people/organizations
    """
    
    def __init__(
        self,
        client: OpenRouterClient,
        model_name: str = "mistralai/mistral-large-2512",
        max_searches: int = 3,
        results_per_search: int = 5,
    ):
        """
        Initialize WebSearchAgent.
        
        Args:
            client: OpenRouter client for LLM calls
            model_name: Model to use for query generation and synthesis
            max_searches: Maximum number of search queries to run
            results_per_search: Number of results per search query
        """
        self.client = client
        self.model_name = model_name
        self.max_searches = max_searches
        self.results_per_search = results_per_search
        self._thinking_callback: Optional[Callable[[str, str], None]] = None
        self._current_year = datetime.now().year
        self._current_date = _get_current_date_str()
    
    def set_thinking_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set callback for emitting thinking/debug events."""
        self._thinking_callback = callback
    
    def _emit(self, stage: str, content: str) -> None:
        """Emit a thinking event if callback is set."""
        if self._thinking_callback:
            self._thinking_callback(stage, content)
    
    def _generate_search_queries(self, question: str, bias_current_year: bool = True) -> List[str]:
        """Generate optimal search queries for the question."""
        self._emit("websearch_query_gen", f"Generating search queries for: {question[:100]}...")
        
        try:
            prompt = SEARCH_QUERY_PROMPT.format(
                question=question,
                current_date=self._current_date,
                year=self._current_year
            )
            result = self.client.complete_chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You generate web search queries. Output only queries, one per line. No numbering."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300,
            )
            
            response = result.text
            
            # Parse queries from response
            queries = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove numbering like "1.", "2.", "-", etc.
                line = re.sub(r'^[\d\-\*\.\)]+\s*', '', line)
                # Remove quotes
                line = line.strip('"\'')
                if line and len(line) > 5:
                    queries.append(line)
            
            # Ensure we have at least one query
            if not queries:
                queries = self._generate_fallback_queries(question, bias_current_year=bias_current_year)
            
            # Replace old years with current year only for genuinely current-events queries
            if bias_current_year:
                queries = [self._fix_year_in_query(q) for q in queries]
            
            self._emit("websearch_queries", f"Search queries: {queries[:self.max_searches]}")
            return queries[:self.max_searches]
            
        except Exception as e:
            self._emit("websearch_query_error", f"Query generation failed: {str(e)[:100]}, using fallback queries")
            return self._generate_fallback_queries(question, bias_current_year=bias_current_year)
    
    def _generate_fallback_queries(self, question: str, bias_current_year: bool = True) -> List[str]:
        """Generate fallback queries when LLM query generation fails."""
        # Extract key terms from question
        q = question.lower()
        
        # Remove common question words
        for word in ["what is", "who is", "when did", "which is", "where is", "how", "the", "a", "an"]:
            q = q.replace(word, "")
        q = q.strip("? ").strip()
        if bias_current_year:
            return [
                f"{q} {self._current_year} news",
                f"{q} {self._current_date}",
                f"{q} latest news today"
            ]
        # Historical/general fallback: do not force current-year framing.
        return [
            q,
            f"{q} wikipedia",
            f"\"{q}\"",
        ]
    
    def _generate_alternative_queries(self, question: str, failed_queries: List[str]) -> List[str]:
        """
        Generate alternative queries when initial searches return irrelevant results.
        Uses multi-hop reasoning to try different approaches.
        """
        q = question.lower()
        alternatives = []
        
        # Detect if it's about a person/artist
        music_keywords = ["album", "song", "music", "singer", "artist", "release", "track", "discography"]
        is_music_query = any(kw in q for kw in music_keywords)
        
        # Better name extraction - look for capitalized words or proper nouns
        import re as _re
        
        # Remove question words and common words to find the entity name
        clean_q = q
        for word in ["which is the", "what is the", "who is the", "latest", "new", "current", 
                     "album of", "song by", "album by", "music by", "release by"]:
            clean_q = clean_q.replace(word, "")
        clean_q = clean_q.strip("? ").strip()
        
        # If we have the original question, try to find proper nouns
        orig_q = question
        # Look for consecutive capitalized words (likely a name)
        name_match = _re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', orig_q)
        if name_match:
            # Use the longest match (likely full name)
            name = max(name_match, key=len)
        else:
            # Fallback to cleaned question
            name = clean_q
        
        self._emit("websearch_name_extracted", f"Extracted entity name: '{name}' from question")
        
        if is_music_query or "album" in q or "song" in q:
            # Music-specific alternative queries with better name
            alternatives = [
                f'"{name}" Pakistani singer latest album',
                f'"{name}" discography Spotify 2024 2025',
                f'"{name}" artist new music release',
                f'site:spotify.com "{name}" artist',
            ]
        else:
            # General alternative queries with more context
            alternatives = [
                f'"{name}" {self._current_year} latest news',
                f'{name} current {self._current_date}',
                f'site:wikipedia.org "{name}"',
            ]
        
        # Remove any queries we already tried
        alternatives = [a for a in alternatives if a.lower() not in [fq.lower() for fq in failed_queries]]
        
        return alternatives[:4]
    
    def _fix_year_in_query(self, query: str) -> str:
        """Replace old years with current year in query."""
        # Replace years from 2020-2024 with current year
        for old_year in range(2020, self._current_year):
            query = query.replace(str(old_year), str(self._current_year))
        return query

    def _detect_intent(self, question: str) -> str:
        """
        Lightweight intent classifier so we can run deterministic extraction.
        """
        q = question.strip().lower()
        if re.search(r"\b(latest|newest|most recent)\s+album\b", q) or re.search(r"\balbum\s+of\b", q):
            return "latest_album_of_artist"
        if re.search(r"\bwho\s+(sang|sings|performed|performs)\b", q) or re.search(r"\bwho\s+is\s+the\s+artist\s+of\b", q):
            return "who_sang_or_performed_song"
        if "current" in q or "reigning" in q:
            if any(k in q for k in ["champion", "winner", "title holder", "holder", "mvp", "president", "prime minister", "pm", "ceo", "chief minister", "cm"]):
                return "current_title_holder"
        return "general_fact_lookup"

    def _extract_artist_from_album_question(self, question: str) -> Optional[str]:
        q = question.strip().strip("?")
        patterns = [
            r"latest album of\s+(?P<name>.+)$",
            r"latest album by\s+(?P<name>.+)$",
            r"(?P<name>.+?)'s latest album$",
            r"(?P<name>.+?)\s+latest album$",
        ]
        for pat in patterns:
            m = re.search(pat, q, flags=re.IGNORECASE)
            if m:
                name = m.group("name").strip().strip("\"'")
                return name if name else None
        # Fallback: best-effort proper noun extraction
        caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", question)
        return max(caps, key=len) if caps else None

    def _extract_song_title(self, question: str) -> Optional[str]:
        # Prefer quoted strings
        quoted = re.findall(r"\"([^\"]{2,120})\"|'([^']{2,120})'|“([^”]{2,120})”", question)
        if quoted:
            for a, b, c in quoted:
                title = (a or b or c).strip()
                if title:
                    return title
        # Fallback: "who sang X" (until 'and' / '?')
        m = re.search(r"\bwho\s+(?:sang|performed)\s+(?P<title>.+?)(?:\s+and\s+|\?|$)", question, flags=re.IGNORECASE)
        if m:
            title = m.group("title").strip().strip("\"'")
            return title if title else None
        return None

    def _domain(self, url: str) -> str:
        try:
            return (urlparse(url).netloc or "").lower()
        except Exception:
            return ""

    def _trust_score(self, domain: str) -> float:
        """
        Heuristic domain trust scores for deterministic extraction.
        """
        if not domain:
            return 0.3
        high = {
            "open.spotify.com": 0.95,
            "spotify.com": 0.9,
            "music.apple.com": 0.92,
            "itunes.apple.com": 0.85,
            "genius.com": 0.82,
            "www.genius.com": 0.82,
            "discogs.com": 0.82,
            "www.discogs.com": 0.82,
            "musicbrainz.org": 0.85,
            "bandcamp.com": 0.88,
            "www.bandcamp.com": 0.88,
        }
        if domain in high:
            return high[domain]
        # Major news aggregators/outlets (moderate trust for factual claims)
        if any(d in domain for d in ["reuters.com", "apnews.com", "bbc.", "nytimes.com", "theguardian.com", "forbes.com", "rollingstone.com", "pitchfork.com", "billboard.com", "msn.com"]):
            return 0.7
        if "wikipedia.org" in domain:
            return 0.6
        return 0.45

    def _execute_searches(self, queries: List[str]) -> Tuple[str, List[WebResult]]:
        """
        Execute search queries and return:
        - a combined formatted text block (for thinking/context)
        - a de-duplicated list of structured WebResult objects
        """
        blocks: List[str] = []
        results: List[WebResult] = []

        for i, query in enumerate(queries):
            self._emit("websearch_executing", f"Executing search {i+1}/{len(queries)}: {query}")
            try:
                text, chunk = search_web(query, max_results=self.results_per_search, return_format="both")  # type: ignore[misc]
                blocks.append(f"=== Search Query: {query} ===\n{text}")
                results.extend(chunk)
                if chunk:
                    self._emit("websearch_result_content", f"Search '{query[:50]}...' returned:\n{text[:900]}...")
                else:
                    self._emit("websearch_no_results", f"Search {i+1} '{query[:50]}' returned no results")
            except Exception as e:
                self._emit("websearch_error", f"Search {i+1} failed: {str(e)[:120]}")

        # Dedupe by URL
        deduped: List[WebResult] = []
        seen: set[str] = set()
        for r in results:
            u = (r.url or "").strip()
            if not u or u in seen:
                continue
            seen.add(u)
            deduped.append(r)

        combined = "\n\n".join(blocks) if blocks else "No search results found."
        self._emit("websearch_total_results", f"Total unique results: {len(deduped)} (from {len(queries)} queries)")
        return combined, deduped

    def _pick_urls_to_fetch(self, results: List[WebResult], max_pages: int = 4) -> List[Tuple[str, str]]:
        """
        Pick a small set of URLs to fetch for deeper in-page evidence.
        Returns (url, title) pairs. Diversifies by domain.
        """
        scored: List[Tuple[float, str, str, str]] = []
        for r in results:
            u = (r.url or "").strip()
            if not u or u == "#" or u.startswith("javascript:"):
                continue
            dom = self._domain(u)
            trust = self._trust_score(dom)
            recency_bonus = 0.05 if (r.date or "").strip() else 0.0
            source_bonus = 0.02 if (r.source or "").upper() == "NEWS" else 0.0
            score = trust + recency_bonus + source_bonus
            scored.append((score, dom, u, (r.title or "").strip()))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked: List[Tuple[str, str]] = []
        seen_domains: set[str] = set()
        seen_urls: set[str] = set()
        for _, dom, u, title in scored:
            if u in seen_urls:
                continue
            # prefer domain diversity
            if dom in seen_domains and len(seen_domains) < max_pages:
                continue
            seen_urls.add(u)
            seen_domains.add(dom)
            picked.append((u, title))
            if len(picked) >= max_pages:
                break
        return picked

    def _fetch_pages_for_question(
        self,
        question: str,
        results: List[WebResult],
        max_pages: int = 4,
        max_chars_per_page: int = 4500,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Fetch top URLs and extract relevant passages for the question.
        """
        cache: Dict[str, FetchResult] = {}
        picked = self._pick_urls_to_fetch(results, max_pages=max_pages)
        if not picked:
            return [], ""

        pages: List[Dict[str, Any]] = []
        blocks: List[str] = []
        for url, title in picked:
            self._emit("web_fetch_start", f"Fetching URL for evidence: {url}")
            fr = fetch_url_text(url, timeout_s=10.0, max_bytes=2_000_000, cache=cache)
            if fr.error or not fr.text:
                self._emit("web_fetch_error", f"Fetch failed: {url} ({fr.error})")
                pages.append(
                    {
                        "url": url,
                        "final_url": fr.final_url,
                        "title": title,
                        "error": fr.error or "empty",
                        "chars": 0,
                    }
                )
                continue

            snippet = select_relevant_passages(fr.text, question, max_chars=max_chars_per_page)
            self._emit("web_fetch_ok", f"Fetched {url} chars={len(fr.text)} snippet_chars={len(snippet)} status={fr.status_code}")
            pages.append(
                {
                    "url": url,
                    "final_url": fr.final_url,
                    "title": title,
                    "status_code": fr.status_code,
                    "content_type": fr.content_type,
                    "chars": len(fr.text),
                    "snippet": snippet,
                }
            )
            if snippet:
                hdr = f"=== Fetched URL: {url} ==="
                if title:
                    hdr += f"\nTitle: {title}"
                blocks.append(f"{hdr}\n{snippet}")

        combined = ("\n\n".join(blocks)).strip()
        return pages, combined

    def _norm(self, s: str) -> str:
        s2 = re.sub(r"\s+", " ", (s or "")).strip().lower()
        s2 = re.sub(r"[“”\"'`]", "", s2)
        return s2

    def _extract_quoted_phrases(self, text: str) -> List[str]:
        # Double quotes, single quotes, and curly quotes
        found: List[str] = []
        for a, b, c in re.findall(r"\"([^\"]{2,120})\"|'([^']{2,120})'|“([^”]{2,120})”", text or ""):
            phrase = (a or b or c).strip()
            if phrase:
                found.append(phrase)
        return found

    def _extract_album_candidates(self, artist: str, results: List[WebResult]) -> List[Dict[str, Any]]:
        """
        Deterministically extract possible album titles from results.
        Returns list of candidates with scores and evidence urls.
        """
        artist_norm = self._norm(artist)
        candidates: Dict[str, Dict[str, Any]] = {}

        def add_candidate(title: str, r: WebResult, bonus: float = 0.0) -> None:
            key = self._norm(title)
            if not key or len(key) < 2:
                return
            # Filter obvious non-album words
            bad = ["wrapped", "podcast", "docuseries", "tour", "episode", "trailer"]
            if any(b in key for b in bad):
                return

            domain = self._domain(r.url)
            trust = self._trust_score(domain)
            entry = candidates.get(key)
            if not entry:
                entry = {
                    "value": title.strip(),
                    "score": 0.0,
                    "sources": [],
                    "domains": set(),
                    "snippets": [],
                }
                candidates[key] = entry

            if r.url and r.url not in entry["sources"]:
                entry["sources"].append(r.url)
            entry["domains"].add(domain)
            entry["score"] += trust + bonus
            if r.title:
                entry["snippets"].append(r.title[:180])

        for r in results:
            title = r.title or ""
            body = r.body or ""
            domain = self._domain(r.url)
            tnorm = self._norm(title)
            bnorm = self._norm(body)

            # Spotify album page pattern: "<Album> - Album by <Artist> | Spotify"
            if "spotify" in domain:
                m = re.search(r"^(?P<album>.+?)\s+-\s+(album|ep|single)\s+by\s+(?P<artist>.+?)(\s+\|\s+spotify)?$", title, flags=re.IGNORECASE)
                if m:
                    a = self._norm(m.group("artist"))
                    if artist_norm and artist_norm in a:
                        add_candidate(m.group("album"), r, bonus=0.25)
                        continue

            # Apple Music pattern: "<Album> by <Artist> on Apple Music"
            if "apple.com" in domain or "music.apple.com" in domain:
                m = re.search(r"^(?P<album>.+?)\s+by\s+(?P<artist>.+?)\s+on\s+apple\s+music", title, flags=re.IGNORECASE)
                if m:
                    a = self._norm(m.group("artist"))
                    if artist_norm and artist_norm in a:
                        add_candidate(m.group("album"), r, bonus=0.25)
                        continue

            # News/articles: look for quoted album names when "album" is mentioned
            if "album" in tnorm or "album" in bnorm or "studio album" in tnorm or "studio album" in bnorm:
                for qp in self._extract_quoted_phrases(title):
                    add_candidate(qp, r, bonus=0.15)
                for qp in self._extract_quoted_phrases(body):
                    # Only accept body quotes if artist is referenced nearby
                    if artist_norm and artist_norm in bnorm:
                        add_candidate(qp, r, bonus=0.05)

        # Convert to list with normalized score + requirement checks
        out: List[Dict[str, Any]] = []
        for k, v in candidates.items():
            domains = v["domains"]
            v["domains"] = sorted(list(domains))
            # Confidence normalization: scale by number of distinct domains
            distinct = len(domains)
            v["confidence"] = min(1.0, v["score"] / (1.2 + 0.2 * max(0, distinct - 1)))
            out.append(v)

        out.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        return out

    def _extract_song_performer_candidates(self, song_title: str, results: List[WebResult]) -> List[Dict[str, Any]]:
        song_norm = self._norm(song_title)
        candidates: Dict[str, Dict[str, Any]] = {}

        def add_candidate(artist: str, r: WebResult, bonus: float = 0.0) -> None:
            key = self._norm(artist)
            if not key or len(key) < 2:
                return
            domain = self._domain(r.url)
            trust = self._trust_score(domain)
            entry = candidates.get(key)
            if not entry:
                entry = {
                    "value": artist.strip(),
                    "score": 0.0,
                    "sources": [],
                    "domains": set(),
                    "snippets": [],
                }
                candidates[key] = entry
            if r.url and r.url not in entry["sources"]:
                entry["sources"].append(r.url)
            entry["domains"].add(domain)
            entry["score"] += trust + bonus
            if r.title:
                entry["snippets"].append(r.title[:180])

        for r in results:
            t = r.title or ""
            b = r.body or ""
            tnorm = self._norm(t)
            bnorm = self._norm(b)
            if song_norm and song_norm not in tnorm and song_norm not in bnorm:
                continue

            # News-style: "<Artist> ... '<Song>' ..." or "'<Song>': <Artist> ..."
            if song_norm and song_norm in tnorm:
                # Pattern: "<Artist> <verb> ... '<Song>'"
                m = re.search(
                    r"^(?P<artist>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(releases|reveals|references|debuts|drops|shares|premieres|unveils|explains)\b",
                    t,
                    flags=re.IGNORECASE,
                )
                if m:
                    add_candidate(m.group("artist"), r, bonus=0.25)
                # Pattern: "'<Song>': <Artist> <verb> ..."
                m = re.search(
                    r":\s*(?P<artist>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(releases|reveals|references|debuts|drops|shares|explains)\b",
                    t,
                    flags=re.IGNORECASE,
                )
                if m:
                    add_candidate(m.group("artist"), r, bonus=0.22)

            # Patterns in titles
            # "<Song> - <Artist>"
            m = re.search(r"^(?P<song>.+?)\s+-\s+(?P<artist>.+?)$", t)
            if m and song_norm and song_norm in self._norm(m.group("song")):
                add_candidate(m.group("artist"), r, bonus=0.15)
                continue
            # "<Song> Lyrics - <Artist>"
            m = re.search(r"^(?P<song>.+?)\s+lyrics\s+-\s+(?P<artist>.+?)$", t, flags=re.IGNORECASE)
            if m and song_norm and song_norm in self._norm(m.group("song")):
                add_candidate(m.group("artist"), r, bonus=0.2)
                continue
            # "<Song> by <Artist>"
            m = re.search(r"^(?P<song>.+?)\s+by\s+(?P<artist>.+?)$", t, flags=re.IGNORECASE)
            if m and song_norm and song_norm in self._norm(m.group("song")):
                add_candidate(m.group("artist"), r, bonus=0.15)

            # Body: "by <Artist>" (guarded to avoid capturing Wikipedia/Songfacts/etc.)
            m2 = re.search(r"\bby\s+([A-Z][A-Za-z0-9&'. ]{2,60})\b", b)
            if m2:
                cand = m2.group(1).strip()
                if cand.lower() not in {"wikipedia", "songfacts", "genius", "allmusic"}:
                    add_candidate(cand, r, bonus=0.03)

        out: List[Dict[str, Any]] = []
        for k, v in candidates.items():
            domains = v["domains"]
            v["domains"] = sorted(list(domains))
            distinct = len(domains)
            v["confidence"] = min(1.0, v["score"] / (1.0 + 0.25 * max(0, distinct - 1)))
            out.append(v)
        out.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        return out

    def _extract_title_holder_candidates(self, question: str, results: List[WebResult]) -> List[Dict[str, Any]]:
        """
        Best-effort extraction for \"current champion/winner/role holder\" questions.
        We look for \"<Name> wins\" and \"<Name> is the current\" patterns in titles/bodies.
        """
        candidates: Dict[str, Dict[str, Any]] = {}

        def add_candidate(name: str, r: WebResult, bonus: float = 0.0) -> None:
            key = self._norm(name)
            if not key or len(key.split()) < 2:
                return
            domain = self._domain(r.url)
            trust = self._trust_score(domain)
            entry = candidates.get(key)
            if not entry:
                entry = {
                    "value": name.strip(),
                    "score": 0.0,
                    "sources": [],
                    "domains": set(),
                    "snippets": [],
                }
                candidates[key] = entry
            if r.url and r.url not in entry["sources"]:
                entry["sources"].append(r.url)
            entry["domains"].add(domain)
            entry["score"] += trust + bonus
            if r.title:
                entry["snippets"].append(r.title[:180])

        for r in results:
            t = r.title or ""
            b = r.body or ""
            # "<Name> wins ..."
            m = re.search(r"^(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+wins\b", t)
            if m:
                add_candidate(m.group("name"), r, bonus=0.2)
                continue
            # "... is the current ..."
            m = re.search(r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+is\s+the\s+current\b", t)
            if m:
                add_candidate(m.group("name"), r, bonus=0.2)
                continue
            # Body patterns
            m = re.search(r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(?:has\s+)?won\b", b)
            if m:
                add_candidate(m.group("name"), r, bonus=0.05)

        out: List[Dict[str, Any]] = []
        for k, v in candidates.items():
            domains = v["domains"]
            v["domains"] = sorted(list(domains))
            distinct = len(domains)
            v["confidence"] = min(1.0, v["score"] / (1.0 + 0.25 * max(0, distinct - 1)))
            out.append(v)
        out.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        return out
    
    def _synthesize_answer(self, question: str, search_results: str) -> str:
        """Synthesize a final answer from search results."""
        self._emit("websearch_synthesize", "Synthesizing answer from search results...")
        
        if "No search results found" in search_results:
            return "I couldn't find current information about this topic through web search."
        
        try:
            prompt = SYNTHESIS_PROMPT.format(
                question=question,
                search_results=search_results,
                current_date=self._current_date
            )
            
            result = self.client.complete_chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_WEBSEARCH},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2500,  # Increased for more detailed answers
            )
            
            answer = result.text.strip()
            self._emit("websearch_answer", f"Final answer: {answer[:500]}...")
            return answer
            
        except Exception as e:
            self._emit("websearch_synth_error", f"Synthesis failed: {str(e)[:100]}")
            # Return raw search results as fallback
            return f"Web search results (synthesis failed):\n\n{search_results[:2500]}"
    
    def _queries_for_hop(self, intent: str, entity: Optional[str], question: str, hop: int) -> List[str]:
        """
        Deterministic query families per intent + hop.
        We intentionally include platform/site queries to enable deterministic extraction.
        """
        y = self._current_year
        q = question.strip()
        e = (entity or "").strip()

        if intent == "latest_album_of_artist" and e:
            if hop == 1:
                return [
                    f"{e} latest album {y}",
                    f"{e} newest album {y}",
                    f"{e} discography latest album",
                    f"{q} {y}",
                ]
            if hop == 2:
                return [
                    f"site:wikipedia.org {e} discography",
                    f"site:discogs.com {e} album",
                    f"site:musicbrainz.org {e} release-group",
                    f"site:music.apple.com {e} album",
                    f"site:open.spotify.com/album {e}",
                ]
            return [
                f"\"{e}\" new album released {y}",
                f"\"{e}\" announces new album {y}",
                f"{e} studio album {y}",
                f"{e} album release date {y}",
            ]

        if intent == "who_sang_or_performed_song" and e:
            if hop == 1:
                return [
                    f"\"{e}\" who sang",
                    f"\"{e}\" performed by",
                    f"\"{e}\" lyrics",
                    f"{q}",
                ]
            if hop == 2:
                return [
                    f"site:genius.com \"{e}\"",
                    f"site:open.spotify.com \"{e}\"",
                    f"site:music.apple.com \"{e}\"",
                    f"site:allmusic.com \"{e}\"",
                ]
            return [
                f"site:bandcamp.com \"{e}\"",
                f"site:musicbrainz.org \"{e}\"",
                f"\"{e}\" credits",
                f"\"{e}\" performed by artist",
            ]

        if intent == "current_title_holder":
            if hop == 1:
                return [f"{q} {y}", f"{q} latest news", f"{q} {self._current_date}"]
            if hop == 2:
                return [f"{q} reigning champion {y}", f"{q} winner {y}", f"{q} title holder {y}"]
            return [f"{q} official site", f"{q} standings {y}", f"{q} results {y}"]

        # general_fact_lookup
        # For general/historical questions, do NOT force current-year in hop 1.
        # Add year only in later hops if needed.
        if hop == 1:
            # Combine LLM-generated queries with a deterministic fallback (yearless)
            llm_qs = self._generate_search_queries(q, bias_current_year=False)
            det = [q, f"site:wikipedia.org {q}"]
            return list(dict.fromkeys(llm_qs + det))  # preserve order, unique
        if hop == 2:
            # If still missing, try adding the current year as a later-hop disambiguator
            return self._generate_fallback_queries(q, bias_current_year=True)
        return [f"{q} {y}", q]

    def _candidate_is_confident(self, cand: Dict[str, Any]) -> bool:
        """
        Conservative stop condition: either 2+ distinct domains OR one very trusted domain.
        """
        conf = float(cand.get("confidence", 0.0) or 0.0)
        domains = cand.get("domains") or []
        sources = cand.get("sources") or []
        if conf < 0.55:
            return False
        # 2+ distinct domains -> good
        if isinstance(domains, list) and len(domains) >= 2:
            return True
        # 1 very trusted domain (spotify/apple/etc)
        if isinstance(domains, list) and len(domains) == 1:
            if self._trust_score(domains[0]) >= 0.88 and len(sources) >= 1:
                return True
        return False

    def _format_extracted_answer(self, pack: WebEvidencePack) -> str:
        src_lines = "\n".join([f"- {u}" for u in pack.extracted_sources[:8]]) if pack.extracted_sources else "- (no source URLs)"
        entity = pack.entity or "the subject"

        if pack.intent == "latest_album_of_artist":
            return (
                f"**Direct Answer**: The latest album by **{entity}** is **{pack.extracted_answer}**.\n\n"
                f"**Key Details**:\n"
                f"- **Confidence**: {pack.extracted_confidence:.2f}\n"
                f"- **Evidence**: extracted from trusted discography/coverage sources\n\n"
                f"**Sources**:\n{src_lines}"
            )
        if pack.intent == "who_sang_or_performed_song":
            return (
                f"**Direct Answer**: The performer/artist for **\"{entity}\"** is **{pack.extracted_answer}**.\n\n"
                f"**Key Details**:\n"
                f"- **Confidence**: {pack.extracted_confidence:.2f}\n\n"
                f"**Sources**:\n{src_lines}"
            )
        if pack.intent == "current_title_holder":
            return (
                f"**Direct Answer**: Based on the most recent web evidence, the answer is **{pack.extracted_answer}**.\n\n"
                f"**Key Details**:\n"
                f"- **Confidence**: {pack.extracted_confidence:.2f}\n\n"
                f"**Sources**:\n{src_lines}"
            )
        return pack.extracted_answer or "No current information found."

    def run(self, question: str) -> str:
        """
        Run the web search agent to answer a question.

        For lookup-style questions (albums / who-performed / title-holder), we try to produce a
        deterministic extracted answer from evidence. If extraction fails, we DO NOT guess.
        """
        self._emit("websearch_start", f"🌐 WebSearchAgent activated for: {question}")
        pack = self.build_evidence(question)

        if pack.intent in {"latest_album_of_artist", "who_sang_or_performed_song", "current_title_holder"}:
            if pack.extracted_answer:
                ans = self._format_extracted_answer(pack)
                self._emit("websearch_complete", f"WebSearchAgent finished (deterministic). Answer length: {len(ans)} chars")
                return ans
            # No deterministic extraction -> do not hallucinate
            msg = (
                "**Direct Answer**: I could not reliably extract the requested fact from web evidence.\n\n"
                "**What I tried**:\n"
                f"- **Intent**: {pack.intent}\n"
                f"- **Queries tried**: {len(pack.queries)}\n\n"
                "**Next step**: provide more context (e.g., exact release year / platform) or allow more time for deeper search."
            )
            self._emit("websearch_complete", f"WebSearchAgent finished (insufficient evidence).")
            return msg

        # General questions: allow LLM synthesis from evidence text
        answer = self._synthesize_answer(question, pack.combined_results)
        self._emit("websearch_complete", f"WebSearchAgent finished. Answer length: {len(answer)} chars")
        return answer

    def build_evidence(self, question: str) -> WebEvidencePack:
        """
        Build a web evidence pack for downstream workers/synthesis.

        Core policy:
        - For lookup intents, we attempt deterministic extraction with multi-hop queries.
        - If extraction fails, we return evidence but DO NOT fabricate an answer.
        """
        intent = self._detect_intent(question)
        entity: Optional[str] = None
        if intent == "latest_album_of_artist":
            entity = self._extract_artist_from_album_question(question)
        elif intent == "who_sang_or_performed_song":
            entity = self._extract_song_title(question)

        self._emit("web_evidence_start", f"Collecting web evidence (intent={intent}, entity={entity})")

        tried_queries: List[str] = []
        all_results: List[WebResult] = []
        blocks: List[str] = []

        extracted_answer: Optional[str] = None
        extracted_confidence: float = 0.0
        extracted_sources: List[str] = []
        candidates: List[Dict[str, Any]] = []

        max_hops = 3
        for hop in range(1, max_hops + 1):
            hop_queries = self._queries_for_hop(intent, entity, question, hop)
            # De-dupe and cap to max_searches for this hop
            hop_queries = [q for q in hop_queries if q and q.lower() not in {t.lower() for t in tried_queries}]
            hop_queries = hop_queries[: self.max_searches]
            if not hop_queries:
                continue

            tried_queries.extend(hop_queries)
            hop_text, hop_results = self._execute_searches(hop_queries)
            blocks.append(hop_text)

            # Merge results by URL
            seen_urls = {r.url for r in all_results if r.url}
            for r in hop_results:
                if r.url and r.url in seen_urls:
                    continue
                all_results.append(r)
                if r.url:
                    seen_urls.add(r.url)

            # Deterministic extraction
            if intent == "latest_album_of_artist" and entity:
                candidates = self._extract_album_candidates(entity, all_results)
            elif intent == "who_sang_or_performed_song" and entity:
                candidates = self._extract_song_performer_candidates(entity, all_results)
            elif intent == "current_title_holder":
                candidates = self._extract_title_holder_candidates(question, all_results)
            else:
                candidates = []

            if candidates:
                top = candidates[0]
                self._emit(
                    "web_evidence_candidates",
                    f"Top candidate: {top.get('value')} (confidence={top.get('confidence'):.2f}, domains={top.get('domains')})",
                )
                if self._candidate_is_confident(top):
                    extracted_answer = str(top.get("value"))
                    extracted_confidence = float(top.get("confidence", 0.0) or 0.0)
                    extracted_sources = list(top.get("sources") or [])[:8]
                    self._emit("web_evidence_extracted", f"Extracted answer: {extracted_answer} (confidence={extracted_confidence:.2f})")
                    break
            else:
                # Missing-answer detection (do not stop early just because results exist)
                if intent in {"latest_album_of_artist", "who_sang_or_performed_song", "current_title_holder"}:
                    self._emit("web_evidence_missing", f"No extractable answer yet (hop {hop}/{max_hops}); continuing search...")

        combined_results = ("\n\n".join(blocks)).strip() if blocks else "No search results found."

        # Fetch deeper in-page evidence for top URLs (not just DDG snippets)
        fetched_pages, fetched_combined = self._fetch_pages_for_question(
            question,
            all_results,
            max_pages=4,
            max_chars_per_page=4500,
        )
        if fetched_combined:
            combined_results = (combined_results + "\n\n" + "=== Fetched Page Evidence ===\n" + fetched_combined).strip()

        pack = WebEvidencePack(
            question=question,
            intent=intent,
            entity=entity,
            queries=tried_queries,
            results=all_results,
            combined_results=combined_results,
            extracted_answer=extracted_answer,
            extracted_confidence=extracted_confidence,
            extracted_sources=extracted_sources,
            candidates=candidates[:10] if candidates else [],
            fetched_pages=fetched_pages,
            fetched_combined=fetched_combined,
        )

        self._emit(
            "web_evidence_complete",
            f"Web evidence ready: intent={intent}, queries={len(tried_queries)}, results={len(all_results)}, extracted={bool(extracted_answer)}",
        )
        return pack
    
    def quick_search(self, query: str) -> str:
        """
        Perform a quick single search without LLM synthesis.
        Useful for simple factual lookups.
        
        Args:
            query: Direct search query
            
        Returns:
            Raw search results
        """
        self._emit("websearch_quick", f"Quick search: {query[:80]}...")
        results = search_web(query, max_results=self.results_per_search)
        self._emit("websearch_quick_results", f"Results: {results[:500]}...")
        return results


def create_websearch_agent(
    client: OpenRouterClient,
    model_name: str = "mistralai/mistral-large-2512",
) -> WebSearchAgent:
    """Factory function to create a WebSearchAgent with default settings."""
    return WebSearchAgent(
        client=client,
        model_name=model_name,
        max_searches=3,
        results_per_search=5,
    )
