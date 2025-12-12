from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

from ..infra.openrouter.client import OpenRouterClient, OpenRouterConfig


class SubTask(BaseModel):
    id: Optional[str] = None
    role: Literal["math", "qa", "logic", "research"]
    instruction: str
    depends_on: List[str] = []


class Plan(BaseModel):
    subtasks: List[SubTask]


SYSTEM_SUPERVISOR = (
    "You are the Supervisor. Decompose the user problem into strict, minimal subtasks.\n"
    "Available roles: {math, qa, logic, research}\n"
    "- math: arithmetic, algebra, numerical computation, mathematical proofs\n"
    "- qa: factual knowledge, humanities, history, literature, science facts, definitions\n"
    "- logic: reasoning puzzles, deduction, multi-step logical analysis, boolean logic\n"
    "- research: complex math/logic tasks that benefit from simulation, enumeration, or code-backed sanity checks\n\n"
    "Guidelines:\n"
    "- For simple arithmetic (e.g., 'What is 12*13?'), use a single math subtask.\n"
    "- For factual questions (e.g., 'Who wrote Hamlet?'), use a single qa subtask.\n"
    "- For complex problems, break into multiple subtasks with appropriate roles.\n"
    "- Use a research subtask when the task involves simulations, enumerations, subgroup counts, eigenvalue-set enumeration, toggling processes, or knot/quandle counting.\n"
    "- ALWAYS provide a non-empty instruction for each subtask.\n"
    "Return ONLY a JSON list with entries having fields: role, instruction."
)


def _format_supervisor_messages(problem: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_SUPERVISOR},
        {
            "role": "user",
            "content": (
                "Decompose the following problem into minimal subtasks. "
                "Respond with JSON only.\n\nProblem:\n" + problem
            ),
        },
    ]


def _parse_plan(text: str) -> Plan:
    import json
    try:
        data = json.loads(text)
        subtasks = [SubTask(**item) for item in data]
        return Plan(subtasks=subtasks)
    except Exception:
        # Fallback: naive heuristic (single generic subtask)
        return Plan(subtasks=[SubTask(role="logic", instruction=text.strip())])


def _needs_research(problem: str) -> bool:
    p = problem.lower()
    research_keys = [
        "eigenvalue", "root of unity", "unit circle", "subgroup", "index",
        "free product", "hotel", "guest", "light", "toggle", "knot", "quandle",
        "rank-1", "rank 1", "matrix", "matrices", "representation"
    ]
    return any(k in p for k in research_keys)


def _format_results_for_synthesis(results: List[Tuple[Any, Any]]) -> str:
    """
    Format results for synthesis, handling both legacy string format and swarm
    List[Tuple[str, str]] format.

    IMPORTANT: Swarm results may contain an early (wrong) answer followed by later
    reconciliation/judgment. We therefore select the BEST response using a heuristic,
    not simply the first response.
    """

    def _extract_final_section(text: str) -> str:
        """
        If the response contains an explicit final verdict, prefer that shorter section.
        This avoids synthesis latching onto earlier 'divergence' claims.
        """
        t = (text or "").strip()
        if not t:
            return t
        lower = t.lower()
        markers = [
            "final answer:",
            "final verdict:",
            "best answer:",
            "final judgment",
            "conclusion:",
            "derived best answer",
        ]
        idx = max((lower.rfind(m) for m in markers), default=-1)
        # Only extract if it looks like a reconciliation-style response
        if idx >= 0 and any(k in lower for k in ["divergence", "reconciliation", "sanity check", "judicial"]):
            extracted = t[idx:].strip()
            # If it's still huge, keep last ~800 chars
            if len(extracted) > 1200:
                extracted = extracted[-1200:]
            return extracted
        return t

    def _score_candidate(model_name: str, response: str) -> float:
        r = (response or "").strip()
        if not r:
            return -1e9
        if r.startswith("["):
            return -1e6
        score = 0.0
        # Prefer longer, more detailed reconciliations
        score += min(len(r) / 500.0, 5.0)
        low = r.lower()
        if any(k in low for k in ["reconciliation", "sanity check", "divergence", "judicial", "verifiable", "cross-reference"]):
            score += 2.0
        if any(k in low for k in ["final answer", "final verdict", "best answer"]):
            score += 1.5
        # Prefer stronger models when all else equal
        mn = (model_name or "").lower()
        if "large" in mn:
            score += 0.5
        elif "medium" in mn:
            score += 0.25
        return score

    formatted: List[str] = []
    for st, res in results:
        if isinstance(res, list):
            candidates: List[Tuple[str, str]] = [(m, r) for m, r in res if (r or "").strip()]
            if not candidates:
                formatted.append(f"[{st.role}] {st.instruction}\nResult: [No response]")
                continue

            best_model, best_resp = max(candidates, key=lambda mr: _score_candidate(mr[0], mr[1]))
            best_resp = _extract_final_section(best_resp)
            formatted.append(
                f"[{st.role}] {st.instruction}\n"
                f"(chosen_from={best_model})\n"
                f"Result: {best_resp}"
            )
        else:
            text = _extract_final_section(str(res))
            formatted.append(f"[{st.role}] {st.instruction}\nResult: {text}")

    return "\n\n".join(formatted)


def _detect_question_type(problem: str) -> str:
    """
    Detect the expected answer format based on question phrasing.
    
    Returns one of:
    - 'numeric': single number answer (e.g., "How many...", "Compute...")
    - 'boolean': yes/no or true/false (e.g., "Is X true?", "Does...")
    - 'multi_quantity': multiple values, JSON format (e.g., "List the values of X, Y, Z")
    - 'explanatory': narrative prose (e.g., "Explain...", "Describe...", "Discuss...")
    - 'factual': concise factual answer (e.g., "Who wrote...", "When did...")
    """
    p = problem.strip().lower()
    
    # Explanatory questions - want narrative prose (check first, higher priority)
    explanatory_cues = [
        "explain", "describe", "discuss", "analyze", "compare",
        "contrast", "elaborate", "significance", "importance",
        "how does", "why did", "what caused", "what are the effects",
        "outline", "summarize", "evaluate", "assess", "interpret",
        "what is the significance", "what is the importance",
        "tell me about", "write about", "essay", "in detail",
        "comprehensive", "thorough", "elaborate on"
    ]
    if any(cue in p for cue in explanatory_cues):
        return "explanatory"
    
    # Numeric questions - want single number
    numeric_cues = [
        "how many", "compute", "calculate", "what is the value",
        "find the number", "count the", "total number", "sum of",
        "product of", "result of", "evaluate"
    ]
    if any(k in p for k in numeric_cues):
        return "numeric"
    
    # Boolean questions - want yes/no or true/false
    boolean_starts = ("is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ", "could ", "will ", "would ", "has ", "have ", "had ")
    if p.startswith(boolean_starts):
        # Check it's not an explanatory follow-up
        if not any(cue in p for cue in ["explain", "describe", "why"]):
            return "boolean"
    
    # Multi-quantity (lists) - want JSON or structured list
    multi_cues = ["list ", "what are the", "name the", "give me all", "enumerate"]
    if any(k in p for k in multi_cues):
        return "multi_quantity"
    
    # Default to factual for simple knowledge questions
    return "factual"


def _needs_current_info(problem: str) -> bool:
    """
    Detect if a question likely requires real-time/current information.
    
    BROADLY triggers for:
    - Questions about whether someone is alive/dead
    - Current status, recent events, news
    - Today/now/current temporal markers
    - Specific recent dates (2024, 2025, etc.)
    - Questions about latest/newest versions, releases, updates
    - Sports champions, award winners, title holders
    - "Who is the current X" questions (any X)
    - General factual questions that may have changed recently
    
    Returns True if the question likely needs web search for current info.
    """
    import re as _re
    p = problem.strip().lower()
    
    # Temporal markers indicating recency
    temporal_cues = [
        "is alive", "still alive", "is dead", "has died", "was killed",
        "was shot", "was assassinated", "was murdered", "passed away",
        "currently", "right now", "today", "this week", "this month",
        "this year", "recent", "latest", "2024", "2025", "2026",
        "what happened to", "current status", "news about",
        "is he alive", "is she alive", "are they alive",
        "did he die", "did she die", "when did he die", "when did she die",
        "breaking news", "just happened", "recently",
        # Additional patterns for version/release questions
        "latest version", "newest version", "most recent version",
        "current version", "latest release", "newest release",
        "latest model", "newest model", "most recent",
        "latest update", "what's new", "what is new",
        "just released", "just launched", "just announced",
        "released today", "announced today", "launched today",
    ]
    
    # Check for temporal cues
    if any(cue in p for cue in temporal_cues):
        return True
    
    # Pattern for "when did [someone] die" questions
    death_pattern = _re.search(r"when did\s+\w+.*\s*(die|pass away|get killed|was killed)", p)
    if death_pattern:
        return True
    
    # Pattern for death questions in general
    death_question = _re.search(r"(did|has|is|was)\s+\w+.*\s+(die|died|dead|killed|assassinated|murdered)", p)
    if death_question:
        return True
    
    # Pattern for "what/which is the latest/newest X" questions
    latest_pattern = _re.search(r"(what|which|what's|which is).*(latest|newest|most recent|current)", p)
    if latest_pattern:
        return True
    
    # Pattern for "latest X" at end of question
    latest_end_pattern = _re.search(r"latest\s+\w+\s*\??$", p)
    if latest_end_pattern:
        return True
    
    # Questions about current state of people (alive/dead patterns)
    alive_pattern = _re.search(r"(who is|is|are|was|were)\s+\w+.*\s+(alive|dead|living|deceased)", p)
    if alive_pattern:
        return True
    
    # ===== EXPANDED: Sports, Awards, Titles =====
    # Sports champions and title holders
    sports_titles = (
        "champion|champions|winner|winners|holder|holders|"
        "mvp|ballon d'or|best player|top scorer|"
        "world cup|super bowl|world series|"
        "olympic|olympics|gold medal|"
        "grand slam|wimbledon|us open|french open|australian open|"
        "formula 1|f1|formula one|nba|nfl|nhl|mlb|"
        "premier league|la liga|bundesliga|serie a|"
        "champions league|europa league|"
        "title|trophy|cup winner"
    )
    sports_pattern = _re.search(rf"(who is|who are|who won|who has|current|reigning)\s+.*({sports_titles})", p)
    if sports_pattern:
        return True
    
    # "Who won the X" questions
    who_won_pattern = _re.search(r"who (won|wins|has won|will win)\s+(the\s+)?", p)
    if who_won_pattern:
        return True
    
    # "current [anything] champion/winner/holder"
    current_title_pattern = _re.search(r"current\s+\w+\s*(champion|winner|holder|title|leader)", p)
    if current_title_pattern:
        return True
    
    # ===== EXPANDED: "Who is the current X" for ANY X =====
    # This catches questions like "Who is the current F1 champion?"
    who_current_pattern = _re.search(r"who is the current\s+\w+", p)
    if who_current_pattern:
        return True
    
    # "Who is the X champion/winner/leader"
    who_is_title_pattern = _re.search(r"who is the\s+\w+\s*(champion|winner|holder|leader|best)", p)
    if who_is_title_pattern:
        return True
    
    # ===== Political and organizational roles =====
    political_roles = (
        "president|vice president|prime minister|pm|chief minister|cm|"
        "governor|mayor|minister|secretary|chancellor|"
        "ceo|cto|cfo|coo|chairman|chairwoman|chairperson|"
        "leader|head|director|commissioner|chief|manager|"
        "king|queen|prince|princess|emperor|pope"
    )
    current_role_pattern = _re.search(
        rf"(who is the current|who is the|who is|who are the|who are)\s+.*\s*({political_roles})", 
        p
    )
    if current_role_pattern:
        return True
    
    # Pattern for "current X of Y" questions
    current_of_pattern = _re.search(rf"current\s+({political_roles})\s+(of|for|in)", p)
    if current_of_pattern:
        return True
    
    # Pattern for abbreviated role questions ("current PM/CM of...")
    abbrev_pattern = _re.search(r"(current|who is the)\s+(pm|cm|ceo|cto)\s+(of|for|in)", p)
    if abbrev_pattern:
        return True
    
    # Questions about software/product versions
    version_pattern = _re.search(r"(what|which).*(version|release|update).*(of|for)\s+\w+", p)
    if version_pattern:
        return True
    
    # ===== EXPANDED: General "Who is/was" questions about potentially changing facts =====
    # These often need current info even if not explicitly asking for "current"
    who_is_pattern = _re.search(r"^who (is|was|are|were) (the )?\w+", p)
    if who_is_pattern:
        # Exclude static historical questions
        static_excludes = ["inventor", "founder", "creator", "author", "writer", "discoverer", "born", "birthday"]
        if not any(excl in p for excl in static_excludes):
            return True
    
    return False


class SupervisorAgent:
    def __init__(
        self,
        client: OpenRouterClient,
        model_name: Optional[str] = None,
        fallback_model: Optional[str] = None,
        secondary_fallback_model: Optional[str] = None,
    ):
        self.client = client
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.secondary_fallback_model = secondary_fallback_model
        self._thinking_callback: Optional[Callable[[str, str], None]] = None

    def set_thinking_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set callback for intermediate thinking: callback(stage, content)"""
        self._thinking_callback = callback

    def _emit_thinking(self, stage: str, content: str) -> None:
        if self._thinking_callback:
            self._thinking_callback(stage, content)

    def decompose(self, problem: str) -> Plan:
        self._emit_thinking("decompose_start", f"Analyzing problem: {problem[:200]}...")
        
        messages = _format_supervisor_messages(problem)
        self._emit_thinking("decompose_model", f"Querying model: {self.model_name or 'default'}")
        result = self.client.complete_chat(messages=messages, model=self.model_name)
        text = result.text.strip()
        
        if not text and self.fallback_model:
            self._emit_thinking("decompose_fallback", f"Primary empty, trying: {self.fallback_model}")
            result = self.client.complete_chat(messages=messages, model=self.fallback_model)
            text = result.text.strip()
        if not text and self.secondary_fallback_model:
            self._emit_thinking("decompose_secondary", f"Fallback empty, trying: {self.secondary_fallback_model}")
            result = self.client.complete_chat(messages=messages, model=self.secondary_fallback_model)
            text = result.text.strip()
            
        plan = _parse_plan(text)
        
        def _is_numeric(q: str) -> bool:
            p = q.strip().lower()
            return any(k in p for k in ("how many", "what is", "compute", "evaluate", "result", "value of"))
        
        def _is_factual(q: str) -> bool:
            p = q.strip().lower()
            return any(k in p for k in ("who ", "when ", "where ", "what year", "which ", "name the", "define ", "explain "))
        
        # If the question is numeric but no math subtask was produced, insert one to force numeric reasoning.
        if _is_numeric(problem) and not any(st.role == "math" for st in plan.subtasks):
            self._emit_thinking("decompose_math_injection", "Numeric query detected; adding math subtask for precise counting.")
            plan.subtasks = [SubTask(role="math", instruction=problem.strip())] + list(plan.subtasks)

        # Inject research subtask for complex math/logic that benefit from simulation
        if _needs_research(problem) and not any(st.role == "research" for st in plan.subtasks):
            self._emit_thinking("decompose_research_injection", "Complex pattern detected; adding research subtask for simulation/code.")
            plan.subtasks = [SubTask(role="research", instruction=problem.strip())] + list(plan.subtasks)
        
        # Guardrail: if decomposition failed (empty instruction), create a minimal fallback plan
        if (not plan.subtasks) or all(not (st.instruction or "").strip() for st in plan.subtasks):
            if _is_numeric(problem):
                role = "math"
            elif _is_factual(problem):
                role = "qa"
            else:
                role = "logic"
            self._emit_thinking("decompose_fallback_plan", f"Using fallback plan with role: {role}")
            return Plan(subtasks=[SubTask(role=role, instruction=problem.strip())])
        
        subtask_summary = ", ".join([f"{st.role}: {st.instruction[:50]}..." for st in plan.subtasks])
        self._emit_thinking("decompose_complete", f"Plan created with {len(plan.subtasks)} subtasks: {subtask_summary}")
        return plan

    def synthesize(self, problem: str, results: List[Tuple[SubTask, Any]], web_evidence: Optional[str] = None) -> str:
        self._emit_thinking("synthesize_start", f"Synthesizing from {len(results)} worker results...")
        
        # Detect question type and use appropriate synthesis prompt
        question_type = _detect_question_type(problem)
        self._emit_thinking("synthesize_question_type", f"Detected question type: {question_type}")
        
        # Use helper to format results (handles both string and swarm format)
        context = _format_results_for_synthesis(results)
        
        # Build format-aware system prompt
        if question_type == "explanatory":
            system_content = (
                "You are the Supervisor synthesizing a final answer from worker outputs.\n"
                "OUTPUT POLICY FOR EXPLANATORY QUESTIONS:\n"
                "- Provide a well-structured, comprehensive narrative answer.\n"
                "- Use clear paragraphs with logical flow.\n"
                "- Include headings (using ###) where appropriate for organization.\n"
                "- Preserve the depth, detail, and nuance from worker outputs.\n"
                "- Do NOT convert to JSON, bullet points, or abbreviated formats.\n"
                "- Write in an educational, engaging, and thorough style.\n"
                "- The answer should read like a well-written essay or explanation.\n"
                "ALWAYS provide a substantive answer. Never return empty."
            )
        elif question_type == "numeric":
            system_content = (
                "You are the Supervisor synthesizing a final answer from worker outputs.\n"
                "OUTPUT POLICY FOR NUMERIC QUESTIONS:\n"
                "- Respond with EXACTLY ONE line containing ONLY the number.\n"
                "- No explanations, units, or additional text.\n"
                "- Example: 42"
            )
        elif question_type == "boolean":
            system_content = (
                "You are the Supervisor synthesizing a final answer from worker outputs.\n"
                "OUTPUT POLICY FOR YES/NO QUESTIONS:\n"
                "- Reply with 'yes' or 'no' only.\n"
                "- For boolean/true-false expressions, reply 'T' or 'F' only.\n"
                "- No explanations or additional text."
            )
        elif question_type == "multi_quantity":
            system_content = (
                "You are the Supervisor synthesizing a final answer from worker outputs.\n"
                "OUTPUT POLICY FOR MULTI-VALUE QUESTIONS:\n"
                "- Return a COMPACT JSON object with snake_case keys.\n"
                "- Include only the requested values.\n"
                "- No prose or explanations."
            )
        else:  # factual
            system_content = (
                "You are the Supervisor synthesizing a final answer from worker outputs.\n"
                "OUTPUT POLICY FOR FACTUAL QUESTIONS:\n"
                "- Provide a clear, complete, and accurate answer.\n"
                "- Be concise but thorough.\n"
                "- Include relevant context if it helps understanding.\n"
                "- GROUNDEDNESS: Do NOT introduce any named-entity claim unless it appears in the Worker outputs OR in the Web Evidence.\n"
                "- If Worker outputs conflict, prefer the most evidence-backed/verified claim.\n"
                "ALWAYS provide an answer. Never return empty."
            )
        
        web_section = ""
        if web_evidence and web_evidence.strip():
            web_section = f"Web Evidence (search results):\n{web_evidence.strip()}\n\n"

        messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": f"Problem:\n{problem}\n\n{web_section}Worker outputs:\n{context}\n\nFinal answer:",
            },
        ]
        
        self._emit_thinking("synthesize_model", f"Querying model: {self.model_name or 'default'}")
        result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.model_name)
        text = result.text.strip()
        
        if not text and self.fallback_model:
            self._emit_thinking("synthesize_fallback", f"Primary empty, trying: {self.fallback_model}")
            result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.fallback_model)
            text = result.text.strip()
        if not text and self.secondary_fallback_model:
            self._emit_thinking("synthesize_secondary", f"Fallback empty, trying: {self.secondary_fallback_model}")
            result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.secondary_fallback_model)
            text = result.text.strip()
        
        # Emergency fallback
        if not text:
            self._emit_thinking("synthesize_emergency", "All models empty, extracting from worker results")
            # Try to extract answer from worker results (handle both formats)
            for st, res in results:
                if isinstance(res, list):
                    # Swarm format: extract first valid response
                    for model_name, response in res:
                        if response and response.strip() and not response.startswith("["):
                            text = response.strip()
                            break
                elif isinstance(res, str) and res.strip():
                    text = res.strip()
                if text:
                    break
            if not text:
                text = "Unable to synthesize an answer from the available information."
        
        self._emit_thinking("synthesize_complete", f"Final answer: {text[:200]}...")
        return text

    def resynthesize_with_critique(
        self,
        problem: str,
        results: List[Tuple[SubTask, Any]],
        critique: str,
        web_evidence: Optional[str] = None,
    ) -> str:
        """
        Second-pass synthesis that explicitly incorporates the critique to fix issues.
        Uses format-aware prompts based on question type.
        """
        # Detect question type and use appropriate repair prompt
        question_type = _detect_question_type(problem)
        
        # Use helper to format results (handles both string and swarm format)
        context = _format_results_for_synthesis(results)
        
        # Build format-aware system prompt for repair
        if question_type == "explanatory":
            # For explanatory questions, only repair factual errors, preserve narrative format
            system_content = (
                "You are the Supervisor making a corrected FINAL answer.\n"
                "Fix any factual errors or gaps noted in the critique.\n"
                "IMPORTANT OUTPUT POLICY FOR EXPLANATORY QUESTIONS:\n"
                "- MAINTAIN the narrative/explanatory prose format.\n"
                "- Do NOT convert to JSON, bullet points, or abbreviated formats.\n"
                "- Preserve the essay-like structure with paragraphs and headings.\n"
                "- Keep the depth and detail - explanations should be thorough.\n"
                "- Write in an educational, engaging style.\n"
                "Your answer should read like a well-written explanation or essay."
            )
        elif question_type == "numeric":
            system_content = (
                "You are the Supervisor making a corrected FINAL answer.\n"
                "Incorporate the critique to fix any calculation errors.\n"
                "STRICT OUTPUT POLICY:\n"
                "- Output ONLY the number (e.g., 18).\n"
                "- No explanations, steps, or additional lines."
            )
        elif question_type == "boolean":
            system_content = (
                "You are the Supervisor making a corrected FINAL answer.\n"
                "Incorporate the critique to verify the answer.\n"
                "STRICT OUTPUT POLICY:\n"
                "- Output 'yes' or 'no' only.\n"
                "- For boolean expressions, output 'T' or 'F' only.\n"
                "- No explanations or additional text."
            )
        elif question_type == "multi_quantity":
            system_content = (
                "You are the Supervisor making a corrected FINAL answer.\n"
                "Incorporate the critique to fix any missing or incorrect values.\n"
                "STRICT OUTPUT POLICY:\n"
                "- Output a compact JSON object with snake_case keys.\n"
                "- Include only the requested values.\n"
                "- No prose or explanations."
            )
        else:  # factual
            system_content = (
                "You are the Supervisor making a corrected FINAL answer.\n"
                "Incorporate the critique to repair any mistakes or missing elements.\n"
                "OUTPUT POLICY:\n"
                "- Provide a clear, complete, and accurate answer.\n"
                "- Be concise but thorough.\n"
                "- IMPORTANT: Do NOT output JSON unless the user explicitly requested JSON.\n"
                "- Prefer natural prose with short paragraphs and, if helpful, a small bullet list."
            )
        
        web_section = ""
        if web_evidence and web_evidence.strip():
            web_section = f"Web Evidence (search results):\n{web_evidence.strip()}\n\n"

        messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem}\n\n"
                    f"{web_section}"
                    f"Worker outputs:\n{context}\n\n"
                    f"Critique (issues to fix):\n{critique}\n\n"
                    "Provide the corrected final answer now:"
                ),
            },
        ]
        result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.model_name)
        text = result.text.strip()
        if not text and self.fallback_model:
            result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.fallback_model)
            text = result.text.strip()
        if not text and self.secondary_fallback_model:
            result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.secondary_fallback_model)
            text = result.text.strip()
        return text

    def critique(self, problem: str, results: List[Tuple[SubTask, Any]], web_evidence: Optional[str] = None) -> str:
        """
        Runtime QA: ask the model to check logical consistency.
        Returns a brief note; empty string if all good.
        """
        self._emit_thinking("critique_start", "Evaluating worker outputs for consistency...")
        
        # Use helper to format results (handles both string and swarm format)
        context = _format_results_for_synthesis(results)
        web_section = ""
        if web_evidence and web_evidence.strip():
            web_section = f"Web Evidence (search results):\n{web_evidence.strip()}\n\n"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Supervisor Critic. Assess the worker outputs for logical consistency "
                    "and completeness. Reply with a short note. If everything looks correct, say 'OK'."
                ),
            },
            {
                "role": "user",
                "content": f"Problem:\n{problem}\n\n{web_section}Worker outputs:\n{context}\n\nCritique:",
            },
        ]
        
        result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.model_name)
        text = result.text.strip()
        if not text and self.fallback_model:
            result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.fallback_model)
            text = result.text.strip()
        if not text and self.secondary_fallback_model:
            result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.secondary_fallback_model)
            text = result.text.strip()
        
        if not text:
            text = "OK"  # Default to OK if critique fails
            
        self._emit_thinking("critique_complete", f"Critique: {text[:200]}...")
        return text

