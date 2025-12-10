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

    def synthesize(self, problem: str, results: List[Tuple[SubTask, str]]) -> str:
        self._emit_thinking("synthesize_start", f"Synthesizing from {len(results)} worker results...")
        
        formatted = []
        for st, res in results:
            formatted.append(f"[{st.role}] {st.instruction}\nResult: {res}")
        context = "\n\n".join(formatted)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Supervisor synthesizing a final answer from worker outputs.\n"
                    "OUTPUT POLICY:\n"
                    "1) If the question requires a single numeric answer (e.g., 'How many', 'What is', 'Compute'),\n"
                    "   respond with EXACTLY ONE line containing ONLY the number.\n"
                    "2) If multiple quantities are requested, return a COMPACT JSON object.\n"
                    "3) If yes/no, reply 'yes' or 'no' only. For boolean expressions, 'T' or 'F' only.\n"
                    "4) For factual/knowledge questions, provide a clear, complete answer.\n"
                    "5) For complex questions, provide a well-structured response.\n"
                    "ALWAYS provide an answer. Never return empty."
                ),
            },
            {
                "role": "user",
                "content": f"Problem:\n{problem}\n\nWorker outputs:\n{context}\n\nFinal answer:",
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
            # Try to extract answer from worker results
            for st, res in results:
                if res.strip():
                    text = res.strip()
                    break
            if not text:
                text = "Unable to synthesize an answer from the available information."
        
        self._emit_thinking("synthesize_complete", f"Final answer: {text[:200]}...")
        return text

    def resynthesize_with_critique(self, problem: str, results: List[Tuple[SubTask, str]], critique: str) -> str:
        """
        Second-pass synthesis that explicitly incorporates the critique to fix issues.
        Enforces strict output policy and format compliance.
        """
        formatted = []
        for st, res in results:
            formatted.append(f"[{st.role}] {st.instruction}\nResult: {res}")
        context = "\n\n".join(formatted)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Supervisor making a corrected FINAL answer.\n"
                    "Incorporate the critique to repair any mistakes or missing elements.\n"
                    "STRICT OUTPUT POLICY:\n"
                    "1) If a single numeric answer is required, output ONLY the number (e.g., 18).\n"
                    "2) If multiple quantities are requested, output a compact JSON object with snake_case keys and numeric values only, no prose.\n"
                    "3) If yes/no, output 'yes' or 'no' only. For boolean expressions, output 'T' or 'F' only.\n"
                    "4) If the problem requests a specific format (e.g., coordinates like (r, θ)), output EXACTLY that format, with no extra text.\n"
                    "Do not include any explanations, steps, or additional lines."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem}\n\n"
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

    def critique(self, problem: str, results: List[Tuple[SubTask, str]]) -> str:
        """
        Runtime QA: ask the model to check logical consistency.
        Returns a brief note; empty string if all good.
        """
        self._emit_thinking("critique_start", "Evaluating worker outputs for consistency...")
        
        formatted = []
        for st, res in results:
            formatted.append(f"[{st.role}] {st.instruction}\nResult: {res}")
        context = "\n\n".join(formatted)
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
                "content": f"Problem:\n{problem}\n\nWorker outputs:\n{context}\n\nCritique:",
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

