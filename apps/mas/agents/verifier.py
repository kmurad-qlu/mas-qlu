from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..infra.openrouter.client import OpenRouterClient


SYSTEM_VERIFIER = (
    "You are a Math Verifier. Independently recompute the final numeric answer.\n"
    "- Re-derive from first principles; do not copy the candidate.\n"
    "- Check the derivation path for logical gaps.\n"
    "- Verify that the method used (e.g., Generating Functions, Recurrences) is applicable.\n"
    "- If confident the candidate is correct, return ONLY the final number.\n"
    "- If the candidate is wrong or missing, return ONLY the corrected number.\n"
    "- No words, no units, no extra text. Output a bare number."
)


class VerifierAgent:
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

    def verify_numeric(self, problem: str, candidate_answer: str, context: str = "") -> str:
        self._emit_thinking("verify_start", f"Verifying answer: {candidate_answer}")
        
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_VERIFIER},
            {
                "role": "user",
                "content": (
                    "Problem:\n"
                    f"{problem}\n\n"
                    "Candidate final answer (may be wrong or empty):\n"
                    f"{candidate_answer}\n\n"
                    "Relevant worker notes (optional):\n"
                    f"{context}\n\n"
                    "Return ONLY the final number:"
                ),
            },
        ]
        
        self._emit_thinking("verify_model", f"Querying verifier model: {self.model_name or 'default'}")
        result = self.client.complete_chat(messages=messages, temperature=0.1, model=self.model_name)
        text = result.text.strip()
        
        if not text and self.fallback_model:
            self._emit_thinking("verify_fallback", f"Primary empty, trying: {self.fallback_model}")
            result = self.client.complete_chat(messages=messages, temperature=0.1, model=self.fallback_model)
            text = result.text.strip()
        if not text and self.secondary_fallback_model:
            self._emit_thinking("verify_secondary", f"Fallback empty, trying: {self.secondary_fallback_model}")
            result = self.client.complete_chat(messages=messages, temperature=0.1, model=self.secondary_fallback_model)
            text = result.text.strip()
        
        if text:
            self._emit_thinking("verify_complete", f"Verified answer: {text}")
        else:
            self._emit_thinking("verify_empty", "Verifier returned empty response")
        
        return text

