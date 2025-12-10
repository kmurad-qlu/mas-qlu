"""
SwarmWorker: Concurrent multi-model worker system.

Dispatches the same task to multiple models in parallel and waits for
at least N valid responses before returning.
"""
from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from ..infra.openrouter.client import OpenRouterClient


# System prompts for different roles
SYSTEM_PROMPTS = {
    "math": (
        "You are MathWorker. Solve arithmetic, algebra, and mathematical problems precisely.\n"
        "- Think step by step internally.\n"
        "- For numeric answers, output EXACTLY one line in the format: '#### <final_numeric_answer>'\n"
        "- For non-numeric math answers (coordinates, expressions), provide the answer clearly.\n"
        "- ALWAYS provide an answer. Never return empty.\n"
        "\n"
        "Cognitive Strategies:\n"
        "1. Empirical Simulation (for Iterative Problems):\n"
        "   - Do not guess. For iterative processes, manually simulate the first N steps (e.g., N=1 to 10) to identify invariants, cycles, or patterns.\n"
        "   - Tabulate the state changes explicitly.\n"
        "2. First-Principles Derivation:\n"
        "   - Do not rely on memorized formulas which may be hallucinated. Derive the necessary results from definitions.\n"
        "3. Structural Mapping (for Algebra/Topology):\n"
        "   - Identify the underlying algebraic structure (Group, Ring, Field). Enumerate elements for small cases.\n"
        "   - Check for isomorphisms to known standard structures.\n"
        "4. Combinatorial Tools:\n"
        "   - For counting problems, consider Generating Functions, Recurrences, or the Exponential Formula."
    ),
    "qa": (
        "You are QAWorker. Answer factual and knowledge questions thoroughly.\n"
        "- For simple factual questions, provide a concise answer (1-2 sentences).\n"
        "- For complex questions (humanities, history, literature, philosophy, science), "
        "provide a well-reasoned explanation with key details.\n"
        "- If yes/no, answer 'yes' or 'no' followed by a brief justification.\n"
        "- ALWAYS provide a substantive answer. Never return empty or refuse to answer.\n"
        "- If uncertain, provide your best assessment with caveats."
    ),
    "logic": (
        "You are LogicWorker. Perform logical deduction and multi-step reasoning.\n"
        "- For simple logical problems, provide a brief conclusion with key reasoning.\n"
        "- For complex reasoning problems (puzzles, proofs, multi-step analysis), "
        "show your step-by-step logical process.\n"
        "- For boolean/true-false questions, state the answer clearly with justification.\n"
        "- ALWAYS provide a substantive answer. Never return empty or refuse to reason.\n"
        "- If the problem is ambiguous, state your assumptions and proceed.\n"
        "\n"
        "Cognitive Strategies:\n"
        "1. Empirical Simulation (for Iterative Problems):\n"
        "   - Do not guess. For iterative processes, manually simulate the first N steps (e.g., N=1 to 10) to identify invariants, cycles, or patterns.\n"
        "   - Tabulate the state changes explicitly.\n"
        "2. First-Principles Derivation:\n"
        "   - Do not rely on memorized formulas which may be hallucinated. Derive the necessary results from definitions.\n"
        "3. Structural Mapping (for Algebra/Topology):\n"
        "   - Identify the underlying algebraic structure (Group, Ring, Field). Enumerate elements for small cases.\n"
        "   - Check for isomorphisms to known standard structures.\n"
        "4. Combinatorial Tools:\n"
        "   - For counting problems, consider Generating Functions, Recurrences, or the Exponential Formula."
    ),
}


class SwarmWorkerManager:
    """
    Concurrent multi-model worker that dispatches tasks to all models in parallel.
    """
    
    def __init__(
        self,
        client: OpenRouterClient,
        models: List[str],
        per_model_timeout: float = 120.0,
        min_responses: int = 2,
        overall_timeout: float = 300.0,
        cooperative: bool = True,
        coop_min_agreement: int = 2,
        coop_reconcile_prompt: str = "Peers disagree; reconcile and provide the single best answer.",
    ):
        """
        Initialize the swarm worker manager.
        
        Args:
            client: OpenRouterClient instance for API calls
            models: List of model names to use in the swarm
            per_model_timeout: Timeout in seconds for each model
            min_responses: Minimum number of valid responses to wait for
            overall_timeout: Overall timeout for the swarm operation
        """
        self.client = client
        self.models = models
        self.per_model_timeout = per_model_timeout
        self.min_responses = min_responses
        self.overall_timeout = overall_timeout
        self.cooperative = cooperative
        self.coop_min_agreement = coop_min_agreement
        self.coop_reconcile_prompt = coop_reconcile_prompt
        # Aggressive reconciliation prompt
        self.coop_reconcile_prompt = (
            "Peers disagree. You must now act as a Judge.\n"
            "1. Read the previous responses carefully.\n"
            "2. Identify the specific step where they diverge.\n"
            "3. Perform a 'Sanity Check' on the conflicting claims using a small example (e.g., N=1, 2, or 3).\n"
            "4. Provide the single, rigorously derived best answer."
        )
        self._thinking_callback: Optional[Callable[[str, str], None]] = None
        self._lock = threading.Lock()
    
    def set_thinking_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set callback for intermediate thinking: callback(stage, content)"""
        self._thinking_callback = callback
    
    def _emit_thinking(self, stage: str, content: str) -> None:
        if self._thinking_callback:
            self._thinking_callback(stage, content)
    
    def _query_single_model(
        self,
        model: str,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, str, float]:
        """
        Query a single model and return (model_name, response, latency_ms).
        Returns empty response on error.
        """
        start = time.perf_counter()
        try:
            result = self.client.complete_chat(
                messages=messages,
                temperature=0.0,
                model=model,
            )
            latency = (time.perf_counter() - start) * 1000
            return (model, result.text.strip(), latency)
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self._emit_thinking("swarm_error", f"Model {model} failed: {str(e)[:100]}")
            return (model, "", latency)
    
    def run(
        self,
        instruction: str,
        role: str = "logic",
        context: str = "",
        overall_timeout: Optional[float] = None,
        peer_responses: Optional[List[Tuple[str, str]]] = None,
        round_idx: int = 1,
        reconcile: bool = False,
    ) -> List[Tuple[str, str]]:
        """
        Run the task on all models concurrently.
        
        Args:
            instruction: The task instruction
            role: One of 'math', 'qa', 'logic'
            context: Optional context
            overall_timeout: Overall timeout (default: self.overall_timeout)
            
        Returns:
            List of (model_name, response) tuples for all valid responses
        """
        if overall_timeout is None:
            overall_timeout = self.overall_timeout
        
        # Get the appropriate system prompt
        system_prompt = SYSTEM_PROMPTS.get(role, SYSTEM_PROMPTS["logic"])
        
        # Build messages
        user_content = f"Task: {instruction}"
        if context:
            user_content += f"\nContext: {context}"
        if peer_responses:
            snippets = []
            for m, r in peer_responses[:6]:
                if r and not r.startswith("["):
                    snippets.append(f"- {m}: {r}")
            if snippets:
                user_content += "\nPeer responses so far:\n" + "\n".join(snippets)
            if reconcile and self.cooperative:
                user_content += f"\nReconcile instruction: {self.coop_reconcile_prompt}"
        
        user_content += "\nProvide your answer:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        self._emit_thinking(
            "swarm_dispatch", 
            f"Dispatching [{role}] task to {len(self.models)} models (round {round_idx}): {', '.join(self.models)}"
        )
        
        # Track responses
        responses: List[Tuple[str, str]] = []
        valid_count = 0
        start_time = time.perf_counter()
        
        # Create executor WITHOUT context manager so we can force immediate shutdown
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.models))
        
        try:
            # Submit all tasks
            future_to_model = {
                executor.submit(
                    self._query_single_model, model, messages
                ): model
                for model in self.models
            }
            
            # Process results as they complete
            try:
                for future in concurrent.futures.as_completed(
                    future_to_model, 
                    timeout=overall_timeout
                ):
                    try:
                        model, response, latency = future.result(timeout=0.1)  # Quick check
                        
                        if response.strip():
                            valid_count += 1
                            responses.append((model, response))
                            self._emit_thinking(
                                "swarm_response",
                                f"[{valid_count}/{len(self.models)}] {model} responded ({latency:.0f}ms): {response[:]}..."
                            )
                        else:
                            self._emit_thinking(
                                "swarm_empty",
                                f"{model} returned empty response ({latency:.0f}ms)"
                            )
                        
                        # Check if we have enough responses - return IMMEDIATELY!
                        if valid_count >= self.min_responses:
                            elapsed = time.perf_counter() - start_time
                            self._emit_thinking(
                                "swarm_sufficient",
                                f"Got {valid_count} valid responses in {elapsed:.1f}s - returning NOW!"
                            )
                            # Force immediate shutdown - don't wait for remaining threads
                            executor.shutdown(wait=False, cancel_futures=True)
                            
                            # Return immediately with what we have
                            self._emit_thinking(
                                "swarm_complete",
                                f"Swarm returned early in {elapsed:.1f}s: {valid_count} valid responses"
                            )
                            return responses
                            
                    except concurrent.futures.TimeoutError:
                        model = future_to_model[future]
                        self._emit_thinking("swarm_timeout", f"Model {model} timed out")
                    except Exception as e:
                        model = future_to_model.get(future, "unknown")
                        self._emit_thinking("swarm_error", f"Model {model} error: {str(e)[:]}")
            except concurrent.futures.TimeoutError:
                self._emit_thinking("swarm_overall_timeout", f"Overall timeout reached after {overall_timeout}s")
        finally:
            # Ensure executor is shut down
            executor.shutdown(wait=False, cancel_futures=True)
        
        elapsed = time.perf_counter() - start_time
        early_exit = valid_count >= self.min_responses and len(responses) < len(self.models)
        self._emit_thinking(
            "swarm_complete",
            f"Swarm {'returned early' if early_exit else 'completed'} in {elapsed:.1f}s: {valid_count} valid responses collected"
        )
        
        # If we don't have minimum responses, add emergency fallback
        if valid_count < self.min_responses:
            self._emit_thinking(
                "swarm_insufficient",
                f"Only got {valid_count} responses (need {self.min_responses}), adding fallback"
            )
            if not responses:
                responses.append((
                    "fallback",
                    f"[Swarm failed] Unable to get sufficient responses for: {instruction[:]}..."
                ))
        
        return responses
