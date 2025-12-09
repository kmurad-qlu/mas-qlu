from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
import concurrent.futures
import threading

import yaml
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from ..infra.openrouter.client import OpenRouterClient, OpenRouterConfig
from ..agents.supervisor import SupervisorAgent, SubTask, Plan
from ..agents.worker_math import MathWorker
from ..agents.worker_qa import QAWorker
from ..agents.worker_logic import LogicWorker
from ..agents.verifier import VerifierAgent
import re
import json
import time


class GraphState(BaseModel):
    problem: str
    plan: Plan | None = None
    results: List[Tuple[SubTask, str]] = []
    final_answer: str | None = None
    critique_note: str | None = None
    thinking_log: List[Tuple[str, str]] = []  # (stage, content) pairs


# Global thinking callback for streaming intermediate results
_thinking_callback: Optional[Callable[[str, str], None]] = None
_thinking_log: List[Tuple[str, str]] = []


def set_thinking_callback(callback: Optional[Callable[[str, str], None]]) -> None:
    """Set a global callback for intermediate thinking updates."""
    global _thinking_callback
    _thinking_callback = callback


def clear_thinking_log() -> None:
    """Clear the thinking log."""
    global _thinking_log
    _thinking_log = []


def get_thinking_log() -> List[Tuple[str, str]]:
    """Get the current thinking log."""
    return list(_thinking_log)


def _emit_thinking(stage: str, content: str) -> None:
    """Emit a thinking update to callback and log."""
    global _thinking_log, _thinking_callback
    _thinking_log.append((stage, content))
    if _thinking_callback:
        _thinking_callback(stage, content)


def _run_with_timeout(func: Callable, timeout_seconds: float, default_value=None):
    """
    Run a function with a timeout. Returns (result, timed_out).
    If the function times out, returns (default_value, True).
    """
    # If timeout is already expired, still try briefly
    if timeout_seconds <= 0:
        timeout_seconds = 5.0  # Give at least 5 seconds
    
    # Use a shared container to capture late results
    result_container = {"value": default_value, "completed": False}
    
    def wrapper():
        try:
            result = func()
            result_container["value"] = result
            result_container["completed"] = True
            return result
        except Exception as e:
            result_container["value"] = default_value
            raise e
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(wrapper)
        try:
            actual_timeout = max(timeout_seconds, 1.0)
            result = future.result(timeout=actual_timeout)
            return result, False
        except concurrent.futures.TimeoutError:
            # Wait a bit more to see if result comes in
            for _ in range(10):  # Check for up to 1 second more
                time.sleep(0.1)
                if result_container["completed"]:
                    _emit_thinking("late_result", f"Result arrived after timeout: {str(result_container['value'])[:80]}...")
                    return result_container["value"], False
            # Check one more time if future completed
            if future.done():
                try:
                    result = future.result(timeout=0.1)
                    return result, False
                except:
                    pass
            return default_value, True
        except Exception as e:
            raise e


def _load_openrouter_config(path: str) -> OpenRouterConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return OpenRouterConfig(
        model=cfg.get("model", "mistralai/mixtral-8x7b-instruct"),
        temperature=float(cfg.get("temperature", 0.2)),
        top_p=float(cfg.get("top_p", 0.95)),
        presence_penalty=float(cfg.get("presence_penalty", 0.0)),
        frequency_penalty=float(cfg.get("frequency_penalty", 0.0)),
        max_output_tokens=int(cfg.get("max_output_tokens", 512)),
        request_timeout_s=int(cfg.get("request_timeout_s", 60)),
        max_retries=int(cfg.get("max_retries", 3)),
    )


def _looks_single_number_question(problem: str, plan: Plan | None) -> bool:
    """
    Heuristic: classify if the prompt expects a single numeric answer.
    Tightened to avoid coercing multi-question prompts into a single number.
    """
    def _is_multi_query(text: str) -> bool:
        t = text.strip().lower()
        if t.count("?") > 1:
            return True
        # Common multi-step discourse markers
        if any(k in t for k in ["and then", "then she", "then he", "also", "as well", "after that"]):
            return True
        return False

    if _is_multi_query(problem):
        return False

    p = problem.lower()
    keywords = ("how many", "what is", "compute", "evaluate", "result", "value of")
    if not any(k in p for k in keywords):
        return False

    if plan is not None:
        math_count = sum(1 for st in plan.subtasks if st.role == "math")
        # Only classify as single-number when there is exactly one math subtask
        if math_count != 1:
            return False
    return True


def _extract_numeric(text: str) -> str | None:
    """
    Extract a single numeric answer from text. Prefer a unique explicit marker.
    Returns the bare number as a string. Avoids guessing when ambiguous.
    """
    # 1) Prefer explicit final-answer marker on its own line
    m = re.search(r"^\s*####\s*([+-]?\d+(?:\.\d+)?)\s*$", text, flags=re.MULTILINE)
    if m:
        return m.group(1)
    # 2) If JSON object with a single numeric field, use it
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            numeric_values = [v for v in obj.values() if isinstance(v, (int, float, str)) and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", str(v).strip())]
            if len(obj) == 1 and len(numeric_values) == 1:
                return str(numeric_values[0]).strip()
    except Exception:
        pass
    # 3) If exactly one number appears, use it; otherwise, do not guess
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    if len(nums) == 1:
        return nums[0]
    return None


def _is_multi_query(problem: str) -> bool:
    """
    Detect whether the prompt likely requests multiple answers.
    """
    t = problem.strip().lower()
    if t.count("?") > 1:
        return True
    if any(k in t for k in ["and then", "then she", "then he", "also", "as well", "after that"]):
        return True
    return False


def build_graph(config_path: str) -> StateGraph:
    cfg = _load_openrouter_config(config_path)
    # Load optional per-role model overrides from config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _raw = yaml.safe_load(f) or {}
    except Exception:
        _raw = {}
    
    # Use sensible defaults that actually exist
    default_model = cfg.model or "openai/gpt-4o"
    default_fallback = "openai/gpt-4o-mini"
    
    supervisor_model = _raw.get("supervisor_model") or default_model
    supervisor_fallback_model = _raw.get("supervisor_fallback_model") or default_fallback
    supervisor_secondary_fallback_model = _raw.get("supervisor_secondary_fallback_model") or "anthropic/claude-3.5-sonnet"
    worker_model = _raw.get("worker_model") or default_model
    worker_fallback_model = _raw.get("worker_fallback_model") or default_fallback
    worker_secondary_fallback_model = _raw.get("worker_secondary_fallback_model") or "anthropic/claude-3.5-sonnet"
    verifier_model = _raw.get("verifier_model") or default_model
    verifier_fallback_model = _raw.get("verifier_fallback_model") or default_fallback
    verifier_secondary_fallback_model = _raw.get("verifier_secondary_fallback_model") or "openai/gpt-4o"

    client = OpenRouterClient(cfg)
    supervisor = SupervisorAgent(
        client,
        model_name=supervisor_model,
        fallback_model=supervisor_fallback_model,
        secondary_fallback_model=supervisor_secondary_fallback_model,
    )
    math = MathWorker(
        client,
        model_name=worker_model,
        fallback_model=worker_fallback_model,
        secondary_fallback_model=worker_secondary_fallback_model,
    )
    qa = QAWorker(
        client,
        model_name=worker_model,
        fallback_model=worker_fallback_model,
        secondary_fallback_model=worker_secondary_fallback_model,
    )
    logic = LogicWorker(
        client,
        model_name=worker_model,
        fallback_model=worker_fallback_model,
        secondary_fallback_model=worker_secondary_fallback_model,
    )
    verifier = VerifierAgent(
        client,
        model_name=verifier_model,
        fallback_model=verifier_fallback_model,
        secondary_fallback_model=verifier_secondary_fallback_model,
    )
    
    # Wire up thinking callbacks
    supervisor.set_thinking_callback(_emit_thinking)
    math.set_thinking_callback(_emit_thinking)
    qa.set_thinking_callback(_emit_thinking)
    logic.set_thinking_callback(_emit_thinking)

    def node_supervisor(state: GraphState) -> GraphState:
        plan = supervisor.decompose(state.problem)
        return GraphState(problem=state.problem, plan=plan, results=[], final_answer=None)

    def node_dispatch_math(state: GraphState) -> GraphState:
        assert state.plan is not None
        results = list(state.results)
        for st in state.plan.subtasks:
            if st.role == "math":
                res = math.run(st.instruction)
                results.append((st, res))
        return GraphState(problem=state.problem, plan=state.plan, results=results, final_answer=None)

    def node_dispatch_qa(state: GraphState) -> GraphState:
        assert state.plan is not None
        results = list(state.results)
        for st in state.plan.subtasks:
            if st.role == "qa":
                res = qa.run(st.instruction)
                results.append((st, res))
        return GraphState(problem=state.problem, plan=state.plan, results=results, final_answer=None)

    def node_dispatch_logic(state: GraphState) -> GraphState:
        assert state.plan is not None
        results = list(state.results)
        for st in state.plan.subtasks:
            if st.role == "logic":
                res = logic.run(st.instruction)
                results.append((st, res))
        return GraphState(problem=state.problem, plan=state.plan, results=results, final_answer=None)

    def node_critic(state: GraphState) -> GraphState:
        assert state.plan is not None
        note = supervisor.critique(state.problem, state.results)
        return GraphState(problem=state.problem, plan=state.plan, results=state.results, critique_note=note)

    def node_synthesize(state: GraphState) -> GraphState:
        assert state.plan is not None
        final = supervisor.synthesize(state.problem, state.results)
        repaired_used = False
        # If critique flagged issues, attempt a second-pass repair synthesis
        if state.critique_note and state.critique_note.strip().upper() != "OK":
            repaired = supervisor.resynthesize_with_critique(state.problem, state.results, state.critique_note)
            if repaired:
                final = repaired
                repaired_used = True
        # Enforce single-number output only when appropriate.
        if _looks_single_number_question(state.problem, state.plan):
            extracted = _extract_numeric(final)
            if extracted:
                final = extracted  # bare number only
            # Non-deterministic verification pass (independent re-check)
            # Provide compact context from worker results
            if state.results:
                ctx_lines = []
                for st, res in state.results[:6]:
                    ctx_lines.append(f"[{st.role}] {st.instruction} -> {res}")
                ctx = "\n".join(ctx_lines)
            else:
                ctx = ""
            # Up to 2 verify iterations
            for _ in range(2):
                v = verifier.verify_numeric(state.problem, final, ctx)
                if v and v.strip() != final.strip():
                    # Accept verifier correction and keep numeric-only
                    nv = _extract_numeric(v) or v.strip()
                    if nv and nv != final:
                        final = nv
                        # Stop if it stabilizes at a numeric
                        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", final):
                            break
        else:
            # Multi-answer prompts: attempt to normalize JSON to readable lines
            if _is_multi_query(state.problem):
                try:
                    obj = json.loads(final)
                    if isinstance(obj, dict) and len(obj) >= 1:
                        # Preserve insertion order if provided
                        lines = []
                        for k, v in obj.items():
                            lines.append(f"{str(k)}: {str(v)}")
                        final = "\n".join(lines).strip()
                except Exception:
                    pass
        # Only append critique to verbose answers when no repair was applied.
        if (not repaired_used) and re.search(r"[^\d\.\-\+\s]", final) and state.critique_note and state.critique_note != "OK":
            final = f"{final}\n\n[Critique] {state.critique_note}"
        # If we repaired, mark critique as OK.
        new_note = "OK" if repaired_used else state.critique_note
        return GraphState(problem=state.problem, plan=state.plan, results=state.results, critique_note=new_note, final_answer=final)

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", node_supervisor)
    graph.add_node("math", node_dispatch_math)
    graph.add_node("qa", node_dispatch_qa)
    graph.add_node("logic", node_dispatch_logic)
    graph.add_node("critic", node_critic)
    graph.add_node("synthesize", node_synthesize)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "math")
    graph.add_edge("math", "qa")
    graph.add_edge("qa", "logic")
    graph.add_edge("logic", "critic")
    graph.add_edge("critic", "synthesize")
    graph.add_edge("synthesize", END)
    return graph


def solve_with_budget(
    problem: str, 
    config_path: str, 
    timeout_s: float = 300.0,
    thinking_callback: Optional[Callable[[str, str], None]] = None
) -> GraphState:
    """
    Best-effort orchestration that respects an overall time budget.
    Returns the best available intermediate result if the budget is exceeded.
    
    Args:
        problem: The problem to solve
        config_path: Path to the config YAML
        timeout_s: Maximum time budget in seconds (default 300s = 5 minutes)
        thinking_callback: Optional callback for intermediate thinking updates
    """
    # Clear and set up thinking log
    clear_thinking_log()
    if thinking_callback:
        set_thinking_callback(thinking_callback)
    
    t0 = time.perf_counter()
    _emit_thinking("pipeline_start", f"Starting problem analysis (timeout: {timeout_s}s)")
    
    cfg = _load_openrouter_config(config_path)
    # Load model overrides
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _raw = yaml.safe_load(f) or {}
    except Exception:
        _raw = {}
    
    # Use sensible defaults that actually exist
    default_model = cfg.model or "openai/gpt-4o"
    default_fallback = "openai/gpt-4o-mini"
    
    supervisor_model = _raw.get("supervisor_model") or default_model
    supervisor_fallback_model = _raw.get("supervisor_fallback_model") or default_fallback
    supervisor_secondary_fallback_model = _raw.get("supervisor_secondary_fallback_model") or "anthropic/claude-3.5-sonnet"
    worker_model = _raw.get("worker_model") or default_model
    worker_fallback_model = _raw.get("worker_fallback_model") or default_fallback
    worker_secondary_fallback_model = _raw.get("worker_secondary_fallback_model") or "anthropic/claude-3.5-sonnet"
    verifier_model = _raw.get("verifier_model") or default_model
    verifier_fallback_model = _raw.get("verifier_fallback_model") or default_fallback
    verifier_secondary_fallback_model = _raw.get("verifier_secondary_fallback_model") or "openai/gpt-4o"

    client = OpenRouterClient(cfg)
    supervisor = SupervisorAgent(
        client,
        model_name=supervisor_model,
        fallback_model=supervisor_fallback_model,
        secondary_fallback_model=supervisor_secondary_fallback_model,
    )
    math = MathWorker(
        client,
        model_name=worker_model,
        fallback_model=worker_fallback_model,
        secondary_fallback_model=worker_secondary_fallback_model,
    )
    qa = QAWorker(
        client,
        model_name=worker_model,
        fallback_model=worker_fallback_model,
        secondary_fallback_model=worker_secondary_fallback_model,
    )
    logic = LogicWorker(
        client,
        model_name=worker_model,
        fallback_model=worker_fallback_model,
        secondary_fallback_model=worker_secondary_fallback_model,
    )
    verifier = VerifierAgent(
        client,
        model_name=verifier_model,
        fallback_model=verifier_fallback_model,
        secondary_fallback_model=verifier_secondary_fallback_model,
    )
    
    # Wire up thinking callbacks
    supervisor.set_thinking_callback(_emit_thinking)
    math.set_thinking_callback(_emit_thinking)
    qa.set_thinking_callback(_emit_thinking)
    logic.set_thinking_callback(_emit_thinking)

    # Helper to check remaining time
    def time_left() -> float:
        return timeout_s - (time.perf_counter() - t0)
    
    def elapsed() -> float:
        return time.perf_counter() - t0

    # Per-operation timeout (use fraction of total budget)
    # Increased to 150s max for complex reasoning tasks
    per_op_timeout = min(timeout_s / 3, 150.0)  # Max 150s per operation
    
    # Decompose
    if time_left() <= 0:
        _emit_thinking("timeout", "Timeout before decomposition")
        return GraphState(problem=problem, thinking_log=get_thinking_log(), final_answer="Timeout before processing could begin.")
    
    _emit_thinking("decompose_phase", "Phase 1: Problem decomposition")
    try:
        plan, timed_out = _run_with_timeout(
            lambda: supervisor.decompose(problem),
            timeout_seconds=min(per_op_timeout, time_left()),
            default_value=None
        )
        if timed_out or plan is None:
            _emit_thinking("decompose_timeout", "Decomposition timed out, using fallback plan")
            plan = Plan(subtasks=[SubTask(role="qa", instruction=problem)])
    except Exception as e:
        _emit_thinking("decompose_error", f"Decomposition failed: {str(e)[:200]}")
        plan = Plan(subtasks=[SubTask(role="qa", instruction=problem)])

    # Dispatch workers
    results: List[Tuple[SubTask, str]] = []
    if time_left() <= 0:
        _emit_thinking("timeout", f"Timeout after decomposition ({elapsed():.1f}s)")
        return GraphState(problem=problem, plan=plan, results=results, thinking_log=get_thinking_log())
    
    _emit_thinking("dispatch_phase", f"Phase 2: Dispatching {len(plan.subtasks)} subtasks to workers")
    for i, st in enumerate(plan.subtasks):
        remaining = time_left()
        if remaining <= 0:
            _emit_thinking("timeout", f"Timeout during worker dispatch (completed {i}/{len(plan.subtasks)})")
            break
        
        _emit_thinking("worker_dispatch", f"Worker {i+1}/{len(plan.subtasks)}: [{st.role}] {st.instruction[:100]}...")
        
        # Use timeout for each worker call
        worker_timeout = min(per_op_timeout, remaining)
        
        try:
            if st.role == "math":
                res, timed_out = _run_with_timeout(
                    lambda inst=st.instruction: math.run(inst),
                    timeout_seconds=worker_timeout,
                    default_value=""
                )
            elif st.role == "qa":
                res, timed_out = _run_with_timeout(
                    lambda inst=st.instruction: qa.run(inst),
                    timeout_seconds=worker_timeout,
                    default_value=""
                )
            elif st.role == "logic":
                res, timed_out = _run_with_timeout(
                    lambda inst=st.instruction: logic.run(inst),
                    timeout_seconds=worker_timeout,
                    default_value=""
                )
            else:
                res = f"Unknown role: {st.role}"
                timed_out = False
            
            if timed_out:
                _emit_thinking("worker_timeout", f"Worker {st.role} timed out after {worker_timeout:.1f}s")
                res = f"[Worker timed out for: {st.instruction[:80]}...]"
            elif not res.strip():
                # Handle empty responses by trying alternate workers with remaining time
                _emit_thinking("worker_retry", f"Empty response from {st.role} worker, trying QA fallback")
                if st.role != "qa" and time_left() > 5:
                    res, _ = _run_with_timeout(
                        lambda inst=st.instruction: qa.run(inst),
                        timeout_seconds=min(30, time_left()),
                        default_value=""
                    )
                if not res.strip():
                    res = f"[No response available for: {st.instruction[:80]}...]"
            
            results.append((st, res))
            _emit_thinking("worker_complete", f"Worker {i+1} completed: {res[:150]}...")
            
        except Exception as e:
            _emit_thinking("worker_error", f"Worker {st.role} failed: {str(e)[:200]}")
            results.append((st, f"[Error: {str(e)[:100]}]"))

    # Critique
    critique = ""
    remaining = time_left()
    if remaining > 0 and results:
        _emit_thinking("critique_phase", "Phase 3: Critiquing worker outputs")
        try:
            critique, timed_out = _run_with_timeout(
                lambda: supervisor.critique(problem, results),
                timeout_seconds=min(per_op_timeout, remaining),
                default_value="OK"
            )
            if timed_out:
                _emit_thinking("critique_timeout", "Critique timed out, skipping")
                critique = "OK"
        except Exception as e:
            _emit_thinking("critique_error", f"Critique failed: {str(e)[:200]}")
            critique = "OK"

    # Synthesize
    final = ""
    remaining = time_left()
    if remaining > 0 or not final:  # Try synthesis even if low on time
        _emit_thinking("synthesize_phase", "Phase 4: Synthesizing final answer")
        try:
            # Give synthesis a reasonable minimum timeout
            synth_timeout = max(min(per_op_timeout, remaining) if remaining > 0 else 30, 30.0)
            synth_result, timed_out = _run_with_timeout(
                lambda: supervisor.synthesize(problem, results),
                timeout_seconds=synth_timeout,
                default_value=""
            )
            
            # IMPORTANT: Only treat as timeout if we got no result
            # The synthesis might have completed just as timeout was checked
            if synth_result and synth_result.strip():
                final = synth_result
                if timed_out:
                    _emit_thinking("synthesize_late", f"Synthesis completed but was slow (still using result: {final[:100]}...)")
            elif timed_out:
                _emit_thinking("synthesize_timeout", "Synthesis timed out with no result")
            
            # Attempt repair if needed and we have time AND critique found issues
            if final and critique and critique.strip().upper() != "OK" and time_left() > 10:
                _emit_thinking("repair_attempt", "Attempting to repair based on critique")
                fixed, _ = _run_with_timeout(
                    lambda: supervisor.resynthesize_with_critique(problem, results, critique),
                    timeout_seconds=min(30, max(time_left(), 15)),
                    default_value=None
                )
                if fixed and fixed.strip():
                    final = fixed
        except Exception as e:
            _emit_thinking("synthesize_error", f"Synthesis failed: {str(e)[:200]}")

    # Numeric enforcement + verifier pass (non-deterministic but low temperature)
    if _looks_single_number_question(problem, plan):
        extracted = _extract_numeric(final)
        if extracted:
            final = extracted
        # Provide compact context from results
        ctx = "\n".join([f"[{st.role}] {st.instruction} -> {res}" for st, res in results[:6]])
        remaining = time_left()
        if remaining > 5:
            _emit_thinking("verify_phase", "Phase 5: Numeric verification")
            try:
                v, timed_out = _run_with_timeout(
                    lambda: verifier.verify_numeric(problem, final, ctx),
                    timeout_seconds=min(30, remaining),
                    default_value=None
                )
                if not timed_out and v and v.strip() != final.strip():
                    nv = _extract_numeric(v) or v.strip()
                    if nv:
                        final = nv
                elif timed_out:
                    _emit_thinking("verify_timeout", "Verification timed out, using current answer")
            except Exception as e:
                _emit_thinking("verify_error", f"Verification failed: {str(e)[:200]}")

    # If timed out / no final yet, present best intermediate
    if not final or not final.strip():
        _emit_thinking("salvage", "Attempting to salvage answer from thinking log and worker outputs")
        
        # First, check thinking log for synthesized answers (these might have arrived late)
        for stage, content in reversed(get_thinking_log()):
            if "synthesize_complete" in stage or "Final answer:" in content:
                # Extract the answer from the log
                if "Final answer:" in content:
                    parts = content.split("Final answer:")
                    if len(parts) > 1:
                        candidate = parts[1].strip().rstrip("...").strip()
                        if candidate and len(candidate) < 500:  # Reasonable length
                            _emit_thinking("salvage_from_log", f"Found synthesized answer in log: {candidate[:100]}")
                            final = candidate
                            break
        
        # If still no answer, try worker outputs
        if not final or not final.strip():
            for st, res in reversed(results):
                if res.strip() and not res.startswith("["):
                    if st.role == "math":
                        num = _extract_numeric(res)
                        if num:
                            final = num
                            break
                    else:
                        # For non-math, prefer shorter answers that look like conclusions
                        if len(res) < 500 or any(k in res.lower() for k in ["answer", "conclusion", "result", "therefore"]):
                            final = res.strip()
                            break
        
        # Final emergency fallback
        if not final or not final.strip():
            final = "Unable to generate a complete answer within the time budget. Please try again or simplify the question."

    note = "OK" if (final and not final.startswith("Unable")) else (critique or "TIMEOUT/NO_ANSWER")
    _emit_thinking("pipeline_complete", f"Completed in {elapsed():.1f}s. Final answer: {final[:200]}...")
    
    return GraphState(
        problem=problem, 
        plan=plan, 
        results=results, 
        critique_note=note, 
        final_answer=final,
        thinking_log=get_thinking_log()
    )
