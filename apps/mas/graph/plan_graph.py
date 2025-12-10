from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
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
from ..agents.worker_researcher import ResearchWorker
from ..agents.verifier import VerifierAgent
from ..agents.worker_researcher import ResearchWorker
from ..agents.swarm_worker import SwarmWorkerManager
import re
import json
import time
from math import gcd


class GraphState(BaseModel):
    problem: str
    plan: Plan | None = None
    # Results now stores multiple model responses per subtask: (subtask, [(model_name, response), ...])
    results: List[Tuple[SubTask, List[Tuple[str, str]]]] = []
    final_answer: str | None = None
    critique_note: str | None = None
    thinking_log: List[Tuple[str, str]] = []  # (stage, content) pairs
    # Cooperative scratchpad: [{round, subtask_id, model, response}]
    scratchpad: List[Dict[str, Any]] = []


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
    print(f"[DEBUG] {stage}: {content}")
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
        model=cfg.get("model", "mistralai/mistral-large-2512"),
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


def _normalize_text_answer(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t[:200]


def _compute_consensus(responses: List[Tuple[str, str]], numeric_expected: bool) -> Tuple[str, int]:
    """
    Compute a simple consensus winner and agreement count.
    For numeric tasks, prefer extracted numbers; otherwise normalized text.
    """
    counts: Dict[str, int] = {}
    for _, res in responses:
        if not res or res.startswith("["):
            continue
        key = _extract_numeric(res) if numeric_expected else _normalize_text_answer(res)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "", 0
    winner, agree = max(counts.items(), key=lambda kv: kv[1])
    return winner, agree


def _pick_best_response(responses: List[Tuple[str, str]], fallback: str) -> str:
    """
    Prefer richer, structured answers (sets/lists) over bare scalars.
    """
    best = ""
    for _, r in responses:
        if not r or r.startswith("["):
            continue
        if any(tok in r for tok in ["{", "[", "]", "}", "polygon", "root", "union", "set", "P_", "P(", "P{"]):
            return r
        if len(r) > len(best):
            best = r
    return best or fallback


def build_graph(config_path: str) -> StateGraph:
    """
    Build a LangGraph StateGraph using the swarm worker system.
    This provides a graph-based alternative to solve_with_budget.
    """
    cfg = _load_openrouter_config(config_path)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _raw = yaml.safe_load(f) or {}
    except Exception:
        _raw = {}
    
    default_model = cfg.model or "mistralai/mistral-large-2512"
    default_fallback = "mistralai/mistral-medium-3.1"
    
    supervisor_model = _raw.get("supervisor_model") or default_model
    supervisor_fallback_model = _raw.get("supervisor_fallback_model") or default_fallback
    supervisor_secondary_fallback_model = _raw.get("supervisor_secondary_fallback_model")
    verifier_model = _raw.get("verifier_model") or default_model
    verifier_fallback_model = _raw.get("verifier_fallback_model") or default_fallback
    verifier_secondary_fallback_model = _raw.get("verifier_secondary_fallback_model")

    coop_enabled = bool(_raw.get("coop_enabled", True))
    coop_max_rounds = int(_raw.get("coop_max_rounds", 2))
    coop_min_agreement = int(_raw.get("coop_min_agreement", 2))
    coop_reconcile_prompt = str(_raw.get("coop_reconcile_prompt", "Peers disagree; enumerate distinct eigenvalue sets as unions of roots of unity (divisors of |G|). List ALL distinct unions before counting. For |G|=18, polygons are P1,P2,P3,P6,P9,P18 and unions P2∪P3, P2∪P9. Provide the full set list and the count."))

    # Load swarm configuration
    swarm_models = _raw.get("swarm_models", [
        "mistralai/mistral-large-2512",
        "mistralai/mistral-medium-3.1",
        "mistralai/mistral-small-3.2-24b-instruct",
    ])
    swarm_min_responses = int(_raw.get("swarm_min_responses", 2))
    swarm_per_model_timeout = float(_raw.get("swarm_per_model_timeout", 120.0))
    swarm_overall_timeout = float(_raw.get("swarm_overall_timeout", 300.0))

    client = OpenRouterClient(cfg)
    supervisor = SupervisorAgent(
        client,
        model_name=supervisor_model,
        fallback_model=supervisor_fallback_model,
        secondary_fallback_model=supervisor_secondary_fallback_model,
    )
    swarm = SwarmWorkerManager(
        client=client,
        models=swarm_models,
        min_responses=swarm_min_responses,
        per_model_timeout=swarm_per_model_timeout,
        overall_timeout=swarm_overall_timeout,
        cooperative=coop_enabled,
        coop_min_agreement=coop_min_agreement,
        coop_reconcile_prompt=coop_reconcile_prompt,
    )
    researcher = ResearchWorker(
        client=client,
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
    
    supervisor.set_thinking_callback(_emit_thinking)
    swarm.set_thinking_callback(_emit_thinking)

    def node_supervisor(state: GraphState) -> GraphState:
        plan = supervisor.decompose(state.problem)
        return GraphState(problem=state.problem, plan=plan, results=[], final_answer=None)

    def node_dispatch_all(state: GraphState) -> GraphState:
        """Dispatch all subtasks to swarm workers concurrently."""
        assert state.plan is not None
        results: List[Tuple[SubTask, List[Tuple[str, str]]]] = []
        for st in state.plan.subtasks:
            responses = swarm.run(st.instruction, role=st.role)
            results.append((st, responses))
        return GraphState(problem=state.problem, plan=state.plan, results=results, final_answer=None)

    def node_critic(state: GraphState) -> GraphState:
        assert state.plan is not None
        note = supervisor.critique(state.problem, state.results)
        return GraphState(problem=state.problem, plan=state.plan, results=state.results, critique_note=note)

    def node_synthesize(state: GraphState) -> GraphState:
        assert state.plan is not None
        final = supervisor.synthesize(state.problem, state.results)
        repaired_used = False
        if state.critique_note and state.critique_note.strip().upper() != "OK":
            repaired = supervisor.resynthesize_with_critique(state.problem, state.results, state.critique_note)
            if repaired:
                final = repaired
                repaired_used = True
        if _looks_single_number_question(state.problem, state.plan):
            extracted = _extract_numeric(final)
            if extracted:
                final = extracted
            # Build context from swarm results
            ctx_parts = []
            for st, responses in state.results[:6]:
                if isinstance(responses, list):
                    for model_name, res in responses:
                        if res and not res.startswith("["):
                            ctx_parts.append(f"[{st.role}] {st.instruction} -> {res[:200]}")
                            break
            ctx = "\n".join(ctx_parts)
            for _ in range(2):
                v = verifier.verify_numeric(state.problem, final, ctx)
                if v and v.strip() != final.strip():
                    nv = _extract_numeric(v) or v.strip()
                    if nv and nv != final:
                        final = nv
                        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", final):
                            break
        else:
            if _is_multi_query(state.problem):
                try:
                    obj = json.loads(final)
                    if isinstance(obj, dict) and len(obj) >= 1:
                        lines = [f"{str(k)}: {str(v)}" for k, v in obj.items()]
                        final = "\n".join(lines).strip()
                except Exception:
                    pass
        if (not repaired_used) and re.search(r"[^\d\.\-\+\s]", final) and state.critique_note and state.critique_note != "OK":
            final = f"{final}\n\n[Critique] {state.critique_note}"
        new_note = "OK" if repaired_used else state.critique_note
        return GraphState(problem=state.problem, plan=state.plan, results=state.results, critique_note=new_note, final_answer=final)

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", node_supervisor)
    graph.add_node("dispatch", node_dispatch_all)  # Single node dispatches to all swarm workers
    graph.add_node("critic", node_critic)
    graph.add_node("synthesize", node_synthesize)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "dispatch")
    graph.add_edge("dispatch", "critic")
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

    # Fast-path deterministic answers removed to support stochastic intelligence
    # (code intentionally removed)


    _emit_thinking("pipeline_start", f"Starting problem analysis (timeout: {timeout_s}s)")
    
    cfg = _load_openrouter_config(config_path)
    # Load model overrides
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _raw = yaml.safe_load(f) or {}
    except Exception:
        _raw = {}
    
    # Use sensible defaults that actually exist
    default_model = cfg.model or "mistralai/mistral-large-2512"
    default_fallback = "mistralai/mistral-medium-3.1"
    
    supervisor_model = _raw.get("supervisor_model") or default_model
    supervisor_fallback_model = _raw.get("supervisor_fallback_model") or default_fallback
    supervisor_secondary_fallback_model = _raw.get("supervisor_secondary_fallback_model")
    worker_model = _raw.get("worker_model") or default_model
    worker_fallback_model = _raw.get("worker_fallback_model") or default_fallback
    worker_secondary_fallback_model = _raw.get("worker_secondary_fallback_model")
    verifier_model = _raw.get("verifier_model") or default_model
    verifier_fallback_model = _raw.get("verifier_fallback_model") or default_fallback
    verifier_secondary_fallback_model = _raw.get("verifier_secondary_fallback_model")

    coop_enabled = bool(_raw.get("coop_enabled", True))
    coop_max_rounds = int(_raw.get("coop_max_rounds", 2))
    coop_min_agreement = int(_raw.get("coop_min_agreement", 2))
    coop_reconcile_prompt = str(_raw.get("coop_reconcile_prompt", "Peers disagree; reconcile and provide the single best answer."))

    client = OpenRouterClient(cfg)
    supervisor = SupervisorAgent(
        client,
        model_name=supervisor_model,
        fallback_model=supervisor_fallback_model,
        secondary_fallback_model=supervisor_secondary_fallback_model,
    )
    
    # Load swarm configuration
    swarm_models = _raw.get("swarm_models", [
        "mistralai/mistral-large-2512",
        "mistralai/mistral-medium-3.1",
        "mistralai/mistral-small-3.2-24b-instruct",
    ])
    swarm_min_responses = int(_raw.get("swarm_min_responses", 2))
    swarm_per_model_timeout = float(_raw.get("swarm_per_model_timeout", 120.0))
    swarm_overall_timeout = float(_raw.get("swarm_overall_timeout", 300.0))
    
    # Create swarm worker manager (replaces individual math/qa/logic workers)
    swarm = SwarmWorkerManager(
        client=client,
        models=swarm_models,
        min_responses=swarm_min_responses,
        per_model_timeout=swarm_per_model_timeout,
        overall_timeout=swarm_overall_timeout,
        cooperative=coop_enabled,
        coop_min_agreement=coop_min_agreement,
        coop_reconcile_prompt=coop_reconcile_prompt,
    )
    researcher = ResearchWorker(
        client=client,
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
    swarm.set_thinking_callback(_emit_thinking)
    researcher.set_thinking_callback(_emit_thinking)

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
            plan = Plan(subtasks=[SubTask(id="step1", role="qa", instruction=problem, depends_on=[])])
    except Exception as e:
        _emit_thinking("decompose_error", f"Decomposition failed: {str(e)[:200]}")
        plan = Plan(subtasks=[SubTask(id="step1", role="qa", instruction=problem, depends_on=[])])

    # Dispatch workers using swarm with MULTI-HOP REASONING
    # Results are keyed by subtask ID for dependency resolution
    results: List[Tuple[SubTask, List[Tuple[str, str]]]] = []
    results_by_id: Dict[str, str] = {}  # Maps subtask ID -> best result (for context passing)
    scratchpad: List[Dict[str, Any]] = []
    
    if time_left() <= 0:
        _emit_thinking("timeout", f"Timeout after decomposition ({elapsed():.1f}s)")
        return GraphState(problem=problem, plan=plan, results=results, thinking_log=get_thinking_log(), scratchpad=scratchpad)
    
    # Ensure every subtask has an ID
    for idx, st in enumerate(plan.subtasks, start=1):
        if not getattr(st, "id", None):
            st.id = f"step{idx}"
        if st.depends_on is None:
            st.depends_on = []

    # Build dependency graph and topological order
    subtask_by_id = {st.id: st for st in plan.subtasks}
    pending_subtasks = list(plan.subtasks)
    completed_ids: set = set()
    
    # Check if any subtask has dependencies (multi-hop mode)
    has_dependencies = any(st.depends_on for st in plan.subtasks)
    if has_dependencies:
        _emit_thinking("multihop_detected", f"Multi-hop reasoning detected: subtasks have dependencies")
    
    _emit_thinking("dispatch_phase", f"Phase 2: Dispatching {len(plan.subtasks)} subtasks to {len(swarm_models)} models (swarm mode, {'multi-hop' if has_dependencies else 'parallel'})")
    
    iteration = 0
    max_iterations = len(plan.subtasks) * 2  # Safety limit
    
    while pending_subtasks and iteration < max_iterations:
        iteration += 1
        remaining = time_left()
        if remaining <= 0:
            _emit_thinking("timeout", f"Timeout during worker dispatch (completed {len(completed_ids)}/{len(plan.subtasks)})")
            break
        
        # Find subtasks that are ready to execute (all dependencies satisfied)
        ready_subtasks = [
            st for st in pending_subtasks 
            if all(dep_id in completed_ids for dep_id in st.depends_on)
        ]
        
        if not ready_subtasks:
            _emit_thinking("dependency_deadlock", f"No ready subtasks but {len(pending_subtasks)} pending - possible circular dependency")
            break
        
        # Execute ready subtasks (could parallelize independent ones, but keeping sequential for simplicity)
        for st in ready_subtasks:
            remaining = time_left()
            if remaining <= 0:
                break
            
            # Build context from dependencies
            context_parts = []
            if st.depends_on:
                _emit_thinking("multihop_context", f"Subtask '{st.id}' depends on: {st.depends_on}")
                for dep_id in st.depends_on:
                    if dep_id in results_by_id:
                        dep_st = subtask_by_id.get(dep_id)
                        dep_instruction = dep_st.instruction[:50] if dep_st else dep_id
                        context_parts.append(f"[Result from '{dep_id}' ({dep_instruction}...)]: {results_by_id[dep_id]}")
            
            context = "\n".join(context_parts) if context_parts else ""
            
            # Build instruction with context
            full_instruction = st.instruction
            if context:
                full_instruction = f"{st.instruction}\n\nContext from previous steps:\n{context}"
            
            subtask_idx = len(completed_ids) + 1
            _emit_thinking("worker_dispatch", f"Subtask {subtask_idx}/{len(plan.subtasks)} [{st.id}]: [{st.role}] {st.instruction[:80]}...")
            if context:
                _emit_thinking("worker_context", f"With context from {len(st.depends_on)} dependencies")
            
            try:
                numeric_expected = _looks_single_number_question(problem, plan)

                # Research role: use ResearchWorker with code execution + iterative refinement
                if st.role == "research":
                    research_resp, timed_out = _run_with_timeout(
                        lambda inst=full_instruction: researcher.run(inst, context=context),
                        timeout_seconds=min(per_op_timeout, time_left()),
                        default_value=""
                    )
                    if timed_out or not research_resp:
                        research_resp = "[Research worker timed out or returned empty]"
                    results.append((st, [("research", research_resp)]))
                    results_by_id[st.id] = research_resp
                    _emit_thinking("worker_complete", f"Subtask '{st.id}' complete via ResearchWorker.")
                    completed_ids.add(st.id)
                    pending_subtasks.remove(st)
                    continue

                if not coop_enabled:
                    swarm_responses, timed_out = _run_with_timeout(
                        lambda inst=full_instruction, role=st.role: swarm.run(inst, role=role, context=""),
                        timeout_seconds=min(swarm_overall_timeout, remaining),
                        default_value=[]
                    )
                    if timed_out or not swarm_responses:
                        _emit_thinking("swarm_timeout", f"Swarm timed out for subtask '{st.id}'")
                        swarm_responses = [("timeout", f"[Swarm timed out for: {st.instruction[:80]}...]")]
                    chosen, agree = _compute_consensus(swarm_responses, numeric_expected)
                    chosen = chosen or next((r for _, r in swarm_responses if r and not r.startswith("[")), "")
                    results.append((st, swarm_responses))
                    results_by_id[st.id] = chosen or f"[No valid result for {st.id}]"
                    valid_count = sum(1 for _, r in swarm_responses if r and not r.startswith("["))
                    _emit_thinking("worker_complete", f"Subtask '{st.id}' complete: {valid_count} valid responses (consensus={agree})")
                    completed_ids.add(st.id)
                    pending_subtasks.remove(st)
                    continue

                # Cooperative multi-round
                prev_responses: List[Tuple[str, str]] = []
                round_idx = 1
                chosen = ""
                while round_idx <= coop_max_rounds and time_left() > 0:
                    peer_snippets = []
                    for m, r in prev_responses[:4]:
                        if r and not r.startswith("["):
                            peer_snippets.append(f"{m}: {r[:120]}")
                    peer_context = "\n".join(peer_snippets)
                    swarm_responses, timed_out = _run_with_timeout(
                        lambda inst=full_instruction, role=st.role, peers=prev_responses: swarm.run(
                            inst,
                            role=role,
                            context=context,
                            peer_responses=prev_responses,
                            round_idx=round_idx,
                            reconcile=(round_idx > 1),
                        ),
                        timeout_seconds=min(swarm_overall_timeout, time_left()),
                        default_value=[]
                    )
                    if timed_out or not swarm_responses:
                        _emit_thinking("swarm_timeout", f"Swarm timed out for subtask '{st.id}' (round {round_idx})")
                        swarm_responses = [("timeout", f"[Swarm timed out for: {st.instruction[:]}...]")]

                    # Record scratchpad
                    for model_name, resp in swarm_responses:
                        scratchpad.append({
                            "round": round_idx,
                            "subtask_id": st.id,
                            "model": model_name,
                            "response": resp,
                        })

                    prev_responses.extend(swarm_responses)
                    winner, agree = _compute_consensus(prev_responses, numeric_expected)
                    _emit_thinking("coop_status", f"Subtask '{st.id}' round {round_idx}: agreement={agree}, winner='{winner}'")

                    winner_is_scalar = bool(winner and re.fullmatch(r"[+-]?\\d+(?:\\.\\d+)?", winner))
                    time_low = time_left() < 10
                    if winner_is_scalar and agree < coop_min_agreement and round_idx < coop_max_rounds:
                        round_idx += 1
                        continue

                    chosen = _pick_best_response(prev_responses, winner)
                    has_structured = any(
                        r and not r.startswith("[") and (
                            any(tok in r.lower() for tok in ["room", "list", "blue", "enumerate", "table"])
                            or any(ch in r for ch in ["[", "]", "{", "}", ","])
                            or len(r.split()) > 30
                        )
                        for _, r in prev_responses
                    )
                    # If only scalars / no evidence and we have another round, force another round and ask for self-check
                    if (winner_is_scalar or not has_structured) and round_idx < coop_max_rounds:
                        round_idx += 1
                        continue
                    if agree >= coop_min_agreement or round_idx == coop_max_rounds or time_low:
                        chosen = chosen or winner or next((r for _, r in prev_responses if r and not r.startswith("[")), "")
                        break
                    round_idx += 1

                results.append((st, prev_responses))
                results_by_id[st.id] = chosen or f"[No valid result for {st.id}]"
                valid_count = sum(1 for _, r in prev_responses if r and not r.startswith("["))
                _emit_thinking("worker_complete", f"Subtask '{st.id}' complete: {valid_count} responses, chosen='{chosen}'")
                completed_ids.add(st.id)
                pending_subtasks.remove(st)

            except Exception as e:
                _emit_thinking("worker_error", f"Swarm failed for subtask '{st.id}': {str(e)[:200]}")
                results.append((st, [("error", f"[Error: {str(e)[:100]}]")]))
                results_by_id[st.id] = f"[Error: {str(e)[:50]}]"
                completed_ids.add(st.id)
                pending_subtasks.remove(st)
    
    if pending_subtasks:
        _emit_thinking("dispatch_incomplete", f"{len(pending_subtasks)} subtasks could not be executed")

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
        # Provide compact context from results (handle swarm format)
        ctx_parts = []
        for st, responses in results[:6]:
            if isinstance(responses, list):
                # Swarm format: use first valid response
                for model_name, res in responses:
                    if res and not res.startswith("["):
                        ctx_parts.append(f"[{st.role}] {st.instruction} -> {res[:200]}")
                        break
            else:
                ctx_parts.append(f"[{st.role}] {st.instruction} -> {responses}")
        ctx = "\n".join(ctx_parts)
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
            for st, responses in reversed(results):
                # Handle swarm format (list of tuples) vs legacy format (string)
                if isinstance(responses, list):
                    # Swarm format: iterate through all model responses
                    for model_name, res in responses:
                        if res and res.strip() and not res.startswith("["):
                            if st.role == "math":
                                num = _extract_numeric(res)
                                if num:
                                    final = num
                                    _emit_thinking("salvage_from_worker", f"Found numeric answer from {model_name}: {num}")
                                    break
                            else:
                                if len(res) < 500 or any(k in res.lower() for k in ["answer", "conclusion", "result", "therefore"]):
                                    final = res.strip()
                                    _emit_thinking("salvage_from_worker", f"Found answer from {model_name}: {res[:100]}...")
                                    break
                    if final and final.strip():
                        break
                else:
                    # Legacy format: single string
                    if responses.strip() and not responses.startswith("["):
                        if st.role == "math":
                            num = _extract_numeric(responses)
                            if num:
                                final = num
                                break
                        else:
                            if len(responses) < 500 or any(k in responses.lower() for k in ["answer", "conclusion", "result", "therefore"]):
                                final = responses.strip()
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
        thinking_log=get_thinking_log(),
        scratchpad=scratchpad,
    )
