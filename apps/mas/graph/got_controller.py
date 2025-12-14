from __future__ import annotations

import concurrent.futures
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..agents.swarm_worker import SwarmWorkerManager
from ..agents.verifier import VerifierAgent
from ..agents.worker_researcher import ResearchWorker
from .archetype_verifier import verify_with_template
from .node_verifier import NodeVerifier, VerificationResult
from .backtrack_manager import BacktrackManager, RetryStrategy

if TYPE_CHECKING:
    from ..learning.trace_store import TraceStore


def _default_emit(stage: str, content: str) -> None:
    return None


def _extract_numeric(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"^\s*####\s*([+-]?\d+(?:\.\d+)?)\s*$", text, flags=re.MULTILINE)
    if m:
        return m.group(1)
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    if len(nums) == 1:
        return nums[0]
    return None


def _normalize_text(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t[:200]


def _compute_consensus(responses: List[Tuple[str, str]], numeric_expected: bool) -> Tuple[str, int]:
    counts: Dict[str, int] = {}
    for _, res in responses:
        if not res or res.startswith("["):
            continue
        key = _extract_numeric(res) if numeric_expected else _normalize_text(res)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "", 0
    winner, agree = max(counts.items(), key=lambda kv: kv[1])
    return winner, agree


def _pick_best_response(responses: List[Tuple[str, str]], fallback: str) -> str:
    best = ""
    for _, r in responses:
        if not r or r.startswith("["):
            continue
        if any(tok in r for tok in ["{", "[", "]", "}", "union", "set", "spectrum", "matrix", "quandle"]):
            return r
        if len(r) > len(best):
            best = r
    return best or fallback


def _run_with_timeout(func: Callable, timeout_seconds: float, default_value=None):
    """
    Run a function with a timeout. Returns (result, timed_out).
    """
    container = {"value": default_value, "completed": False}

    def wrapper():
        try:
            res = func()
            container["value"] = res
            container["completed"] = True
            return res
        except Exception:
            container["value"] = default_value
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(wrapper)
        try:
            result = future.result(timeout=timeout_seconds)
            return result, False
        except concurrent.futures.TimeoutError:
            return container["value"], True


@dataclass
class NodeSpec:
    id: str
    type: str
    role: str
    instruction: str


@dataclass
class TGRResult:
    final_answer: str
    trace: List[Dict[str, Any]]
    template_id: str
    verified: bool = False
    verification_method: str = "none"
    total_duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GoTController:
    """
    Template-guided Graph-of-Thought executor.
    Hydrates a graph blueprint into runnable steps that leverage
    the existing swarm + research workers plus verifier.
    
    Now supports RAG integration via:
    - retrieval node type for mid-reasoning document retrieval
    - knowledge seeds augmentation with retrieved context
    """

    def __init__(
        self,
        problem: str,
        template: Dict[str, Any],
        swarm: SwarmWorkerManager,
        researcher: ResearchWorker,
        verifier: VerifierAgent,
        knowledge_seeds: Optional[List[str]] = None,
        node_timeout: float = 90.0,
        overall_timeout: float = 240.0,
        thinking_callback: Optional[Callable[[str, str], None]] = None,
        retriever: Optional[Any] = None,  # HybridRetriever for RAG
        augment_seeds_with_rag: bool = True,
        trace_store: Optional["TraceStore"] = None,  # For distillation learning
        record_traces: bool = True,  # Whether to record traces
        enable_backtracking: bool = False,  # Enable intelligent retry
        max_backtrack_depth: int = 3,  # Maximum backtrack depth
        max_retries_per_node: int = 2,  # Maximum retries per node
    ):
        self.problem = problem
        self.template = template
        self.swarm = swarm
        self.researcher = researcher
        self.verifier = verifier
        self.node_timeout = node_timeout
        self.overall_timeout = overall_timeout
        self._emit = thinking_callback or _default_emit
        self.knowledge_seeds = knowledge_seeds or []
        self.template_id = template.get("template_id", "")
        self.retriever = retriever
        self.augment_seeds_with_rag = augment_seeds_with_rag
        self.trace_store = trace_store
        self.record_traces = record_traces
        self.enable_backtracking = enable_backtracking
        
        # Cache for augmented seeds (computed once)
        self._augmented_seeds: Optional[List[str]] = None
        
        # Node execution timing
        self._node_start_times: Dict[str, float] = {}
        
        # Backtracking components
        self._node_verifier = NodeVerifier(thinking_callback=thinking_callback)
        self._backtrack_manager = BacktrackManager(
            max_depth=max_backtrack_depth,
            max_retries_per_node=max_retries_per_node,
            thinking_callback=thinking_callback,
        )

        blueprint = template.get("graph_blueprint", {})
        nodes = blueprint.get("nodes", [])
        self.nodes: Dict[str, NodeSpec] = {
            n["id"]: NodeSpec(
                id=n["id"],
                type=n.get("type", "logic"),
                role=n.get("role", "logic"),
                instruction=n.get("instruction", ""),
            )
            for n in nodes
        }
        self.edges: List[Tuple[str, str]] = [
            (e["source"], e["target"]) for e in blueprint.get("edges", [])
        ]
        self.entrypoint = blueprint.get("entrypoint") or (nodes[0]["id"] if nodes else None)

    def _augment_seeds_with_rag(self) -> List[str]:
        """
        Enrich static knowledge seeds with retrieved context from RAG.
        
        This method queries the retriever with each knowledge seed to find
        supporting documentation, then appends relevant snippets.
        
        Returns:
            Augmented list of knowledge seeds with retrieved context
        """
        if not self.retriever:
            return self.knowledge_seeds
        
        if not self.knowledge_seeds:
            # If no seeds, try to retrieve based on problem
            try:
                chunks = self.retriever.fusion_search(self.problem, k=3)
                if chunks:
                    self._emit("rag_seed_augment", f"Retrieved {len(chunks)} chunks for problem context")
                    return [
                        f"[Retrieved] {chunk.text[:200]}..."
                        if hasattr(chunk, 'text') else f"[Retrieved] {str(chunk)[:200]}..."
                        for chunk in chunks
                    ]
            except Exception as e:
                self._emit("rag_seed_error", f"Failed to retrieve problem context: {e}")
            return self.knowledge_seeds
        
        augmented = list(self.knowledge_seeds)
        
        # Retrieve supporting context for first few seeds
        for seed in self.knowledge_seeds[:3]:  # Limit to avoid context bloat
            try:
                chunks = self.retriever.fusion_search(seed, k=2)
                for chunk in chunks:
                    chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                    if len(chunk_text) > 50:
                        augmented.append(f"[Retrieved] {chunk_text[:200]}...")
            except Exception as e:
                self._emit("rag_seed_error", f"Failed to augment seed: {e}")
        
        self._emit("rag_seed_augment", f"Augmented {len(self.knowledge_seeds)} seeds to {len(augmented)} entries")
        return augmented
    
    def _run_retrieval(self, instruction: str, context: str) -> str:
        """
        Execute a retrieval node - fetch relevant documents from RAG.
        
        This node type allows mid-reasoning document retrieval to
        ground subsequent reasoning steps in factual content.
        
        Args:
            instruction: The retrieval instruction/query
            context: Current context from previous nodes
        
        Returns:
            Formatted string of retrieved documents
        """
        if not self.retriever:
            self._emit("rag_no_retriever", "No retriever configured for retrieval node")
            return "[No retriever configured - skipping retrieval]"
        
        # Extract query from instruction and context
        query = f"{instruction} {context}"[:500]
        
        try:
            chunks = self.retriever.fusion_search(query, k=5)
            
            if not chunks:
                self._emit("rag_no_results", f"No results for: {query[:100]}...")
                return "[No relevant documents found]"
            
            # Format retrieved documents
            formatted = []
            for i, chunk in enumerate(chunks, 1):
                title = chunk.title if hasattr(chunk, 'title') else "Document"
                text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                score = chunk.score if hasattr(chunk, 'score') else 0.0
                
                formatted.append(f"[{i}] {title} (relevance: {score:.3f})\n{text[:400]}...")
            
            result = "\n\n".join(formatted)
            self._emit("rag_retrieval_complete", f"Retrieved {len(chunks)} documents")
            return result
            
        except Exception as e:
            self._emit("rag_retrieval_error", f"Retrieval failed: {e}")
            return f"[Retrieval error: {str(e)[:100]}]"
    
    def _build_context(self, dep_ids: List[str], results: Dict[str, str]) -> str:
        ctx_parts: List[str] = []
        
        # Use augmented seeds if RAG is enabled
        if self.augment_seeds_with_rag and self.retriever:
            if self._augmented_seeds is None:
                self._augmented_seeds = self._augment_seeds_with_rag()
            seeds_to_use = self._augmented_seeds
        else:
            seeds_to_use = self.knowledge_seeds
        
        if seeds_to_use:
            ctx_parts.append("Knowledge seeds:\n- " + "\n- ".join(seeds_to_use))
        
        for dep in dep_ids:
            if dep in results:
                snippet = results[dep]
                ctx_parts.append(f"[{dep}] {snippet[:400]}")
        ctx_parts.append(f"[Problem] {self.problem[:400]}")
        return "\n".join(ctx_parts)

    def _run_logic(self, instruction: str, role: str, context: str, numeric_expected: bool) -> str:
        responses, timed_out = _run_with_timeout(
            lambda: self.swarm.run(instruction, role=role, context=context),
            timeout_seconds=self.node_timeout,
            default_value=[],
        )
        if timed_out:
            self._emit("tgr_timeout", f"Swarm timed out for: {instruction[:80]}")
        winner, agree = _compute_consensus(responses, numeric_expected)
        chosen = _pick_best_response(responses, winner)
        chosen = chosen or winner or f"[no response for {instruction[:60]}]"
        self._emit("tgr_node_complete", f"Role={role} agree={agree} winner={winner}")
        return chosen

    def _run_research(self, instruction: str, context: str) -> str:
        result, timed_out = _run_with_timeout(
            lambda: self.researcher.run(instruction, context=context),
            timeout_seconds=self.node_timeout,
            default_value="[research timed out]",
        )
        if timed_out:
            self._emit("tgr_timeout", f"Research timed out for: {instruction[:80]}")
        return result

    def _topological_order(self) -> List[str]:
        indegree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for src, tgt in self.edges:
            if tgt in indegree:
                indegree[tgt] += 1
        queue = [nid for nid, deg in indegree.items() if deg == 0]
        ordered: List[str] = []
        while queue:
            nid = queue.pop(0)
            ordered.append(nid)
            for src, tgt in self.edges:
                if src == nid:
                    indegree[tgt] -= 1
                    if indegree[tgt] == 0:
                        queue.append(tgt)
        return ordered

    def _execute_node(
        self,
        node: NodeSpec,
        context: str,
        retry_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Execute a single node with optional retry parameters.
        
        Args:
            node: The node specification
            context: Context from dependencies
            retry_params: Optional retry adjustments (role_override, temperature, etc.)
        
        Returns:
            Tuple of (output, success, error_message)
        """
        instruction = f"{node.instruction}\n\nContext:\n{context}"
        
        # Apply retry adjustments if present
        role = node.role
        extra_context = ""
        if retry_params:
            if retry_params.get("role_override"):
                role = retry_params["role_override"]
            if retry_params.get("extra_context"):
                extra_context = f"\n\nNote: {retry_params['extra_context']}"
                instruction += extra_context
        
        numeric_expected = bool(re.search(r"####", node.instruction))
        error_msg: Optional[str] = None
        node_success = True
        
        try:
            # Handle different node types
            if node.type == "retrieval" or node.role == "rag":
                output = self._run_retrieval(node.instruction, context)
            elif role == "research" or node.type == "calculation":
                output = self._run_research(instruction, context=context)
            elif node.type == "aggregation":
                output = self._run_logic(instruction, role="logic", context=context, numeric_expected=numeric_expected)
            elif node.type == "verification":
                # For verification, we need access to results - handled in run()
                output = "[verification handled separately]"
            else:
                output = self._run_logic(instruction, role=role, context=context, numeric_expected=numeric_expected)
            
            # Check for error indicators
            if output.startswith("[") and ("error" in output.lower() or "timeout" in output.lower()):
                node_success = False
                error_msg = output[:200]
        except Exception as e:
            output = f"[error: {str(e)[:150]}]"
            error_msg = str(e)[:200]
            node_success = False
        
        return output, node_success, error_msg

    def _execute_node_with_backtracking(
        self,
        node: NodeSpec,
        deps: List[str],
        results: Dict[str, str],
        trace: List[Dict[str, Any]],
        start_time: float,
    ) -> Tuple[str, bool, Optional[str], int]:
        """
        Execute a node with backtracking support.
        
        Returns:
            Tuple of (output, success, error_message, retry_count)
        """
        context = self._build_context(deps, results)
        retry_count = 0
        
        while True:
            # Check timeout
            if (time.perf_counter() - start_time) > self.overall_timeout:
                return "[timeout]", False, "Overall timeout reached", retry_count
            
            # Get retry parameters if this is a retry
            node_state = self._backtrack_manager.state_manager.get_node_state(node.id)
            retry_params = None
            if node_state and node_state.retry_count > 0:
                retry_params = {
                    "role_override": node_state.role_override,
                    "extra_context": node_state.extra_context,
                    "temperature_adjustment": node_state.temperature_adjustment,
                }
            
            # Execute the node
            node_start = time.perf_counter()
            output, success, error_msg = self._execute_node(node, context, retry_params)
            node_duration_ms = (time.perf_counter() - node_start) * 1000
            
            # For verification nodes, handle specially
            if node.type == "verification":
                candidate = results.get(deps[-1], "") if deps else ""
                numeric_guess = _extract_numeric(candidate) or candidate
                verified = self.verifier.verify_numeric(self.problem, numeric_guess, context)
                output = verified or candidate
                success = bool(verified and verified.strip())
            
            # If backtracking is disabled or node succeeded, return
            if not self.enable_backtracking or success:
                return output, success, error_msg, retry_count
            
            # Check if we should verify this node type
            if not self._backtrack_manager.should_verify_node(node.type, node.role):
                return output, success, error_msg, retry_count
            
            # Verify the output
            verification = self._node_verifier.verify_node_output(
                node_id=node.id,
                node_type=node.type,
                role=node.role,
                output=output,
                context=context,
                instruction=node.instruction,
            )
            
            if verification.passed:
                return output, True, None, retry_count
            
            # Decide whether to backtrack
            decision = self._backtrack_manager.decide_backtrack(
                failed_node_id=node.id,
                verification_result=verification,
                node_type=node.type,
                node_role=node.role,
                edges=self.edges,
            )
            
            if not decision.should_backtrack:
                # Can't backtrack further, return what we have
                return output, success, error_msg, retry_count
            
            # Prepare for retry
            self._backtrack_manager.prepare_retry(node.id, decision)
            retry_count += 1
            
            self._emit(
                "backtrack_retry",
                f"Retrying {node.id} (attempt {retry_count + 1}) with strategy {decision.strategy.value}"
            )
            
            # Rebuild context in case of role/approach change
            if decision.strategy == RetryStrategy.EXPAND_CONTEXT and self.retriever:
                # Add more RAG context
                try:
                    extra_chunks = self.retriever.fusion_search(self.problem, k=3)
                    extra_context = "\n".join([
                        f"[Extra] {c.text[:200]}..." if hasattr(c, 'text') else str(c)[:200]
                        for c in extra_chunks
                    ])
                    node_state = self._backtrack_manager.state_manager.get_node_state(node.id)
                    if node_state:
                        node_state.extra_context = extra_context
                except Exception:
                    pass

    def run(self) -> TGRResult:
        start = time.perf_counter()
        results: Dict[str, str] = {}
        trace: List[Dict[str, Any]] = []
        ordered = self._topological_order()
        if not ordered:
            return TGRResult(final_answer="", trace=[], template_id=self.template.get("template_id", ""))
        
        # Reset backtrack manager for fresh run
        if self.enable_backtracking:
            self._backtrack_manager.reset()

        for node_id in ordered:
            if (time.perf_counter() - start) > self.overall_timeout:
                self._emit("tgr_timeout", "Overall TGR timeout reached")
                trace.append({
                    "node": node_id, 
                    "type": "timeout", 
                    "role": "system", 
                    "output": "[timeout]",
                    "error": "Overall timeout reached",
                    "duration_ms": 0,
                    "success": False,
                })
                break
            node = self.nodes[node_id]
            deps = [src for src, tgt in self.edges if tgt == node_id]
            context = self._build_context(deps, results)

            self._emit("tgr_node_start", f"{node_id} ({node.type}/{node.role})")
            
            # Track node timing
            node_start = time.perf_counter()
            self._node_start_times[node_id] = node_start
            
            # Use backtracking execution if enabled for appropriate node types
            if self.enable_backtracking and self._backtrack_manager.should_verify_node(node.type, node.role):
                output, node_success, error_msg, retry_count = self._execute_node_with_backtracking(
                    node=node,
                    deps=deps,
                    results=results,
                    trace=trace,
                    start_time=start,
                )
            else:
                # Standard execution without backtracking
                instruction = f"{node.instruction}\n\nContext:\n{context}"
                numeric_expected = bool(re.search(r"####", node.instruction))
                error_msg: Optional[str] = None
                node_success = True
                retry_count = 0
                
                try:
                    # Handle different node types
                    if node.type == "retrieval" or node.role == "rag":
                        # RAG retrieval node - fetch documents from knowledge base
                        output = self._run_retrieval(node.instruction, context)
                    elif node.role == "research" or node.type == "calculation":
                        output = self._run_research(instruction, context=context)
                    elif node.type == "aggregation":
                        output = self._run_logic(instruction, role="logic", context=context, numeric_expected=numeric_expected)
                    elif node.type == "verification":
                        candidate = results.get(deps[-1], "") if deps else ""
                        numeric_guess = _extract_numeric(candidate) or candidate
                        verified = self.verifier.verify_numeric(self.problem, numeric_guess, context)
                        output = verified or candidate
                    else:
                        output = self._run_logic(instruction, role=node.role, context=context, numeric_expected=numeric_expected)
                    
                    # Check for error indicators in output
                    if output.startswith("[") and ("error" in output.lower() or "timeout" in output.lower()):
                        node_success = False
                        error_msg = output[:200]
                except Exception as e:
                    output = f"[error: {str(e)[:150]}]"
                    error_msg = str(e)[:200]
                    node_success = False

            node_duration_ms = (time.perf_counter() - node_start) * 1000
            results[node_id] = output
            
            # Enhanced trace entry with timing and context info
            trace.append({
                "node": node_id, 
                "type": node.type, 
                "role": node.role, 
                "instruction": node.instruction[:300],
                "context": context[:500],
                "output": output,
                "duration_ms": node_duration_ms,
                "success": node_success,
                "error": error_msg,
                "retry_count": retry_count,
            })

        # Pick last available node output as final
        final_answer = ""
        if ordered:
            for nid in reversed(ordered):
                if nid in results:
                    final_answer = results[nid]
                    break

        # Final verifier pass if numeric marker present
        verified = False
        verification_method = "none"
        if final_answer:
            if _extract_numeric(final_answer):
                verifier_result = self.verifier.verify_numeric(self.problem, final_answer, "\n".join([t["output"] for t in trace[-3:]]))
                if verifier_result and verifier_result.strip():
                    final_answer = verifier_result.strip()
                    verified = True
                    verification_method = "verifier"
            # Archetype-specific correction
            if self.template_id:
                corrected = verify_with_template(self.problem, final_answer, self.template_id)
                if corrected:
                    final_answer = corrected
                    verified = True
                    verification_method = "archetype"

        elapsed = time.perf_counter() - start
        elapsed_ms = elapsed * 1000
        self._emit("tgr_complete", f"TGR finished in {elapsed:.1f}s with final: {final_answer[:120]}")
        
        # Build result with metadata
        metadata: Dict[str, Any] = {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "rag_enabled": self.retriever is not None,
            "seeds_augmented": self._augmented_seeds is not None,
        }
        
        # Add backtracking statistics if enabled
        if self.enable_backtracking:
            backtrack_stats = self._backtrack_manager.get_statistics()
            metadata["backtracking"] = backtrack_stats
        
        result = TGRResult(
            final_answer=final_answer, 
            trace=trace, 
            template_id=self.template.get("template_id", ""),
            verified=verified,
            verification_method=verification_method,
            total_duration_ms=elapsed_ms,
            metadata=metadata,
        )
        
        # Record trace for distillation learning
        if self.record_traces and self.trace_store:
            try:
                self._record_trace(result)
            except Exception as e:
                self._emit("trace_record_error", f"Failed to record trace: {str(e)[:100]}")
        
        return result
    
    def _record_trace(self, result: TGRResult) -> None:
        """
        Record execution trace for distillation learning.
        
        Only records high-quality traces (verified with reasonable success rate).
        """
        if not self.trace_store:
            return
        
        # Import here to avoid circular imports
        from ..learning.trace_recorder import TraceRecorder, ExecutionTrace, NodeTrace
        
        # Convert trace entries to NodeTrace objects
        nodes: List[NodeTrace] = []
        for entry in result.trace:
            nodes.append(NodeTrace(
                node_id=entry.get("node", "unknown"),
                node_type=entry.get("type", "unknown"),
                role=entry.get("role", "unknown"),
                instruction=entry.get("instruction", ""),
                context=entry.get("context", ""),
                output=entry.get("output", ""),
                duration_ms=entry.get("duration_ms", 0.0),
                success=entry.get("success", True),
                error=entry.get("error"),
                retry_count=entry.get("retry_count", 0),
            ))
        
        # Create execution trace
        import uuid
        from datetime import datetime
        
        trace = ExecutionTrace(
            trace_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            problem=self.problem,
            template_id=result.template_id,
            nodes=nodes,
            final_answer=result.final_answer,
            verified=result.verified,
            verification_method=result.verification_method,
            total_duration_ms=result.total_duration_ms,
            metadata=result.metadata,
        )
        
        # Check quality before saving
        recorder = TraceRecorder()
        if recorder.is_high_quality(trace):
            self.trace_store.save(trace)
            self._emit("trace_recorded", f"Recorded high-quality trace: {trace.trace_id[:8]}")
        else:
            self._emit("trace_skipped", "Trace not high-quality enough for learning")

