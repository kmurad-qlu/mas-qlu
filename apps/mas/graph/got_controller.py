from __future__ import annotations

import concurrent.futures
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..agents.swarm_worker import SwarmWorkerManager
from ..agents.verifier import VerifierAgent
from ..agents.worker_researcher import ResearchWorker
from .archetype_verifier import verify_with_template


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
        
        # Cache for augmented seeds (computed once)
        self._augmented_seeds: Optional[List[str]] = None

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

    def run(self) -> TGRResult:
        start = time.perf_counter()
        results: Dict[str, str] = {}
        trace: List[Dict[str, Any]] = []
        ordered = self._topological_order()
        if not ordered:
            return TGRResult(final_answer="", trace=[], template_id=self.template.get("template_id", ""))

        for node_id in ordered:
            if (time.perf_counter() - start) > self.overall_timeout:
                self._emit("tgr_timeout", "Overall TGR timeout reached")
                break
            node = self.nodes[node_id]
            deps = [src for src, tgt in self.edges if tgt == node_id]
            context = self._build_context(deps, results)
            instruction = f"{node.instruction}\n\nContext:\n{context}"
            numeric_expected = bool(re.search(r"####", node.instruction))

            self._emit("tgr_node_start", f"{node_id} ({node.type}/{node.role})")
            
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

            results[node_id] = output
            trace.append({"node": node_id, "type": node.type, "role": node.role, "output": output})

        # Pick last available node output as final
        final_answer = ""
        if ordered:
            for nid in reversed(ordered):
                if nid in results:
                    final_answer = results[nid]
                    break

        # Final verifier pass if numeric marker present
        if final_answer:
            if _extract_numeric(final_answer):
                verified = self.verifier.verify_numeric(self.problem, final_answer, "\n".join([t["output"] for t in trace[-3:]]))
                if verified and verified.strip():
                    final_answer = verified.strip()
            # Archetype-specific correction
            if self.template_id:
                corrected = verify_with_template(self.problem, final_answer, self.template_id)
                if corrected:
                    final_answer = corrected

        elapsed = time.perf_counter() - start
        self._emit("tgr_complete", f"TGR finished in {elapsed:.1f}s with final: {final_answer[:120]}")
        return TGRResult(final_answer=final_answer, trace=trace, template_id=self.template.get("template_id", ""))

