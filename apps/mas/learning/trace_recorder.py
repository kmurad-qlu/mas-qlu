"""
Trace recording for TGR execution.

Captures complete execution traces including node inputs, outputs,
timings, and verification status for distillation learning.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class NodeTrace:
    """Trace for a single node execution."""
    
    node_id: str
    node_type: str
    role: str
    instruction: str
    context: str
    output: str
    duration_ms: float
    success: bool
    error: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "role": self.role,
            "instruction": self.instruction[:500],  # Truncate for storage
            "context": self.context[:1000],  # Truncate for storage
            "output": self.output[:2000],  # Truncate for storage
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "retry_count": self.retry_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeTrace":
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            node_type=data["node_type"],
            role=data["role"],
            instruction=data.get("instruction", ""),
            context=data.get("context", ""),
            output=data.get("output", ""),
            duration_ms=data.get("duration_ms", 0.0),
            success=data.get("success", True),
            error=data.get("error"),
            retry_count=data.get("retry_count", 0),
        )


@dataclass
class ExecutionTrace:
    """Complete execution trace for a TGR run."""
    
    trace_id: str
    timestamp: datetime
    problem: str
    template_id: str
    nodes: List[NodeTrace]
    final_answer: str
    verified: bool
    verification_method: str  # "archetype", "verifier", "benchmark", "none"
    total_duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "problem": self.problem[:1000],  # Truncate for storage
            "template_id": self.template_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "final_answer": self.final_answer[:500],
            "verified": self.verified,
            "verification_method": self.verification_method,
            "total_duration_ms": self.total_duration_ms,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionTrace":
        """Create from dictionary."""
        return cls(
            trace_id=data["trace_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            problem=data.get("problem", ""),
            template_id=data.get("template_id", ""),
            nodes=[NodeTrace.from_dict(n) for n in data.get("nodes", [])],
            final_answer=data.get("final_answer", ""),
            verified=data.get("verified", False),
            verification_method=data.get("verification_method", "none"),
            total_duration_ms=data.get("total_duration_ms", 0.0),
            metadata=data.get("metadata", {}),
        )
    
    def problem_hash(self) -> str:
        """Generate a hash of the problem for deduplication."""
        normalized = self.problem.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    @property
    def success_rate(self) -> float:
        """Calculate node success rate."""
        if not self.nodes:
            return 0.0
        successful = sum(1 for n in self.nodes if n.success)
        return successful / len(self.nodes)
    
    @property
    def had_retries(self) -> bool:
        """Check if any node required retries."""
        return any(n.retry_count > 0 for n in self.nodes)


class TraceRecorder:
    """
    Records execution traces from TGR runs.
    
    Captures node-level details including inputs, outputs, timings,
    and verification status for later analysis and learning.
    """
    
    def __init__(self):
        self._current_nodes: List[NodeTrace] = []
        self._start_time: Optional[float] = None
        self._node_start_times: Dict[str, float] = {}
    
    def start_trace(self) -> None:
        """Start recording a new trace."""
        self._current_nodes = []
        self._start_time = time.perf_counter()
        self._node_start_times = {}
    
    def start_node(self, node_id: str) -> None:
        """Mark the start of a node execution."""
        self._node_start_times[node_id] = time.perf_counter()
    
    def record_node(
        self,
        node_id: str,
        node_type: str,
        role: str,
        instruction: str,
        context: str,
        output: str,
        success: bool = True,
        error: Optional[str] = None,
        retry_count: int = 0,
    ) -> None:
        """Record a completed node execution."""
        start_time = self._node_start_times.get(node_id, time.perf_counter())
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        trace = NodeTrace(
            node_id=node_id,
            node_type=node_type,
            role=role,
            instruction=instruction,
            context=context,
            output=output,
            duration_ms=duration_ms,
            success=success,
            error=error,
            retry_count=retry_count,
        )
        self._current_nodes.append(trace)
    
    def finalize_trace(
        self,
        problem: str,
        template_id: str,
        final_answer: str,
        verified: bool = False,
        verification_method: str = "none",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionTrace:
        """
        Finalize and return the complete execution trace.
        
        Args:
            problem: The original problem text
            template_id: ID of the template used
            final_answer: The final answer produced
            verified: Whether the answer was verified
            verification_method: How verification was done
            metadata: Additional metadata (RAG context, etc.)
        
        Returns:
            Complete ExecutionTrace object
        """
        total_duration = 0.0
        if self._start_time is not None:
            total_duration = (time.perf_counter() - self._start_time) * 1000
        
        trace = ExecutionTrace(
            trace_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            problem=problem,
            template_id=template_id,
            nodes=list(self._current_nodes),
            final_answer=final_answer,
            verified=verified,
            verification_method=verification_method,
            total_duration_ms=total_duration,
            metadata=metadata or {},
        )
        
        # Reset for next trace
        self._current_nodes = []
        self._start_time = None
        self._node_start_times = {}
        
        return trace
    
    def is_high_quality(self, trace: ExecutionTrace) -> bool:
        """
        Determine if a trace is suitable for learning.
        
        High-quality traces:
        - Are verified (correct answer)
        - Have high node success rate (>80%)
        - Have reasonable duration (not timed out)
        - Have non-empty final answer
        """
        if not trace.verified:
            return False
        if not trace.final_answer or not trace.final_answer.strip():
            return False
        if trace.success_rate < 0.8:
            return False
        # Exclude traces that took too long (likely timeout issues)
        if trace.total_duration_ms > 300_000:  # 5 minutes
            return False
        return True
    
    def record_from_got_result(
        self,
        result: Any,  # TGRResult
        problem: str,
        verified: bool = False,
        verification_method: str = "none",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionTrace:
        """
        Create an ExecutionTrace from a TGRResult object.
        
        This is a convenience method for integration with GoTController.
        """
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
                success=not entry.get("error"),
                error=entry.get("error"),
                retry_count=entry.get("retry_count", 0),
            ))
        
        return ExecutionTrace(
            trace_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            problem=problem,
            template_id=result.template_id,
            nodes=nodes,
            final_answer=result.final_answer,
            verified=verified,
            verification_method=verification_method,
            total_duration_ms=sum(n.duration_ms for n in nodes),
            metadata=metadata or {},
        )

