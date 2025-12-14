"""
Backtracking and state management for TGR execution.

Provides intelligent retry mechanisms when node verification fails,
with state preservation for efficient re-execution.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .node_verifier import VerificationResult


class RetryStrategy(Enum):
    """Strategies for retrying failed nodes."""
    RETRY_SAME = "retry_same"  # Retry with same parameters
    ADJUST_PARAMS = "adjust_params"  # Lower temperature, add constraints
    ALTERNATIVE_APPROACH = "alternative_approach"  # Switch role or method
    EXPAND_CONTEXT = "expand_context"  # Add more RAG context
    SKIP = "skip"  # Skip this node and continue


@dataclass
class BacktrackDecision:
    """Decision about whether and how to backtrack."""
    
    should_backtrack: bool
    target_node_id: str
    strategy: RetryStrategy
    reason: str
    adjustments: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        return self.should_backtrack


@dataclass
class NodeState:
    """State of a single node during execution."""
    
    node_id: str
    output: str = ""
    retry_count: int = 0
    max_retries: int = 2
    last_error: Optional[str] = None
    verified: bool = False
    temperature_adjustment: float = 0.0
    role_override: Optional[str] = None
    extra_context: Optional[str] = None
    
    @property
    def can_retry(self) -> bool:
        """Check if this node can be retried."""
        return self.retry_count < self.max_retries


class StateManager:
    """
    Manages execution state for backtracking.
    
    Provides checkpointing and rollback capabilities to support
    efficient re-execution of failed nodes.
    """
    
    def __init__(self):
        self.node_states: Dict[str, NodeState] = {}
        self.checkpoints: List[Dict[str, str]] = []
        self._results_history: List[Dict[str, str]] = []
    
    def initialize_node(self, node_id: str, max_retries: int = 2) -> NodeState:
        """
        Initialize state for a node.
        
        Args:
            node_id: The node identifier
            max_retries: Maximum retry attempts
        
        Returns:
            NodeState object for the node
        """
        if node_id not in self.node_states:
            self.node_states[node_id] = NodeState(
                node_id=node_id,
                max_retries=max_retries,
            )
        return self.node_states[node_id]
    
    def get_node_state(self, node_id: str) -> Optional[NodeState]:
        """Get the state of a node."""
        return self.node_states.get(node_id)
    
    def update_node(
        self,
        node_id: str,
        output: str,
        verified: bool = False,
        error: Optional[str] = None,
    ) -> NodeState:
        """
        Update a node's state after execution.
        
        Args:
            node_id: The node identifier
            output: The node's output
            verified: Whether the output was verified
            error: Any error that occurred
        
        Returns:
            Updated NodeState
        """
        state = self.initialize_node(node_id)
        state.output = output
        state.verified = verified
        state.last_error = error
        return state
    
    def increment_retry(self, node_id: str) -> int:
        """
        Increment the retry count for a node.
        
        Returns:
            New retry count
        """
        state = self.initialize_node(node_id)
        state.retry_count += 1
        return state.retry_count
    
    def save_checkpoint(self, results: Dict[str, str]) -> int:
        """
        Save current state for potential rollback.
        
        Args:
            results: Current results dictionary
        
        Returns:
            Checkpoint ID
        """
        checkpoint_id = len(self.checkpoints)
        self.checkpoints.append(copy.deepcopy(results))
        self._results_history.append(copy.deepcopy(results))
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: int) -> Dict[str, str]:
        """
        Restore to a previous checkpoint.
        
        Args:
            checkpoint_id: The checkpoint to restore
        
        Returns:
            Results dictionary from that checkpoint
        """
        if 0 <= checkpoint_id < len(self.checkpoints):
            return copy.deepcopy(self.checkpoints[checkpoint_id])
        return {}
    
    def get_last_checkpoint(self) -> Optional[Dict[str, str]]:
        """Get the most recent checkpoint."""
        if self.checkpoints:
            return copy.deepcopy(self.checkpoints[-1])
        return None
    
    def invalidate_downstream(
        self,
        node_id: str,
        edges: List[Tuple[str, str]],
    ) -> List[str]:
        """
        Mark all nodes dependent on node_id as needing re-execution.
        
        Args:
            node_id: The failed node
            edges: List of (source, target) edges
        
        Returns:
            List of invalidated node IDs
        """
        invalidated: Set[str] = set()
        to_check = [node_id]
        
        while to_check:
            current = to_check.pop(0)
            for src, tgt in edges:
                if src == current and tgt not in invalidated:
                    invalidated.add(tgt)
                    to_check.append(tgt)
                    # Clear the state for downstream nodes
                    if tgt in self.node_states:
                        self.node_states[tgt].output = ""
                        self.node_states[tgt].verified = False
        
        return list(invalidated)
    
    def apply_adjustments(
        self,
        node_id: str,
        adjustments: Dict[str, Any],
    ) -> NodeState:
        """
        Apply retry adjustments to a node's state.
        
        Args:
            node_id: The node to adjust
            adjustments: Dictionary of adjustments to apply
        
        Returns:
            Updated NodeState
        """
        state = self.initialize_node(node_id)
        
        if "temperature_delta" in adjustments:
            state.temperature_adjustment += adjustments["temperature_delta"]
        if "role_override" in adjustments:
            state.role_override = adjustments["role_override"]
        if "extra_context" in adjustments:
            state.extra_context = adjustments["extra_context"]
        
        return state
    
    def reset(self) -> None:
        """Reset all state."""
        self.node_states.clear()
        self.checkpoints.clear()
        self._results_history.clear()


def _default_emit(stage: str, content: str) -> None:
    """Default no-op emit function."""
    pass


class BacktrackManager:
    """
    Manages backtracking decisions for TGR execution.
    
    Determines when to retry nodes and how to adjust parameters
    based on verification results.
    """
    
    def __init__(
        self,
        max_depth: int = 3,
        max_retries_per_node: int = 2,
        thinking_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize the backtrack manager.
        
        Args:
            max_depth: Maximum backtrack depth (nodes to go back)
            max_retries_per_node: Maximum retries for any single node
            thinking_callback: Callback for logging decisions
        """
        self.max_depth = max_depth
        self.max_retries_per_node = max_retries_per_node
        self._emit = thinking_callback or _default_emit
        self.state_manager = StateManager()
        self.backtrack_history: List[BacktrackDecision] = []
        self._total_backtracks = 0
    
    @property
    def total_backtracks(self) -> int:
        """Total number of backtracks performed."""
        return self._total_backtracks
    
    def decide_backtrack(
        self,
        failed_node_id: str,
        verification_result: VerificationResult,
        node_type: str,
        node_role: str,
        edges: List[Tuple[str, str]],
    ) -> BacktrackDecision:
        """
        Decide whether to backtrack and how.
        
        Args:
            failed_node_id: The node that failed verification
            verification_result: The verification result
            node_type: Type of the failed node
            node_role: Role of the failed node
            edges: Graph edges for dependency analysis
        
        Returns:
            BacktrackDecision with strategy and adjustments
        """
        node_state = self.state_manager.get_node_state(failed_node_id)
        
        # Check if we've exceeded retry limits
        if node_state and not node_state.can_retry:
            self._emit(
                "backtrack_exhausted",
                f"Node {failed_node_id} has exhausted retries ({node_state.retry_count}/{node_state.max_retries})"
            )
            return BacktrackDecision(
                should_backtrack=False,
                target_node_id=failed_node_id,
                strategy=RetryStrategy.SKIP,
                reason="Max retries exceeded",
            )
        
        # Check total backtrack limit
        if self._total_backtracks >= self.max_depth * len(self.state_manager.node_states or [1]):
            self._emit("backtrack_limit", "Global backtrack limit reached")
            return BacktrackDecision(
                should_backtrack=False,
                target_node_id=failed_node_id,
                strategy=RetryStrategy.SKIP,
                reason="Global backtrack limit reached",
            )
        
        # Determine strategy based on suggested fix and node type
        strategy, adjustments = self._select_strategy(
            node_type,
            node_role,
            verification_result,
            node_state,
        )
        
        # Find the best node to retry
        target_node = self._find_backtrack_target(
            failed_node_id,
            node_type,
            edges,
        )
        
        decision = BacktrackDecision(
            should_backtrack=True,
            target_node_id=target_node,
            strategy=strategy,
            reason=self._format_reason(verification_result, strategy),
            adjustments=adjustments,
        )
        
        self.backtrack_history.append(decision)
        self._total_backtracks += 1
        
        self._emit(
            "backtrack_decision",
            f"Backtracking to {target_node} with strategy {strategy.value}: {decision.reason}"
        )
        
        return decision
    
    def _select_strategy(
        self,
        node_type: str,
        node_role: str,
        verification_result: VerificationResult,
        node_state: Optional[NodeState],
    ) -> Tuple[RetryStrategy, Dict[str, Any]]:
        """
        Select the appropriate retry strategy.
        
        Returns:
            Tuple of (strategy, adjustments dictionary)
        """
        adjustments: Dict[str, Any] = {}
        
        # Use suggested fix from verification if available
        suggested = verification_result.suggested_fix
        if suggested == "retry_same":
            return RetryStrategy.RETRY_SAME, adjustments
        elif suggested == "adjust_params":
            adjustments["temperature_delta"] = -0.1
            return RetryStrategy.ADJUST_PARAMS, adjustments
        elif suggested == "alternative_approach":
            # Switch role based on current type
            if node_role == "logic":
                adjustments["role_override"] = "research"
            elif node_role == "math":
                adjustments["role_override"] = "research"
            else:
                adjustments["role_override"] = "logic"
            return RetryStrategy.ALTERNATIVE_APPROACH, adjustments
        
        # Default strategy based on node type and retry count
        retry_count = node_state.retry_count if node_state else 0
        
        if retry_count == 0:
            # First retry: try same approach
            return RetryStrategy.RETRY_SAME, adjustments
        elif retry_count == 1:
            # Second retry: adjust parameters
            adjustments["temperature_delta"] = -0.1
            return RetryStrategy.ADJUST_PARAMS, adjustments
        else:
            # Third+ retry: try alternative approach
            if node_type == "calculation":
                adjustments["role_override"] = "research"
            elif node_type == "verification":
                # For verification failures, retry the calculation node instead
                adjustments["extra_context"] = "Previous verification failed. Be more careful."
            return RetryStrategy.ALTERNATIVE_APPROACH, adjustments
    
    def _find_backtrack_target(
        self,
        failed_node_id: str,
        node_type: str,
        edges: List[Tuple[str, str]],
    ) -> str:
        """
        Find the best node to retry.
        
        For verification nodes, often want to retry the upstream calculation.
        """
        if node_type == "verification":
            # Find the dependency of this verification node
            deps = [src for src, tgt in edges if tgt == failed_node_id]
            if deps:
                # Retry the last dependency (usually the calculation)
                return deps[-1]
        
        # Default: retry the failed node itself
        return failed_node_id
    
    def _format_reason(
        self,
        verification_result: VerificationResult,
        strategy: RetryStrategy,
    ) -> str:
        """Format a human-readable reason for the backtrack."""
        issues = verification_result.issues[:2] if verification_result.issues else []
        issue_str = "; ".join(issues) if issues else "verification failed"
        return f"{issue_str} -> {strategy.value}"
    
    def prepare_retry(
        self,
        node_id: str,
        decision: BacktrackDecision,
    ) -> Dict[str, Any]:
        """
        Prepare adjustments for a retry attempt.
        
        Args:
            node_id: The node to retry
            decision: The backtrack decision
        
        Returns:
            Dictionary of parameters to adjust
        """
        # Increment retry count
        self.state_manager.increment_retry(node_id)
        
        # Apply adjustments
        if decision.adjustments:
            self.state_manager.apply_adjustments(node_id, decision.adjustments)
        
        # Return parameters for the retry
        state = self.state_manager.get_node_state(node_id)
        return {
            "retry_count": state.retry_count if state else 1,
            "temperature_adjustment": state.temperature_adjustment if state else 0.0,
            "role_override": state.role_override if state else None,
            "extra_context": state.extra_context if state else None,
        }
    
    def should_verify_node(self, node_type: str, node_role: str) -> bool:
        """
        Determine if a node should be verified.
        
        Some nodes (e.g., definition nodes) may not need verification
        to avoid unnecessary backtracking.
        """
        # Always verify calculation and verification nodes
        if node_type in ("calculation", "verification", "aggregation"):
            return True
        
        # Verify math roles
        if node_role == "math":
            return True
        
        # Optional: verify definition and enumeration on later retries
        return False
    
    def reset(self) -> None:
        """Reset the backtrack manager."""
        self.state_manager.reset()
        self.backtrack_history.clear()
        self._total_backtracks = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get backtracking statistics."""
        strategy_counts: Dict[str, int] = {}
        for decision in self.backtrack_history:
            key = decision.strategy.value
            strategy_counts[key] = strategy_counts.get(key, 0) + 1
        
        return {
            "total_backtracks": self._total_backtracks,
            "strategy_counts": strategy_counts,
            "decisions": len(self.backtrack_history),
            "nodes_with_retries": sum(
                1 for s in self.state_manager.node_states.values()
                if s.retry_count > 0
            ),
        }

