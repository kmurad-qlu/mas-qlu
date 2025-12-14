"""
Distillation manager for TGR self-improvement.

Coordinates trace collection, pattern analysis, and prompt enhancement
for continuous learning from successful executions.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .trace_recorder import ExecutionTrace, TraceRecorder
from .trace_store import TraceStore
from .pattern_analyzer import PatternAnalyzer, ReasoningPattern, FailurePattern
from .prompt_enhancer import PromptEnhancer, EnhancedInstruction

if TYPE_CHECKING:
    from ..graph.got_controller import TGRResult


def _default_emit(stage: str, content: str) -> None:
    """Default no-op emit function."""
    pass


class DistillationManager:
    """
    Manages the complete distillation loop.
    
    Coordinates:
    - Trace collection from TGR executions
    - Periodic pattern analysis
    - Prompt enhancement for future runs
    - Statistics tracking and reporting
    """
    
    DEFAULT_TRACES_PATH = "apps/mas/data/traces"
    DEFAULT_PATTERNS_PATH = "apps/mas/data/patterns"
    
    def __init__(
        self,
        traces_path: Optional[str] = None,
        patterns_path: Optional[str] = None,
        min_traces_for_analysis: int = 50,
        analysis_interval: int = 100,  # Traces between analyses
        thinking_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize the distillation manager.
        
        Args:
            traces_path: Path to store traces
            patterns_path: Path to store patterns
            min_traces_for_analysis: Minimum traces before first analysis
            analysis_interval: Number of new traces between analyses
            thinking_callback: Callback for logging
        """
        self.traces_path = traces_path or self.DEFAULT_TRACES_PATH
        self.patterns_path = patterns_path or self.DEFAULT_PATTERNS_PATH
        self.min_traces_for_analysis = min_traces_for_analysis
        self.analysis_interval = analysis_interval
        self._emit = thinking_callback or _default_emit
        
        # Initialize components
        self.trace_store = TraceStore(storage_path=self.traces_path)
        self.trace_recorder = TraceRecorder()
        self.pattern_analyzer = PatternAnalyzer()
        self.prompt_enhancer = PromptEnhancer()
        
        # State tracking
        self._traces_since_analysis = 0
        self._last_analysis_time: Optional[datetime] = None
        self._patterns: List[ReasoningPattern] = []
        self._failure_patterns: List[FailurePattern] = []
        
        # Load existing patterns if available
        self._load_patterns()
    
    def _load_patterns(self) -> None:
        """Load patterns from disk if available."""
        patterns_file = Path(self.patterns_path) / "patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._patterns = [
                    ReasoningPattern(**p) for p in data.get("success_patterns", [])
                ]
                self._failure_patterns = [
                    FailurePattern(**p) for p in data.get("failure_patterns", [])
                ]
                self.prompt_enhancer.update_patterns(
                    self._patterns,
                    self._failure_patterns,
                )
                self._emit(
                    "distill_patterns_loaded",
                    f"Loaded {len(self._patterns)} success patterns, {len(self._failure_patterns)} failure patterns"
                )
            except Exception as e:
                self._emit("distill_load_error", f"Failed to load patterns: {str(e)[:100]}")
    
    def _save_patterns(self) -> None:
        """Save patterns to disk."""
        patterns_dir = Path(self.patterns_path)
        patterns_dir.mkdir(parents=True, exist_ok=True)
        
        patterns_file = patterns_dir / "patterns.json"
        data = {
            "success_patterns": [p.to_dict() for p in self._patterns],
            "failure_patterns": [p.to_dict() for p in self._failure_patterns],
            "last_analysis": self._last_analysis_time.isoformat() if self._last_analysis_time else None,
            "total_traces": self.trace_store.count(),
        }
        
        with open(patterns_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        self._emit("distill_patterns_saved", f"Saved {len(self._patterns)} patterns")
    
    def record_trace(
        self,
        result: "TGRResult",
        problem: str,
        verified: bool = False,
        verification_method: str = "none",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExecutionTrace]:
        """
        Record a TGR execution trace.
        
        Args:
            result: TGR execution result
            problem: Original problem
            verified: Whether the answer was verified correct
            verification_method: How verification was done
            metadata: Additional metadata
        
        Returns:
            The recorded trace if saved, None otherwise
        """
        trace = self.trace_recorder.record_from_got_result(
            result=result,
            problem=problem,
            verified=verified,
            verification_method=verification_method,
            metadata=metadata,
        )
        
        # Check quality before saving
        if self.trace_recorder.is_high_quality(trace):
            self.trace_store.save(trace)
            self._traces_since_analysis += 1
            self._emit("distill_trace_recorded", f"Recorded trace {trace.trace_id[:8]} (verified={verified})")
            
            # Check if we should trigger analysis
            self._maybe_trigger_analysis()
            
            return trace
        else:
            self._emit("distill_trace_skipped", "Trace not high-quality, skipped")
            return None
    
    def _maybe_trigger_analysis(self) -> None:
        """Check if analysis should be triggered."""
        total_traces = self.trace_store.count(verified_only=True)
        
        # Check if we have enough traces
        if total_traces < self.min_traces_for_analysis:
            return
        
        # Check if we've collected enough new traces
        if self._traces_since_analysis < self.analysis_interval:
            return
        
        # Trigger analysis
        self._emit("distill_analysis_triggered", f"Triggering analysis after {self._traces_since_analysis} new traces")
        self.run_analysis()
    
    def run_analysis(self, force: bool = False) -> Dict[str, Any]:
        """
        Run pattern analysis on collected traces.
        
        Args:
            force: Force analysis even if not enough traces
        
        Returns:
            Analysis results summary
        """
        total_traces = self.trace_store.count(verified_only=True)
        
        if not force and total_traces < self.min_traces_for_analysis:
            self._emit(
                "distill_analysis_skipped",
                f"Not enough traces ({total_traces} < {self.min_traces_for_analysis})"
            )
            return {"skipped": True, "reason": "insufficient_traces"}
        
        self._emit("distill_analysis_start", f"Analyzing {total_traces} traces...")
        
        # Load traces
        all_traces = self.trace_store.query(verified_only=False, limit=1000)
        verified_traces = [t for t in all_traces if t.verified]
        failed_traces = [t for t in all_traces if not t.verified]
        
        # Extract patterns
        self._patterns = self.pattern_analyzer.extract_patterns(verified_traces)
        self._failure_patterns = self.pattern_analyzer.identify_failure_patterns(failed_traces)
        
        # Update prompt enhancer
        self.prompt_enhancer.update_patterns(self._patterns, self._failure_patterns)
        
        # Save patterns
        self._save_patterns()
        
        # Reset counter
        self._traces_since_analysis = 0
        self._last_analysis_time = datetime.now()
        
        # Get statistics
        stats = self.pattern_analyzer.get_template_statistics(all_traces)
        
        result = {
            "total_traces_analyzed": len(all_traces),
            "verified_traces": len(verified_traces),
            "failed_traces": len(failed_traces),
            "success_patterns_found": len(self._patterns),
            "failure_patterns_found": len(self._failure_patterns),
            "template_statistics": stats,
            "analysis_time": self._last_analysis_time.isoformat(),
        }
        
        self._emit(
            "distill_analysis_complete",
            f"Found {len(self._patterns)} success patterns, {len(self._failure_patterns)} failure patterns"
        )
        
        return result
    
    def enhance_instruction(
        self,
        instruction: str,
        node_type: str,
        node_role: str,
        template_id: str,
    ) -> EnhancedInstruction:
        """
        Enhance a node instruction using learned patterns.
        
        Args:
            instruction: Original instruction
            node_type: Node type
            node_role: Node role
            template_id: Template ID
        
        Returns:
            Enhanced instruction
        """
        # Get relevant traces for few-shot examples
        traces = self.trace_store.get_template_traces(
            template_id=template_id,
            verified_only=True,
            limit=20,
        )
        
        return self.prompt_enhancer.enhance_instruction(
            instruction=instruction,
            node_type=node_type,
            node_role=node_role,
            template_id=template_id,
            traces=traces,
        )
    
    def get_similar_traces(
        self,
        problem: str,
        limit: int = 5,
    ) -> List[ExecutionTrace]:
        """
        Find traces with similar problems.
        
        Useful for few-shot example selection.
        """
        traces = self.trace_store.query(verified_only=True, limit=100)
        return self.pattern_analyzer.find_similar_successful_traces(
            problem=problem,
            traces=traces,
            max_results=limit,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive distillation statistics."""
        trace_stats = self.trace_store.get_statistics()
        enhancer_stats = self.prompt_enhancer.get_enhancement_statistics()
        
        return {
            "traces": trace_stats,
            "patterns": {
                "success_patterns": len(self._patterns),
                "failure_patterns": len(self._failure_patterns),
                "last_analysis": self._last_analysis_time.isoformat() if self._last_analysis_time else None,
                "traces_since_analysis": self._traces_since_analysis,
            },
            "enhancement": enhancer_stats,
        }
    
    def prune_old_traces(
        self,
        keep_verified: int = 1000,
        keep_unverified: int = 100,
    ) -> int:
        """
        Prune old traces to manage storage.
        
        Returns:
            Number of traces pruned
        """
        pruned = self.trace_store.prune_old_traces(
            keep_verified=keep_verified,
            keep_unverified=keep_unverified,
        )
        if pruned > 0:
            self._emit("distill_pruned", f"Pruned {pruned} old traces")
        return pruned
    
    def reset(self) -> None:
        """Reset all distillation state."""
        self._patterns = []
        self._failure_patterns = []
        self._traces_since_analysis = 0
        self._last_analysis_time = None
        self.prompt_enhancer.update_patterns([], [])
        self._emit("distill_reset", "Distillation state reset")


# Global singleton for easy access
_distillation_manager: Optional[DistillationManager] = None


def get_distillation_manager(
    traces_path: Optional[str] = None,
    patterns_path: Optional[str] = None,
    thinking_callback: Optional[Callable[[str, str], None]] = None,
) -> DistillationManager:
    """
    Get or create the global distillation manager.
    
    Args:
        traces_path: Optional custom traces path
        patterns_path: Optional custom patterns path
        thinking_callback: Optional logging callback
    
    Returns:
        The global DistillationManager instance
    """
    global _distillation_manager
    
    if _distillation_manager is None:
        _distillation_manager = DistillationManager(
            traces_path=traces_path,
            patterns_path=patterns_path,
            thinking_callback=thinking_callback,
        )
    elif thinking_callback is not None:
        _distillation_manager._emit = thinking_callback
    
    return _distillation_manager

