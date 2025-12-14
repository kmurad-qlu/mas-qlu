"""
Learning module for TGR distillation loop.

This module provides:
- TraceRecorder: Capture execution traces from TGR runs
- TraceStore: Persist and query traces
- PatternAnalyzer: Extract reasoning patterns from successful traces
- PromptEnhancer: Augment prompts with learned patterns
- DistillationManager: Coordinates the complete distillation loop
"""

from .trace_recorder import (
    ExecutionTrace,
    NodeTrace,
    TraceRecorder,
)
from .trace_store import TraceStore
from .pattern_analyzer import (
    PatternAnalyzer,
    ReasoningPattern,
    FailurePattern,
)
from .prompt_enhancer import (
    PromptEnhancer,
    EnhancedInstruction,
    create_enhanced_instruction,
)
from .distillation_manager import (
    DistillationManager,
    get_distillation_manager,
)

__all__ = [
    "ExecutionTrace",
    "NodeTrace",
    "TraceRecorder",
    "TraceStore",
    "PatternAnalyzer",
    "ReasoningPattern",
    "FailurePattern",
    "PromptEnhancer",
    "EnhancedInstruction",
    "create_enhanced_instruction",
    "DistillationManager",
    "get_distillation_manager",
]

