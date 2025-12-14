"""
Pattern analysis for TGR execution traces.

Extracts reasoning patterns from successful traces to inform
prompt enhancement and template improvement.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .trace_recorder import ExecutionTrace, NodeTrace


@dataclass
class ReasoningPattern:
    """A pattern extracted from successful execution traces."""
    
    pattern_id: str
    template_id: str
    node_sequence: List[str]  # Ordered node types that led to success
    key_phrases: List[str]  # Common phrases in successful outputs
    context_requirements: List[str]  # What context was needed
    success_rate: float
    sample_count: int
    avg_duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "template_id": self.template_id,
            "node_sequence": self.node_sequence,
            "key_phrases": self.key_phrases,
            "context_requirements": self.context_requirements,
            "success_rate": self.success_rate,
            "sample_count": self.sample_count,
            "avg_duration_ms": self.avg_duration_ms,
        }


@dataclass
class FailurePattern:
    """A pattern of common failures."""
    
    pattern_id: str
    template_id: str
    failure_type: str  # "timeout", "verification", "logic_error", etc.
    common_errors: List[str]
    affected_nodes: List[str]
    frequency: int
    suggested_fix: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "template_id": self.template_id,
            "failure_type": self.failure_type,
            "common_errors": self.common_errors,
            "affected_nodes": self.affected_nodes,
            "frequency": self.frequency,
            "suggested_fix": self.suggested_fix,
        }


class PatternAnalyzer:
    """
    Analyzes execution traces to extract reasoning patterns.
    
    Identifies:
    - Successful reasoning sequences
    - Key phrases correlated with success
    - Common failure modes
    - Context requirements for each template
    """
    
    # Common mathematical/logical phrases to look for
    LOGIC_PHRASES = [
        "therefore", "thus", "hence", "since", "because",
        "implies", "follows", "conclude", "deduce",
        "by definition", "by assumption", "given that",
    ]
    
    MATH_PHRASES = [
        "equals", "sum", "product", "factor", "divisor",
        "eigenvalue", "matrix", "vector", "group",
        "proof", "theorem", "lemma", "corollary",
    ]
    
    STRUCTURE_PHRASES = [
        "first", "second", "third", "next", "finally",
        "step 1", "step 2", "step 3",
        "case 1", "case 2", "case 3",
    ]
    
    def __init__(
        self,
        min_samples_for_pattern: int = 3,
        min_phrase_frequency: int = 2,
    ):
        """
        Initialize the pattern analyzer.
        
        Args:
            min_samples_for_pattern: Minimum traces needed to form a pattern
            min_phrase_frequency: Minimum times a phrase must appear to be included
        """
        self.min_samples_for_pattern = min_samples_for_pattern
        self.min_phrase_frequency = min_phrase_frequency
    
    def extract_patterns(
        self,
        traces: List[ExecutionTrace],
    ) -> List[ReasoningPattern]:
        """
        Analyze successful traces to extract reasoning patterns.
        
        Args:
            traces: List of execution traces (should be verified/successful)
        
        Returns:
            List of ReasoningPattern objects
        """
        if not traces:
            return []
        
        # Group traces by template
        by_template: Dict[str, List[ExecutionTrace]] = defaultdict(list)
        for trace in traces:
            if trace.verified:  # Only analyze verified traces
                by_template[trace.template_id].append(trace)
        
        patterns: List[ReasoningPattern] = []
        
        for template_id, template_traces in by_template.items():
            if len(template_traces) < self.min_samples_for_pattern:
                continue
            
            pattern = self._analyze_template_traces(template_id, template_traces)
            if pattern:
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_template_traces(
        self,
        template_id: str,
        traces: List[ExecutionTrace],
    ) -> Optional[ReasoningPattern]:
        """Analyze traces for a single template."""
        if not traces:
            return None
        
        # Extract common node sequences
        node_sequences = []
        all_outputs = []
        total_duration = 0.0
        context_keywords: Counter = Counter()
        
        for trace in traces:
            # Get node type sequence
            sequence = [n.node_type for n in trace.nodes if n.success]
            node_sequences.append(tuple(sequence))
            
            # Collect outputs
            for node in trace.nodes:
                if node.output and node.success:
                    all_outputs.append(node.output.lower())
                
                # Extract context keywords
                if node.context:
                    words = re.findall(r'\b\w{4,}\b', node.context.lower())
                    context_keywords.update(words)
            
            total_duration += trace.total_duration_ms
        
        # Find most common sequence
        sequence_counts = Counter(node_sequences)
        most_common_seq = sequence_counts.most_common(1)
        node_sequence = list(most_common_seq[0][0]) if most_common_seq else []
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(all_outputs)
        
        # Get context requirements (most common context keywords)
        context_requirements = [
            word for word, count in context_keywords.most_common(10)
            if count >= self.min_phrase_frequency
        ]
        
        # Calculate success rate
        success_count = sum(1 for t in traces if t.verified)
        success_rate = success_count / len(traces) if traces else 0.0
        
        return ReasoningPattern(
            pattern_id=f"pattern_{template_id}_{len(traces)}",
            template_id=template_id,
            node_sequence=node_sequence,
            key_phrases=key_phrases[:10],  # Top 10 phrases
            context_requirements=context_requirements[:5],  # Top 5 context words
            success_rate=success_rate,
            sample_count=len(traces),
            avg_duration_ms=total_duration / len(traces) if traces else 0.0,
        )
    
    def _extract_key_phrases(self, outputs: List[str]) -> List[str]:
        """Extract key phrases that appear frequently in successful outputs."""
        phrase_counts: Counter = Counter()
        
        all_phrases = self.LOGIC_PHRASES + self.MATH_PHRASES + self.STRUCTURE_PHRASES
        
        for output in outputs:
            output_lower = output.lower()
            for phrase in all_phrases:
                if phrase in output_lower:
                    phrase_counts[phrase] += 1
        
        # Also extract common n-grams
        for output in outputs:
            # Extract 2-3 word phrases
            words = re.findall(r'\b\w+\b', output.lower())
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) > 8:  # Skip very short phrases
                    phrase_counts[bigram] += 1
        
        # Return phrases that appear frequently enough
        return [
            phrase for phrase, count in phrase_counts.most_common(20)
            if count >= self.min_phrase_frequency
        ]
    
    def identify_failure_patterns(
        self,
        traces: List[ExecutionTrace],
    ) -> List[FailurePattern]:
        """
        Identify common failure modes from traces.
        
        Args:
            traces: List of execution traces (including failures)
        
        Returns:
            List of FailurePattern objects
        """
        # Group failed traces by template
        by_template: Dict[str, List[ExecutionTrace]] = defaultdict(list)
        for trace in traces:
            if not trace.verified:  # Only analyze failed traces
                by_template[trace.template_id].append(trace)
        
        patterns: List[FailurePattern] = []
        
        for template_id, template_traces in by_template.items():
            if len(template_traces) < 2:  # Need at least 2 failures to form pattern
                continue
            
            failure_patterns = self._analyze_failures(template_id, template_traces)
            patterns.extend(failure_patterns)
        
        return patterns
    
    def _analyze_failures(
        self,
        template_id: str,
        traces: List[ExecutionTrace],
    ) -> List[FailurePattern]:
        """Analyze failures for a single template."""
        patterns: List[FailurePattern] = []
        
        # Categorize failures by type
        failure_types: Dict[str, List[ExecutionTrace]] = defaultdict(list)
        
        for trace in traces:
            failure_type = self._classify_failure(trace)
            failure_types[failure_type].append(trace)
        
        for failure_type, type_traces in failure_types.items():
            if len(type_traces) < 2:
                continue
            
            # Find common errors and affected nodes
            error_counts: Counter = Counter()
            affected_nodes: Counter = Counter()
            
            for trace in type_traces:
                for node in trace.nodes:
                    if not node.success and node.error:
                        error_counts[node.error[:100]] += 1
                        affected_nodes[node.node_id] += 1
            
            common_errors = [err for err, _ in error_counts.most_common(3)]
            top_affected = [node for node, _ in affected_nodes.most_common(3)]
            
            suggested_fix = self._suggest_fix(failure_type, common_errors)
            
            patterns.append(FailurePattern(
                pattern_id=f"failure_{template_id}_{failure_type}",
                template_id=template_id,
                failure_type=failure_type,
                common_errors=common_errors,
                affected_nodes=top_affected,
                frequency=len(type_traces),
                suggested_fix=suggested_fix,
            ))
        
        return patterns
    
    def _classify_failure(self, trace: ExecutionTrace) -> str:
        """Classify the type of failure in a trace."""
        # Check for timeout
        for node in trace.nodes:
            if node.error and "timeout" in node.error.lower():
                return "timeout"
        
        # Check for verification failure
        for node in trace.nodes:
            if node.node_type == "verification" and not node.success:
                return "verification"
        
        # Check for calculation errors
        for node in trace.nodes:
            if node.node_type == "calculation" and not node.success:
                return "calculation"
        
        # Check for logic errors
        for node in trace.nodes:
            if node.role == "logic" and not node.success:
                return "logic_error"
        
        # Default
        return "unknown"
    
    def _suggest_fix(self, failure_type: str, errors: List[str]) -> Optional[str]:
        """Suggest a fix based on failure type."""
        suggestions = {
            "timeout": "Consider breaking into smaller steps or increasing timeout",
            "verification": "Add intermediate verification steps or use alternative calculation method",
            "calculation": "Try code-first approach (research role) or add more context",
            "logic_error": "Provide more specific instructions or add few-shot examples",
            "unknown": "Review trace for specific issues",
        }
        return suggestions.get(failure_type)
    
    def get_template_statistics(
        self,
        traces: List[ExecutionTrace],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get per-template statistics from traces.
        
        Returns:
            Dictionary mapping template_id to statistics
        """
        stats: Dict[str, Dict[str, Any]] = {}
        
        # Group by template
        by_template: Dict[str, List[ExecutionTrace]] = defaultdict(list)
        for trace in traces:
            by_template[trace.template_id].append(trace)
        
        for template_id, template_traces in by_template.items():
            verified = sum(1 for t in template_traces if t.verified)
            total = len(template_traces)
            durations = [t.total_duration_ms for t in template_traces]
            
            stats[template_id] = {
                "total_traces": total,
                "verified_count": verified,
                "success_rate": verified / total if total > 0 else 0.0,
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0.0,
                "min_duration_ms": min(durations) if durations else 0.0,
                "max_duration_ms": max(durations) if durations else 0.0,
            }
        
        return stats
    
    def find_similar_successful_traces(
        self,
        problem: str,
        traces: List[ExecutionTrace],
        max_results: int = 5,
    ) -> List[ExecutionTrace]:
        """
        Find successful traces with similar problems.
        
        Uses simple keyword matching for similarity.
        
        Args:
            problem: The problem to find similar traces for
            traces: Pool of traces to search
            max_results: Maximum number of results
        
        Returns:
            List of similar successful traces
        """
        problem_words = set(re.findall(r'\b\w{4,}\b', problem.lower()))
        
        scored_traces: List[Tuple[float, ExecutionTrace]] = []
        
        for trace in traces:
            if not trace.verified:
                continue
            
            trace_words = set(re.findall(r'\b\w{4,}\b', trace.problem.lower()))
            
            # Jaccard similarity
            intersection = len(problem_words & trace_words)
            union = len(problem_words | trace_words)
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity > 0.1:  # Minimum threshold
                scored_traces.append((similarity, trace))
        
        # Sort by similarity and return top results
        scored_traces.sort(key=lambda x: x[0], reverse=True)
        return [trace for _, trace in scored_traces[:max_results]]

