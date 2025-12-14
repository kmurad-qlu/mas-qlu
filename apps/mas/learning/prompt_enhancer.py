"""
Prompt enhancement for TGR using learned patterns.

Uses patterns extracted from successful traces to improve
node instructions and provide few-shot examples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .trace_recorder import ExecutionTrace, NodeTrace
from .pattern_analyzer import ReasoningPattern, FailurePattern


@dataclass
class EnhancedInstruction:
    """An enhanced node instruction with improvements."""
    
    original: str
    enhanced: str
    few_shot_examples: List[str]
    guardrails: List[str]
    source_pattern_id: Optional[str] = None


class PromptEnhancer:
    """
    Enhances prompts using patterns from successful traces.
    
    Provides:
    - Few-shot examples from successful executions
    - Guardrails against common failure modes
    - Key phrase injection for better reasoning
    """
    
    def __init__(
        self,
        patterns: Optional[List[ReasoningPattern]] = None,
        failure_patterns: Optional[List[FailurePattern]] = None,
        max_examples_per_type: int = 2,
        max_guardrails: int = 3,
    ):
        """
        Initialize the prompt enhancer.
        
        Args:
            patterns: Success patterns to use for enhancement
            failure_patterns: Failure patterns to use for guardrails
            max_examples_per_type: Maximum few-shot examples per node type
            max_guardrails: Maximum guardrail statements to add
        """
        self.patterns = patterns or []
        self.failure_patterns = failure_patterns or []
        self.max_examples_per_type = max_examples_per_type
        self.max_guardrails = max_guardrails
        
        # Index patterns by template
        self._patterns_by_template: Dict[str, List[ReasoningPattern]] = {}
        for p in self.patterns:
            if p.template_id not in self._patterns_by_template:
                self._patterns_by_template[p.template_id] = []
            self._patterns_by_template[p.template_id].append(p)
        
        # Index failure patterns by template
        self._failures_by_template: Dict[str, List[FailurePattern]] = {}
        for f in self.failure_patterns:
            if f.template_id not in self._failures_by_template:
                self._failures_by_template[f.template_id] = []
            self._failures_by_template[f.template_id].append(f)
    
    def update_patterns(
        self,
        patterns: List[ReasoningPattern],
        failure_patterns: Optional[List[FailurePattern]] = None,
    ) -> None:
        """Update the patterns used for enhancement."""
        self.patterns = patterns
        if failure_patterns is not None:
            self.failure_patterns = failure_patterns
        
        # Rebuild indexes
        self._patterns_by_template = {}
        for p in self.patterns:
            if p.template_id not in self._patterns_by_template:
                self._patterns_by_template[p.template_id] = []
            self._patterns_by_template[p.template_id].append(p)
        
        self._failures_by_template = {}
        for f in self.failure_patterns:
            if f.template_id not in self._failures_by_template:
                self._failures_by_template[f.template_id] = []
            self._failures_by_template[f.template_id].append(f)
    
    def enhance_instruction(
        self,
        instruction: str,
        node_type: str,
        node_role: str,
        template_id: str,
        traces: Optional[List[ExecutionTrace]] = None,
    ) -> EnhancedInstruction:
        """
        Enhance a node instruction using learned patterns.
        
        Args:
            instruction: Original node instruction
            node_type: Type of the node
            node_role: Role of the node
            template_id: ID of the template
            traces: Optional list of traces for few-shot examples
        
        Returns:
            EnhancedInstruction with improvements
        """
        enhanced_parts = [instruction]
        few_shot_examples: List[str] = []
        guardrails: List[str] = []
        source_pattern_id = None
        
        # Get relevant patterns
        template_patterns = self._patterns_by_template.get(template_id, [])
        
        # Add key phrases from patterns
        if template_patterns:
            best_pattern = max(template_patterns, key=lambda p: p.success_rate)
            source_pattern_id = best_pattern.pattern_id
            
            if best_pattern.key_phrases:
                phrase_hint = self._format_phrase_hint(best_pattern.key_phrases[:5])
                if phrase_hint:
                    enhanced_parts.append(phrase_hint)
        
        # Add few-shot examples from traces
        if traces:
            examples = self.generate_few_shot_examples(
                template_id=template_id,
                node_type=node_type,
                traces=traces,
                k=self.max_examples_per_type,
            )
            few_shot_examples.extend(examples)
            
            if examples:
                enhanced_parts.append("\n\nExamples of successful reasoning:")
                for i, ex in enumerate(examples, 1):
                    enhanced_parts.append(f"\nExample {i}:\n{ex}")
        
        # Add guardrails from failure patterns
        template_failures = self._failures_by_template.get(template_id, [])
        if template_failures:
            guardrails = self._generate_guardrails(template_failures, node_type)
            if guardrails:
                enhanced_parts.append("\n\nIMPORTANT - Avoid these pitfalls:")
                for guardrail in guardrails[:self.max_guardrails]:
                    enhanced_parts.append(f"- {guardrail}")
        
        enhanced = "\n".join(enhanced_parts)
        
        return EnhancedInstruction(
            original=instruction,
            enhanced=enhanced,
            few_shot_examples=few_shot_examples,
            guardrails=guardrails,
            source_pattern_id=source_pattern_id,
        )
    
    def _format_phrase_hint(self, phrases: List[str]) -> str:
        """Format key phrases as a hint."""
        if not phrases:
            return ""
        
        phrase_list = ", ".join(f'"{p}"' for p in phrases[:5])
        return f"\nTip: Successful solutions often use reasoning like: {phrase_list}"
    
    def _generate_guardrails(
        self,
        failures: List[FailurePattern],
        node_type: str,
    ) -> List[str]:
        """Generate guardrail statements from failure patterns."""
        guardrails: List[str] = []
        
        for failure in failures:
            if failure.suggested_fix:
                guardrails.append(failure.suggested_fix)
            
            # Generate specific warnings based on failure type
            if failure.failure_type == "timeout":
                guardrails.append("Be efficient - avoid unnecessary computation")
            elif failure.failure_type == "verification":
                guardrails.append("Double-check numeric results before final output")
            elif failure.failure_type == "calculation":
                guardrails.append("Show intermediate steps and verify each calculation")
            elif failure.failure_type == "logic_error":
                guardrails.append("Clearly state assumptions and check logical consistency")
        
        # Deduplicate
        seen = set()
        unique_guardrails = []
        for g in guardrails:
            if g not in seen:
                seen.add(g)
                unique_guardrails.append(g)
        
        return unique_guardrails
    
    def generate_few_shot_examples(
        self,
        template_id: str,
        node_type: str,
        traces: List[ExecutionTrace],
        k: int = 2,
    ) -> List[str]:
        """
        Extract successful input/output pairs for few-shot prompting.
        
        Args:
            template_id: Template to find examples for
            node_type: Node type to find examples for
            traces: Pool of traces to search
            k: Number of examples to return
        
        Returns:
            List of formatted example strings
        """
        examples: List[str] = []
        
        # Filter to verified traces with matching template
        matching_traces = [
            t for t in traces
            if t.verified and t.template_id == template_id
        ]
        
        # Sort by success rate / quality indicators
        matching_traces.sort(key=lambda t: t.success_rate, reverse=True)
        
        for trace in matching_traces:
            if len(examples) >= k:
                break
            
            # Find nodes of the specified type
            for node in trace.nodes:
                if len(examples) >= k:
                    break
                
                if node.node_type == node_type and node.success and node.output:
                    # Format as example
                    example = self._format_example(node, trace.problem)
                    if example and len(example) > 50:  # Skip too-short examples
                        examples.append(example)
        
        return examples
    
    def _format_example(self, node: NodeTrace, problem: str) -> str:
        """Format a node trace as a few-shot example."""
        # Truncate for reasonable length
        instruction = node.instruction[:200] if node.instruction else ""
        output = node.output[:400] if node.output else ""
        
        if not instruction or not output:
            return ""
        
        return f"Input: {instruction}\nOutput: {output}"
    
    def enhance_knowledge_seeds(
        self,
        seeds: List[str],
        template_id: str,
    ) -> List[str]:
        """
        Enhance knowledge seeds with learned context requirements.
        
        Args:
            seeds: Original knowledge seeds
            template_id: Template ID
        
        Returns:
            Enhanced list of knowledge seeds
        """
        enhanced = list(seeds)
        
        # Get patterns for this template
        template_patterns = self._patterns_by_template.get(template_id, [])
        
        if template_patterns:
            best_pattern = max(template_patterns, key=lambda p: p.success_rate)
            
            # Add context requirements as seeds if not already present
            for req in best_pattern.context_requirements[:3]:
                # Check if already covered
                if not any(req.lower() in s.lower() for s in seeds):
                    enhanced.append(f"Context hint: {req} may be relevant")
        
        return enhanced
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """Get statistics about available enhancements."""
        return {
            "total_patterns": len(self.patterns),
            "total_failure_patterns": len(self.failure_patterns),
            "templates_with_patterns": len(self._patterns_by_template),
            "templates_with_failures": len(self._failures_by_template),
            "avg_success_rate": (
                sum(p.success_rate for p in self.patterns) / len(self.patterns)
                if self.patterns else 0.0
            ),
            "avg_sample_count": (
                sum(p.sample_count for p in self.patterns) / len(self.patterns)
                if self.patterns else 0.0
            ),
        }


def create_enhanced_instruction(
    instruction: str,
    node_type: str,
    template_id: str,
    patterns: List[ReasoningPattern],
) -> str:
    """
    Convenience function to create an enhanced instruction.
    
    Args:
        instruction: Original instruction
        node_type: Type of node
        template_id: Template ID
        patterns: Patterns to use
    
    Returns:
        Enhanced instruction string
    """
    enhancer = PromptEnhancer(patterns=patterns)
    result = enhancer.enhance_instruction(
        instruction=instruction,
        node_type=node_type,
        node_role="logic",
        template_id=template_id,
    )
    return result.enhanced

