"""
Node output verification for TGR backtracking.

Provides type-specific validation of node outputs to determine
if backtracking/retry is needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..infra.openrouter.client import OpenRouterClient


@dataclass
class VerificationResult:
    """Result of node output verification."""
    
    passed: bool
    confidence: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    suggested_fix: Optional[str] = None
    
    def __bool__(self) -> bool:
        return self.passed


# Patterns for detecting common output issues
_EMPTY_PATTERNS = [
    r"^\s*$",  # Empty or whitespace only
    r"^\s*\[no\s+response",  # No response marker
    r"^\s*\[error",  # Error marker
    r"^\s*\[timeout",  # Timeout marker
    r"^\s*\[research\s+timed\s+out",  # Research timeout
]

_DEFINITION_KEYWORDS = [
    "define", "defined", "definition", "let", "denote", "consider",
    "given", "suppose", "assume", "notation", "concept", "means",
]

_ENUMERATION_PATTERNS = [
    r"\d+\.",  # Numbered list
    r"[-•*]",  # Bullet points
    r"\[\d+\]",  # Bracketed numbers
    r"first.*second",  # Sequential words
    r",\s*and\s*",  # List with 'and'
]

_CALCULATION_PATTERNS = [
    r"####\s*[+-]?\d+",  # Explicit numeric marker
    r"=\s*[+-]?\d+(?:\.\d+)?",  # Equation result
    r"result\s*[:=]\s*[+-]?\d+",  # Result marker
    r"answer\s*[:=]\s*[+-]?\d+",  # Answer marker
    r"total\s*[:=]\s*[+-]?\d+",  # Total marker
    r"count\s*[:=]\s*[+-]?\d+",  # Count marker
]

_AGGREGATION_KEYWORDS = [
    "therefore", "thus", "hence", "in summary", "combining",
    "overall", "conclusion", "final", "together", "aggregate",
]


def _default_emit(stage: str, content: str) -> None:
    """Default no-op emit function."""
    pass


class NodeVerifier:
    """
    Verifies node outputs based on node type.
    
    Provides type-specific validation to determine if a node's output
    is acceptable or if backtracking/retry is needed.
    """
    
    # Minimum lengths for different node types
    MIN_LENGTHS = {
        "definition": 50,
        "enumeration": 30,
        "calculation": 10,
        "aggregation": 40,
        "verification": 5,
        "retrieval": 20,
    }
    
    # Confidence thresholds for passing
    PASS_THRESHOLD = 0.5
    
    def __init__(
        self,
        client: Optional["OpenRouterClient"] = None,
        model_name: Optional[str] = None,
        thinking_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize the node verifier.
        
        Args:
            client: Optional LLM client for semantic verification
            model_name: Model to use for semantic checks
            thinking_callback: Callback for logging
        """
        self.client = client
        self.model_name = model_name
        self._emit = thinking_callback or _default_emit
    
    def verify_node_output(
        self,
        node_id: str,
        node_type: str,
        role: str,
        output: str,
        context: str = "",
        instruction: str = "",
        expected_patterns: Optional[List[str]] = None,
    ) -> VerificationResult:
        """
        Verify a node's output based on its type.
        
        Args:
            node_id: The node identifier
            node_type: Type of node (definition, enumeration, calculation, etc.)
            role: Role of node (logic, math, research, etc.)
            output: The node's output to verify
            context: Context that was provided to the node
            instruction: The node's instruction
            expected_patterns: Optional list of regex patterns expected in output
        
        Returns:
            VerificationResult with pass/fail status and details
        """
        issues: List[str] = []
        confidence = 1.0
        
        # Check for empty or error outputs
        if self._is_empty_or_error(output):
            return VerificationResult(
                passed=False,
                confidence=0.0,
                issues=["Output is empty or contains error marker"],
                suggested_fix="retry_same",
            )
        
        # Check minimum length
        min_len = self.MIN_LENGTHS.get(node_type, 20)
        if len(output.strip()) < min_len:
            issues.append(f"Output too short ({len(output.strip())} < {min_len})")
            confidence -= 0.3
        
        # Type-specific validation
        type_result = self._verify_by_type(node_type, role, output, instruction)
        confidence *= type_result.confidence
        issues.extend(type_result.issues)
        
        # Check expected patterns if provided
        if expected_patterns:
            pattern_matches = sum(
                1 for p in expected_patterns
                if re.search(p, output, re.IGNORECASE)
            )
            pattern_ratio = pattern_matches / len(expected_patterns)
            if pattern_ratio < 0.5:
                issues.append(f"Only {pattern_matches}/{len(expected_patterns)} expected patterns found")
                confidence *= pattern_ratio + 0.3  # Penalize but don't zero out
        
        passed = confidence >= self.PASS_THRESHOLD and not type_result.suggested_fix
        
        result = VerificationResult(
            passed=passed,
            confidence=confidence,
            issues=issues,
            suggested_fix=type_result.suggested_fix if not passed else None,
        )
        
        self._emit(
            "node_verify",
            f"{node_id}: passed={passed}, confidence={confidence:.2f}, issues={len(issues)}"
        )
        
        return result
    
    def _is_empty_or_error(self, output: str) -> bool:
        """Check if output is empty or contains error markers."""
        for pattern in _EMPTY_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return True
        return False
    
    def _verify_by_type(
        self,
        node_type: str,
        role: str,
        output: str,
        instruction: str,
    ) -> VerificationResult:
        """
        Apply type-specific verification rules.
        
        Returns:
            VerificationResult with type-specific validation
        """
        output_lower = output.lower()
        
        if node_type == "definition":
            return self._verify_definition(output_lower, instruction)
        elif node_type == "enumeration":
            return self._verify_enumeration(output_lower)
        elif node_type == "calculation":
            return self._verify_calculation(output, instruction)
        elif node_type == "aggregation":
            return self._verify_aggregation(output_lower)
        elif node_type == "verification":
            return self._verify_verification(output)
        elif node_type == "retrieval":
            return self._verify_retrieval(output)
        else:
            # Default verification based on role
            return self._verify_by_role(role, output, instruction)
    
    def _verify_definition(self, output_lower: str, instruction: str) -> VerificationResult:
        """Verify definition node output."""
        issues: List[str] = []
        confidence = 1.0
        
        # Check for definition keywords
        has_definition_keyword = any(kw in output_lower for kw in _DEFINITION_KEYWORDS)
        if not has_definition_keyword:
            issues.append("Missing definition keywords (define, let, denote, etc.)")
            confidence -= 0.2
        
        # Check for mathematical notation if instruction suggests math
        if any(term in instruction.lower() for term in ["group", "matrix", "function", "set"]):
            has_math = any(c in output_lower for c in ["=", "∈", "⊆", "→", "×"])
            if not has_math:
                # Also check for text equivalents
                has_math = any(term in output_lower for term in ["equals", "element", "subset", "maps"])
            if not has_math:
                issues.append("Mathematical definition may lack formal notation")
                confidence -= 0.1
        
        return VerificationResult(passed=confidence > 0.5, confidence=confidence, issues=issues)
    
    def _verify_enumeration(self, output_lower: str) -> VerificationResult:
        """Verify enumeration node output."""
        issues: List[str] = []
        confidence = 1.0
        
        # Check for list structure
        has_list = any(re.search(p, output_lower) for p in _ENUMERATION_PATTERNS)
        if not has_list:
            issues.append("Output lacks enumeration structure (numbered list, bullets, etc.)")
            confidence -= 0.3
        
        # Check for multiple items
        lines = [l.strip() for l in output_lower.split("\n") if l.strip()]
        if len(lines) < 2:
            issues.append("Enumeration has too few items")
            confidence -= 0.2
        
        return VerificationResult(passed=confidence > 0.5, confidence=confidence, issues=issues)
    
    def _verify_calculation(self, output: str, instruction: str) -> VerificationResult:
        """Verify calculation node output."""
        issues: List[str] = []
        confidence = 1.0
        suggested_fix = None
        
        # Check for numeric result
        has_numeric = any(re.search(p, output, re.IGNORECASE) for p in _CALCULATION_PATTERNS)
        
        # Also check for standalone numbers
        if not has_numeric:
            numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", output)
            has_numeric = len(numbers) > 0
        
        if not has_numeric and "####" in instruction:
            issues.append("Calculation missing numeric result (expected #### marker)")
            confidence = 0.2
            suggested_fix = "adjust_params"
        elif not has_numeric:
            issues.append("Calculation may be missing numeric result")
            confidence -= 0.3
        
        # Check for code execution errors
        if "error" in output.lower() or "exception" in output.lower():
            issues.append("Calculation contains error indicators")
            confidence -= 0.4
            suggested_fix = "alternative_approach"
        
        return VerificationResult(
            passed=confidence > 0.5,
            confidence=confidence,
            issues=issues,
            suggested_fix=suggested_fix,
        )
    
    def _verify_aggregation(self, output_lower: str) -> VerificationResult:
        """Verify aggregation node output."""
        issues: List[str] = []
        confidence = 1.0
        
        # Check for synthesis keywords
        has_synthesis = any(kw in output_lower for kw in _AGGREGATION_KEYWORDS)
        if not has_synthesis:
            issues.append("Aggregation may lack synthesis language (therefore, thus, combining, etc.)")
            confidence -= 0.2
        
        # Check for reference to prior results
        has_reference = any(
            marker in output_lower
            for marker in ["from above", "previous", "earlier", "based on", "given that"]
        )
        if not has_reference:
            issues.append("Aggregation may not reference prior results")
            confidence -= 0.1
        
        return VerificationResult(passed=confidence > 0.5, confidence=confidence, issues=issues)
    
    def _verify_verification(self, output: str) -> VerificationResult:
        """Verify verification node output."""
        issues: List[str] = []
        confidence = 1.0
        
        # Verification should produce a clear result
        output_stripped = output.strip()
        
        # Check for numeric output
        numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", output_stripped)
        if not numbers:
            issues.append("Verification output lacks numeric value")
            confidence -= 0.3
        
        # If output is just a number, that's ideal
        try:
            float(output_stripped)
            confidence = 1.0  # Pure numeric output is perfect for verification
        except ValueError:
            pass
        
        return VerificationResult(passed=confidence > 0.5, confidence=confidence, issues=issues)
    
    def _verify_retrieval(self, output: str) -> VerificationResult:
        """Verify retrieval node output."""
        issues: List[str] = []
        confidence = 1.0
        
        # Check for retrieved content markers
        has_content = "[" in output and "]" in output
        if not has_content:
            issues.append("Retrieval output lacks document markers")
            confidence -= 0.2
        
        # Check for "no results" or "no documents"
        if "no relevant" in output.lower() or "no documents" in output.lower():
            issues.append("Retrieval found no relevant documents")
            confidence -= 0.3
        
        return VerificationResult(passed=confidence > 0.5, confidence=confidence, issues=issues)
    
    def _verify_by_role(self, role: str, output: str, instruction: str) -> VerificationResult:
        """Fallback verification based on role."""
        issues: List[str] = []
        confidence = 1.0
        
        if role == "math":
            # Check for mathematical content
            has_math = any(c in output for c in ["=", "+", "-", "*", "/", "^"])
            has_numbers = bool(re.search(r"\d+", output))
            if not has_math and not has_numbers:
                issues.append("Math role output lacks mathematical content")
                confidence -= 0.2
        
        elif role == "logic":
            # Check for logical reasoning
            logic_words = ["therefore", "because", "since", "if", "then", "implies"]
            has_logic = any(word in output.lower() for word in logic_words)
            if not has_logic:
                issues.append("Logic role output may lack explicit reasoning")
                confidence -= 0.1
        
        elif role == "research":
            # Check for code or structured output
            has_code = "```" in output or "def " in output or "import " in output
            if not has_code:
                issues.append("Research role output may lack code")
                confidence -= 0.1
        
        return VerificationResult(passed=confidence > 0.5, confidence=confidence, issues=issues)
    
    def verify_with_llm(
        self,
        node_id: str,
        node_type: str,
        output: str,
        instruction: str,
        context: str = "",
    ) -> VerificationResult:
        """
        Use LLM to semantically verify node output.
        
        This is a more expensive but more accurate verification method.
        Use sparingly, e.g., for high-stakes calculations.
        
        Args:
            node_id: Node identifier
            node_type: Type of node
            output: Output to verify
            instruction: Original instruction
            context: Context provided to node
        
        Returns:
            VerificationResult with LLM-based assessment
        """
        if not self.client or not self.model_name:
            return VerificationResult(
                passed=True,
                confidence=0.5,
                issues=["LLM verification unavailable"],
            )
        
        prompt = f"""Verify if this output correctly addresses the instruction.

Instruction: {instruction[:500]}

Output: {output[:1000]}

Evaluate:
1. Does the output address the instruction?
2. Is the output logically coherent?
3. Are there any obvious errors?

Respond with a JSON object:
{{"passed": true/false, "confidence": 0.0-1.0, "issues": ["issue1", "issue2"]}}
"""
        
        try:
            response = self.client.complete_chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.1,
                max_tokens=200,
            )
            
            # Parse response
            import json
            text = response.text.strip()
            # Extract JSON from response
            if "{" in text:
                json_str = text[text.index("{"):text.rindex("}") + 1]
                result = json.loads(json_str)
                return VerificationResult(
                    passed=result.get("passed", True),
                    confidence=result.get("confidence", 0.7),
                    issues=result.get("issues", []),
                )
        except Exception as e:
            self._emit("llm_verify_error", f"LLM verification failed: {str(e)[:100]}")
        
        # Fallback to basic verification
        return self.verify_node_output(node_id, node_type, "", output, context, instruction)


def quick_verify(output: str, node_type: str = "logic") -> bool:
    """
    Quick check if output is acceptable.
    
    Convenience function for simple verification needs.
    """
    verifier = NodeVerifier()
    result = verifier.verify_node_output(
        node_id="quick",
        node_type=node_type,
        role="logic",
        output=output,
    )
    return result.passed

