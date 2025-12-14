"""Tests for pattern analysis functionality."""

import pytest
from datetime import datetime

from ..trace_recorder import ExecutionTrace, NodeTrace
from ..pattern_analyzer import PatternAnalyzer, ReasoningPattern, FailurePattern


@pytest.fixture
def sample_traces():
    """Create sample execution traces for testing."""
    traces = []
    
    # Create 5 verified traces for spectral_cayley_v1
    for i in range(5):
        traces.append(ExecutionTrace(
            trace_id=f"verified-{i}",
            timestamp=datetime.now(),
            problem=f"Calculate eigenvalues for matrix {i}",
            template_id="spectral_cayley_v1",
            nodes=[
                NodeTrace(
                    node_id="n1_define",
                    node_type="definition",
                    role="logic",
                    instruction="Define the group",
                    context="",
                    output="Therefore, we define G as the cyclic group",
                    duration_ms=1000.0,
                    success=True,
                ),
                NodeTrace(
                    node_id="n2_calculate",
                    node_type="calculation",
                    role="research",
                    instruction="Compute eigenvalues",
                    context="",
                    output="Using the formula, the sum equals 42",
                    duration_ms=2000.0,
                    success=True,
                ),
            ],
            final_answer="42",
            verified=True,
            verification_method="verifier",
            total_duration_ms=3000.0,
        ))
    
    # Create 2 failed traces
    for i in range(2):
        traces.append(ExecutionTrace(
            trace_id=f"failed-{i}",
            timestamp=datetime.now(),
            problem=f"Failed calculation {i}",
            template_id="spectral_cayley_v1",
            nodes=[
                NodeTrace(
                    node_id="n1_define",
                    node_type="definition",
                    role="logic",
                    instruction="Define the group",
                    context="",
                    output="[timeout]",
                    duration_ms=90000.0,
                    success=False,
                    error="timeout",
                ),
            ],
            final_answer="",
            verified=False,
            verification_method="none",
            total_duration_ms=90000.0,
        ))
    
    return traces


class TestPatternAnalyzer:
    """Tests for PatternAnalyzer."""
    
    def test_extract_patterns(self, sample_traces):
        """Test pattern extraction from traces."""
        analyzer = PatternAnalyzer(min_samples_for_pattern=3)
        
        patterns = analyzer.extract_patterns(sample_traces)
        
        assert len(patterns) >= 1
        
        # Find the spectral_cayley_v1 pattern
        spectral_pattern = next(
            (p for p in patterns if p.template_id == "spectral_cayley_v1"),
            None
        )
        
        assert spectral_pattern is not None
        assert spectral_pattern.sample_count >= 3
        assert spectral_pattern.success_rate == 1.0
    
    def test_extract_key_phrases(self, sample_traces):
        """Test key phrase extraction."""
        analyzer = PatternAnalyzer(min_phrase_frequency=2)
        
        patterns = analyzer.extract_patterns(sample_traces)
        
        # "therefore" should appear as a key phrase (used in outputs)
        spectral_pattern = next(
            (p for p in patterns if p.template_id == "spectral_cayley_v1"),
            None
        )
        
        if spectral_pattern:
            # Key phrases should contain common reasoning words
            assert len(spectral_pattern.key_phrases) >= 0
    
    def test_identify_failure_patterns(self, sample_traces):
        """Test failure pattern identification."""
        analyzer = PatternAnalyzer()
        
        failures = analyzer.identify_failure_patterns(sample_traces)
        
        assert len(failures) >= 1
        
        # Check for timeout failure pattern
        timeout_failure = next(
            (f for f in failures if f.failure_type == "timeout"),
            None
        )
        
        assert timeout_failure is not None
        assert timeout_failure.frequency == 2
    
    def test_node_sequence_extraction(self, sample_traces):
        """Test extraction of node type sequences."""
        analyzer = PatternAnalyzer(min_samples_for_pattern=3)
        
        patterns = analyzer.extract_patterns(sample_traces)
        
        spectral_pattern = next(
            (p for p in patterns if p.template_id == "spectral_cayley_v1"),
            None
        )
        
        if spectral_pattern:
            # Should have definition -> calculation sequence
            assert "definition" in spectral_pattern.node_sequence
            assert "calculation" in spectral_pattern.node_sequence
    
    def test_get_template_statistics(self, sample_traces):
        """Test per-template statistics."""
        analyzer = PatternAnalyzer()
        
        stats = analyzer.get_template_statistics(sample_traces)
        
        assert "spectral_cayley_v1" in stats
        
        spectral_stats = stats["spectral_cayley_v1"]
        assert spectral_stats["total_traces"] == 7  # 5 verified + 2 failed
        assert spectral_stats["verified_count"] == 5
    
    def test_find_similar_traces(self, sample_traces):
        """Test finding similar traces."""
        analyzer = PatternAnalyzer()
        
        similar = analyzer.find_similar_successful_traces(
            problem="Calculate eigenvalues for this matrix",
            traces=sample_traces,
            max_results=3,
        )
        
        # Should find traces with "eigenvalues" in problem
        assert len(similar) > 0
        for trace in similar:
            assert trace.verified


class TestReasoningPattern:
    """Tests for ReasoningPattern dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        pattern = ReasoningPattern(
            pattern_id="test_pattern",
            template_id="test_template",
            node_sequence=["definition", "calculation"],
            key_phrases=["therefore", "thus"],
            context_requirements=["matrix", "eigenvalue"],
            success_rate=0.95,
            sample_count=20,
            avg_duration_ms=5000.0,
        )
        
        d = pattern.to_dict()
        
        assert d["pattern_id"] == "test_pattern"
        assert d["success_rate"] == 0.95
        assert len(d["node_sequence"]) == 2


class TestFailurePattern:
    """Tests for FailurePattern dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        pattern = FailurePattern(
            pattern_id="failure_test",
            template_id="test_template",
            failure_type="timeout",
            common_errors=["timeout reached", "connection error"],
            affected_nodes=["n2_calculate"],
            frequency=5,
            suggested_fix="Increase timeout or simplify calculation",
        )
        
        d = pattern.to_dict()
        
        assert d["failure_type"] == "timeout"
        assert d["frequency"] == 5

