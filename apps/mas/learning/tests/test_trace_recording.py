"""Tests for trace recording functionality."""

import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from ..trace_recorder import (
    ExecutionTrace,
    NodeTrace,
    TraceRecorder,
)
from ..trace_store import TraceStore


@pytest.fixture
def sample_node_traces():
    """Create sample node traces for testing."""
    return [
        NodeTrace(
            node_id="n1_define",
            node_type="definition",
            role="logic",
            instruction="Define the problem setup",
            context="Problem context here",
            output="This problem involves eigenvalue computation",
            duration_ms=1500.0,
            success=True,
        ),
        NodeTrace(
            node_id="n2_calculate",
            node_type="calculation",
            role="research",
            instruction="Compute the eigenvalues",
            context="Definition from n1",
            output="Eigenvalues: [1, 2, 3, 4]",
            duration_ms=5000.0,
            success=True,
        ),
        NodeTrace(
            node_id="n3_verify",
            node_type="verification",
            role="verifier",
            instruction="Verify the result",
            context="Calculation from n2",
            output="4",
            duration_ms=1000.0,
            success=True,
        ),
    ]


@pytest.fixture
def sample_trace(sample_node_traces):
    """Create a sample execution trace."""
    return ExecutionTrace(
        trace_id="test-trace-001",
        timestamp=datetime.now(),
        problem="What are the eigenvalues of this matrix?",
        template_id="spectral_cayley_v1",
        nodes=sample_node_traces,
        final_answer="4",
        verified=True,
        verification_method="verifier",
        total_duration_ms=7500.0,
        metadata={"rag_enabled": True},
    )


class TestNodeTrace:
    """Tests for NodeTrace dataclass."""
    
    def test_to_dict(self, sample_node_traces):
        """Test conversion to dictionary."""
        node = sample_node_traces[0]
        d = node.to_dict()
        
        assert d["node_id"] == "n1_define"
        assert d["node_type"] == "definition"
        assert d["success"] is True
        assert d["duration_ms"] == 1500.0
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "node_id": "test_node",
            "node_type": "calculation",
            "role": "research",
            "instruction": "Do math",
            "context": "Context",
            "output": "42",
            "duration_ms": 100.0,
            "success": True,
            "error": None,
            "retry_count": 0,
        }
        node = NodeTrace.from_dict(d)
        
        assert node.node_id == "test_node"
        assert node.node_type == "calculation"
        assert node.output == "42"


class TestExecutionTrace:
    """Tests for ExecutionTrace dataclass."""
    
    def test_to_dict(self, sample_trace):
        """Test conversion to dictionary."""
        d = sample_trace.to_dict()
        
        assert d["trace_id"] == "test-trace-001"
        assert d["template_id"] == "spectral_cayley_v1"
        assert d["verified"] is True
        assert len(d["nodes"]) == 3
    
    def test_from_dict(self, sample_trace):
        """Test creation from dictionary."""
        d = sample_trace.to_dict()
        restored = ExecutionTrace.from_dict(d)
        
        assert restored.trace_id == sample_trace.trace_id
        assert restored.template_id == sample_trace.template_id
        assert len(restored.nodes) == len(sample_trace.nodes)
    
    def test_problem_hash(self, sample_trace):
        """Test problem hash generation."""
        hash1 = sample_trace.problem_hash()
        
        # Same problem should give same hash
        sample_trace.problem = "  What are the eigenvalues of this matrix?  "
        hash2 = sample_trace.problem_hash()
        
        # Different case should also give same hash (normalized)
        sample_trace.problem = "WHAT ARE THE EIGENVALUES OF THIS MATRIX?"
        hash3 = sample_trace.problem_hash()
        
        assert hash1 == hash2
        assert hash2 == hash3
    
    def test_success_rate(self, sample_trace):
        """Test success rate calculation."""
        assert sample_trace.success_rate == 1.0
        
        # Add a failed node
        sample_trace.nodes.append(NodeTrace(
            node_id="n4_failed",
            node_type="calculation",
            role="math",
            instruction="This fails",
            context="",
            output="[error]",
            duration_ms=100.0,
            success=False,
            error="Test error",
        ))
        
        assert sample_trace.success_rate == 0.75
    
    def test_had_retries(self, sample_trace):
        """Test retry detection."""
        assert sample_trace.had_retries is False
        
        sample_trace.nodes[0].retry_count = 1
        assert sample_trace.had_retries is True


class TestTraceRecorder:
    """Tests for TraceRecorder."""
    
    def test_basic_recording(self):
        """Test basic trace recording flow."""
        recorder = TraceRecorder()
        recorder.start_trace()
        
        recorder.start_node("n1")
        recorder.record_node(
            node_id="n1",
            node_type="definition",
            role="logic",
            instruction="Define it",
            context="",
            output="Defined",
            success=True,
        )
        
        trace = recorder.finalize_trace(
            problem="Test problem",
            template_id="test_template",
            final_answer="42",
            verified=True,
            verification_method="verifier",
        )
        
        assert trace.problem == "Test problem"
        assert len(trace.nodes) == 1
        assert trace.verified is True
    
    def test_is_high_quality(self, sample_trace):
        """Test high quality trace detection."""
        recorder = TraceRecorder()
        
        assert recorder.is_high_quality(sample_trace) is True
        
        # Unverified trace is not high quality
        sample_trace.verified = False
        assert recorder.is_high_quality(sample_trace) is False
        
        # Empty answer is not high quality
        sample_trace.verified = True
        sample_trace.final_answer = ""
        assert recorder.is_high_quality(sample_trace) is False


class TestTraceStore:
    """Tests for TraceStore."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary trace store."""
        temp_dir = tempfile.mkdtemp()
        store = TraceStore(storage_path=temp_dir)
        yield store
        shutil.rmtree(temp_dir)
    
    def test_save_and_load(self, temp_store, sample_trace):
        """Test saving and loading traces."""
        temp_store.save(sample_trace)
        
        loaded = temp_store.load(sample_trace.trace_id, sample_trace.template_id)
        
        assert loaded is not None
        assert loaded.trace_id == sample_trace.trace_id
        assert loaded.template_id == sample_trace.template_id
    
    def test_query_verified_only(self, temp_store, sample_trace):
        """Test querying verified traces only."""
        temp_store.save(sample_trace)
        
        # Create an unverified trace
        unverified = ExecutionTrace(
            trace_id="unverified-001",
            timestamp=datetime.now(),
            problem="Another problem",
            template_id="other_template",
            nodes=[],
            final_answer="",
            verified=False,
            verification_method="none",
            total_duration_ms=1000.0,
        )
        temp_store.save(unverified)
        
        # Query verified only
        results = temp_store.query(verified_only=True)
        assert len(results) == 1
        assert results[0].trace_id == sample_trace.trace_id
        
        # Query all
        results = temp_store.query(verified_only=False)
        assert len(results) == 2
    
    def test_get_statistics(self, temp_store, sample_trace):
        """Test statistics calculation."""
        temp_store.save(sample_trace)
        
        stats = temp_store.get_statistics()
        
        assert stats["total_traces"] == 1
        assert stats["verified_traces"] == 1
        assert stats["overall_success_rate"] == 1.0
    
    def test_count(self, temp_store, sample_trace):
        """Test trace counting."""
        assert temp_store.count() == 0
        
        temp_store.save(sample_trace)
        
        assert temp_store.count() == 1
        assert temp_store.count(verified_only=True) == 1

