"""Tests for backtracking functionality."""

import pytest

from ..node_verifier import NodeVerifier, VerificationResult
from ..backtrack_manager import (
    BacktrackManager,
    StateManager,
    BacktrackDecision,
    RetryStrategy,
)


class TestNodeVerifier:
    """Tests for NodeVerifier."""
    
    def test_verify_empty_output(self):
        """Test verification of empty output."""
        verifier = NodeVerifier()
        
        result = verifier.verify_node_output(
            node_id="test",
            node_type="calculation",
            role="math",
            output="",
        )
        
        assert result.passed is False
        assert result.confidence == 0.0
    
    def test_verify_error_output(self):
        """Test verification of error output."""
        verifier = NodeVerifier()
        
        result = verifier.verify_node_output(
            node_id="test",
            node_type="calculation",
            role="math",
            output="[error: something went wrong]",
        )
        
        assert result.passed is False
    
    def test_verify_definition_node(self):
        """Test verification of definition node."""
        verifier = NodeVerifier()
        
        # Good definition output
        result = verifier.verify_node_output(
            node_id="test",
            node_type="definition",
            role="logic",
            output="Let G denote the cyclic group of order n. We define the connection set S as...",
        )
        
        assert result.passed is True
        assert result.confidence > 0.5
    
    def test_verify_calculation_node(self):
        """Test verification of calculation node."""
        verifier = NodeVerifier()
        
        # Good calculation output with numeric result
        result = verifier.verify_node_output(
            node_id="test",
            node_type="calculation",
            role="research",
            output="The result of the computation is #### 42",
            instruction="Compute the value. End with #### number.",
        )
        
        assert result.passed is True
    
    def test_verify_enumeration_node(self):
        """Test verification of enumeration node."""
        verifier = NodeVerifier()
        
        # Good enumeration with list structure
        result = verifier.verify_node_output(
            node_id="test",
            node_type="enumeration",
            role="logic",
            output="1. First case\n2. Second case\n3. Third case",
        )
        
        assert result.passed is True
    
    def test_verify_with_expected_patterns(self):
        """Test verification with expected patterns."""
        verifier = NodeVerifier()
        
        result = verifier.verify_node_output(
            node_id="test",
            node_type="calculation",
            role="math",
            output="The eigenvalue is 42",
            expected_patterns=["eigenvalue", r"\d+"],
        )
        
        assert result.passed is True


class TestStateManager:
    """Tests for StateManager."""
    
    def test_checkpoint_and_restore(self):
        """Test checkpoint save and restore."""
        manager = StateManager()
        
        results = {"n1": "output1", "n2": "output2"}
        checkpoint_id = manager.save_checkpoint(results)
        
        # Modify results
        results["n3"] = "output3"
        
        # Restore
        restored = manager.restore_checkpoint(checkpoint_id)
        
        assert "n1" in restored
        assert "n2" in restored
        assert "n3" not in restored
    
    def test_node_state_tracking(self):
        """Test node state tracking."""
        manager = StateManager()
        
        state = manager.initialize_node("n1", max_retries=3)
        assert state.retry_count == 0
        assert state.can_retry is True
        
        manager.increment_retry("n1")
        state = manager.get_node_state("n1")
        assert state.retry_count == 1
        
        manager.increment_retry("n1")
        manager.increment_retry("n1")
        state = manager.get_node_state("n1")
        assert state.retry_count == 3
        assert state.can_retry is False
    
    def test_invalidate_downstream(self):
        """Test downstream invalidation."""
        manager = StateManager()
        
        # Initialize some nodes
        manager.initialize_node("n1")
        manager.initialize_node("n2")
        manager.initialize_node("n3")
        
        manager.update_node("n1", "output1", verified=True)
        manager.update_node("n2", "output2", verified=True)
        manager.update_node("n3", "output3", verified=True)
        
        # Define edges: n1 -> n2 -> n3
        edges = [("n1", "n2"), ("n2", "n3")]
        
        # Invalidate n1 should invalidate n2 and n3
        invalidated = manager.invalidate_downstream("n1", edges)
        
        assert "n2" in invalidated
        assert "n3" in invalidated
        assert "n1" not in invalidated
    
    def test_apply_adjustments(self):
        """Test applying retry adjustments."""
        manager = StateManager()
        manager.initialize_node("n1")
        
        adjustments = {
            "temperature_delta": -0.1,
            "role_override": "research",
            "extra_context": "Be more careful",
        }
        
        state = manager.apply_adjustments("n1", adjustments)
        
        assert state.temperature_adjustment == -0.1
        assert state.role_override == "research"
        assert state.extra_context == "Be more careful"


class TestBacktrackManager:
    """Tests for BacktrackManager."""
    
    def test_decide_backtrack_exhausted(self):
        """Test backtrack decision when retries exhausted."""
        manager = BacktrackManager(max_retries_per_node=2)
        
        # Exhaust retries
        manager.state_manager.initialize_node("n1", max_retries=2)
        manager.state_manager.increment_retry("n1")
        manager.state_manager.increment_retry("n1")
        
        verification = VerificationResult(
            passed=False,
            confidence=0.3,
            issues=["Output too short"],
        )
        
        decision = manager.decide_backtrack(
            failed_node_id="n1",
            verification_result=verification,
            node_type="calculation",
            node_role="research",
            edges=[],
        )
        
        assert decision.should_backtrack is False
        assert decision.strategy == RetryStrategy.SKIP
    
    def test_decide_backtrack_with_retries(self):
        """Test backtrack decision when retries available."""
        manager = BacktrackManager(max_retries_per_node=2)
        manager.state_manager.initialize_node("n1", max_retries=2)
        
        verification = VerificationResult(
            passed=False,
            confidence=0.3,
            issues=["Calculation error"],
            suggested_fix="adjust_params",
        )
        
        decision = manager.decide_backtrack(
            failed_node_id="n1",
            verification_result=verification,
            node_type="calculation",
            node_role="research",
            edges=[],
        )
        
        assert decision.should_backtrack is True
        assert decision.target_node_id == "n1"
    
    def test_verification_node_targets_upstream(self):
        """Test that verification failure targets upstream calculation."""
        manager = BacktrackManager()
        manager.state_manager.initialize_node("n2_calculate")
        manager.state_manager.initialize_node("n3_verify")
        
        verification = VerificationResult(
            passed=False,
            confidence=0.2,
            issues=["Verification failed"],
        )
        
        edges = [("n1_define", "n2_calculate"), ("n2_calculate", "n3_verify")]
        
        decision = manager.decide_backtrack(
            failed_node_id="n3_verify",
            verification_result=verification,
            node_type="verification",
            node_role="verifier",
            edges=edges,
        )
        
        # Should target the calculation node, not the verification node
        assert decision.target_node_id == "n2_calculate"
    
    def test_prepare_retry(self):
        """Test preparing for a retry."""
        manager = BacktrackManager()
        manager.state_manager.initialize_node("n1")
        
        decision = BacktrackDecision(
            should_backtrack=True,
            target_node_id="n1",
            strategy=RetryStrategy.ADJUST_PARAMS,
            reason="Test reason",
            adjustments={"temperature_delta": -0.1},
        )
        
        params = manager.prepare_retry("n1", decision)
        
        assert params["retry_count"] == 1
        assert params["temperature_adjustment"] == -0.1
    
    def test_should_verify_node(self):
        """Test node verification decision."""
        manager = BacktrackManager()
        
        # Calculation and verification nodes should be verified
        assert manager.should_verify_node("calculation", "research") is True
        assert manager.should_verify_node("verification", "verifier") is True
        assert manager.should_verify_node("aggregation", "logic") is True
        
        # Math role should be verified
        assert manager.should_verify_node("definition", "math") is True
        
        # Definition with logic role should not be verified (to reduce overhead)
        assert manager.should_verify_node("definition", "logic") is False
    
    def test_get_statistics(self):
        """Test statistics gathering."""
        manager = BacktrackManager()
        
        # Simulate some backtracks
        manager.state_manager.initialize_node("n1")
        manager.state_manager.increment_retry("n1")
        
        manager.backtrack_history.append(BacktrackDecision(
            should_backtrack=True,
            target_node_id="n1",
            strategy=RetryStrategy.RETRY_SAME,
            reason="Test",
        ))
        manager._total_backtracks = 1
        
        stats = manager.get_statistics()
        
        assert stats["total_backtracks"] == 1
        assert stats["decisions"] == 1
        assert stats["nodes_with_retries"] == 1

