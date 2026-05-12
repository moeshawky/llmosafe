"""
Tests for llmosafe Python bindings.

Run with: pytest tests/test_llmosafe.py -v
"""

import pytest
import llmosafe


class TestCalculateHalo:
    """Tests for calculate_halo function."""

    def test_empty_string(self):
        """Empty string should return 0 bias."""
        assert llmosafe.calculate_halo("") == 0

    def test_no_bias(self):
        """Normal text should have low bias."""
        result = llmosafe.calculate_halo("This is a normal sentence.")
        assert result == 0

    def test_authority_bias(self):
        """Authority keywords should increase bias score."""
        result = llmosafe.calculate_halo("The expert recommendation")
        assert result > 0

    def test_multiple_bias_types(self):
        """Multiple bias keywords should accumulate."""
        result = llmosafe.calculate_halo(
            "The expert official certified recommendation"
        )
        assert result >= 300  # At least 3 patterns × 100

    def test_case_insensitive(self):
        """Bias detection should be case-insensitive."""
        lower = llmosafe.calculate_halo("the expert")
        upper = llmosafe.calculate_halo("THE EXPERT")
        mixed = llmosafe.calculate_halo("ThE ExPeRt")
        assert lower == upper == mixed
        assert lower > 0


class TestCheckResources:
    """Tests for check_resources function."""

    def test_reasonable_ceiling(self):
        """1GB ceiling should be OK."""
        result = llmosafe.check_resources(1024)
        assert result == 0

    def test_zero_ceiling(self):
        """Zero ceiling should fail."""
        with pytest.raises(llmosafe.ResourceExhaustedError):
            llmosafe.check_resources(0)

    def test_small_ceiling(self):
        """Very small ceiling may fail."""
        # This might or might not fail depending on system state
        result = llmosafe.check_resources(1)
        assert result in (0, -5)  # OK or ResourceExhaustion


class TestGetResourcePressure:
    """Tests for get_resource_pressure function."""

    def test_zero_ceiling(self):
        """Zero ceiling should return 100% pressure."""
        assert llmosafe.get_resource_pressure(0) == 100

    def test_reasonable_ceiling(self):
        """1GB ceiling should show some pressure."""
        pressure = llmosafe.get_resource_pressure(1024)
        assert 0 <= pressure <= 100


class TestGetStability:
    """Tests for get_stability function."""

    def test_stable_synapse(self):
        """Low entropy synapse should be stable."""
        result = llmosafe.get_stability(400)
        assert result == 0

    def test_unstable_synapse(self):
        """High entropy synapse should be unstable."""
        result = llmosafe.get_stability(1100)
        assert result == -2  # CognitiveInstability

    def test_zero_synapse(self):
        """Zero synapse should be stable (edge case)."""
        result = llmosafe.get_stability(0)
        assert result == 0


class TestSystemMetrics:
    """Tests for system metric functions."""

    def test_cpu_load_bounded(self):
        """CPU load should be 0-100."""
        load = llmosafe.get_system_cpu_load()
        assert 0 <= load <= 100

    def test_environmental_entropy_bounded(self):
        """Environmental entropy should be 0-1000."""
        entropy = llmosafe.get_environmental_entropy()
        assert 0 <= entropy <= 1000


class TestProcessSynapse:
    """Tests for process_synapse function."""

    def test_valid_synapse(self):
        """Valid synapse should process successfully."""
        result = llmosafe.process_synapse(400)
        assert result == 0

    def test_unstable_synapse(self):
        """Unstable synapse should return error."""
        result = llmosafe.process_synapse(1100)
        assert result < 0  # Error code


class TestExceptions:
    """Tests for exception handling."""

    def test_resource_exhausted_error(self):
        """Resource exhausted should raise correct exception."""
        with pytest.raises(llmosafe.ResourceExhaustedError):
            llmosafe.check_resources(0)

    def test_exception_inheritance(self):
        """Custom exceptions should inherit from LLMOSafeError."""
        with pytest.raises(llmosafe.LLMOSafeError):
            llmosafe.check_resources(0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
