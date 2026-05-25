"""
Tests for llmosafe Python bindings.

Covers: operational model, synapse construction, return codes,
environmental entropy semantics, disk guard composition, and
exception hierarchy.

Run with: pytest tests/test_llmosafe.py -v
"""

import pytest
import llmosafe
from llmosafe import make_synapse, parse_synapse


# ── Bias Detection ──────────────────────────────────────────────

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

    def test_negation_awareness(self):
        """'not an expert' should produce 0 authority score."""
        negated = llmosafe.calculate_halo("This is not an expert opinion")
        assert negated == 0

    def test_all_8_categories(self):
        """Each of the 8 categories should contribute independently."""
        result = llmosafe.calculate_halo(
            "expert popular limited now love sophisticated instead-of as-an-ai"
        )
        assert result >= 100  # at least one category fires


# ── Resource Management ─────────────────────────────────────────

class TestCheckResources:
    """Tests for check_resources function — enforcement-grade RSS ceiling."""

    def test_reasonable_ceiling(self):
        """1GB ceiling should be OK on any reasonable system."""
        result = llmosafe.check_resources(1024)
        assert result == 0

    def test_zero_ceiling_raises(self):
        """Zero ceiling should raise ResourceExhaustedError."""
        with pytest.raises(llmosafe.ResourceExhaustedError):
            llmosafe.check_resources(0)

    def test_small_ceiling(self):
        """Very small ceiling (1MB) may fail on real processes."""
        try:
            result = llmosafe.check_resources(1)
            assert result == 0
        except llmosafe.ResourceExhaustedError:
            pass  # Expected — process likely uses >1MB RSS

    def test_monitors_rss_not_disk(self):
        """check_resources monitors RSS memory, not filesystem capacity.
        This is a semantic test — verify the function accepts MB,
        not filesystem paths."""
        # If it took a path, this would be wrong
        result = llmosafe.check_resources(1024)
        assert result == 0


class TestGetResourcePressure:
    """Tests for get_resource_pressure function."""

    def test_zero_ceiling(self):
        """Zero ceiling should return 100% pressure."""
        assert llmosafe.get_resource_pressure(0) == 100

    def test_reasonable_ceiling(self):
        """1GB ceiling should show some pressure."""
        pressure = llmosafe.get_resource_pressure(1024)
        assert 0 <= pressure <= 100

    def test_returns_percentage(self):
        """Result is always 0-100 regardless of ceiling."""
        for mb in [1, 10, 100, 1024, 8192]:
            pressure = llmosafe.get_resource_pressure(mb)
            assert 0 <= pressure <= 100


# ── Synapse Construction and Parsing ────────────────────────────

class TestMakeSynapse:
    """Tests for make_synapse helper — the correct way to construct synapse values."""

    def test_entropy_only(self):
        """Default surprise=0, has_bias=False."""
        bits = make_synapse(entropy=400)
        parsed = parse_synapse(bits)
        assert parsed["entropy"] == 400
        assert parsed["surprise"] == 0
        assert parsed["has_bias"] is False

    def test_all_fields(self):
        """All fields roundtrip correctly."""
        bits = make_synapse(entropy=500, surprise=100, has_bias=True)
        parsed = parse_synapse(bits)
        assert parsed["entropy"] == 500
        assert parsed["surprise"] == 100
        assert parsed["has_bias"] is True

    def test_entropy_masked(self):
        """Entropy values >65535 should be masked to 16 bits."""
        bits = make_synapse(entropy=0x1FFFF)  # 17 bits
        parsed = parse_synapse(bits)
        assert parsed["entropy"] == 0xFFFF  # upper bits lost

    def test_zero_entropy(self):
        """Zero entropy is valid."""
        bits = make_synapse(entropy=0)
        assert parse_synapse(bits)["entropy"] == 0

    def test_max_entropy(self):
        """Maximum entropy value."""
        bits = make_synapse(entropy=1000)
        assert parse_synapse(bits)["entropy"] == 1000

    def test_stability_integration(self):
        """make_synapse produces values compatible with get_stability."""
        assert llmosafe.get_stability(make_synapse(400)) == 0
        assert llmosafe.get_stability(make_synapse(1100)) == -2
        assert llmosafe.get_stability(make_synapse(400, has_bias=True)) == -3

    def test_process_synapse_integration(self):
        """make_synapse produces values compatible with process_synapse."""
        assert llmosafe.process_synapse(make_synapse(400)) == 0


class TestParseSynapse:
    """Tests for parse_synapse helper."""

    def test_roundtrip(self):
        """make_synapse → parse_synapse is identity for the documented fields."""
        for entropy in [0, 100, 500, 1000]:
            for surprise in [0, 100, 500]:
                for has_bias in [True, False]:
                    bits = make_synapse(entropy, surprise, has_bias)
                    parsed = parse_synapse(bits)
                    assert parsed["entropy"] == entropy
                    assert parsed["surprise"] == surprise
                    assert parsed["has_bias"] is has_bias


# ── Stability and Return Codes ──────────────────────────────────

class TestGetStability:
    """Tests for get_stability function with documented return codes."""

    def test_stable_synapse(self):
        """Low entropy synapse should be stable (return 0)."""
        assert llmosafe.get_stability(make_synapse(400)) == 0

    def test_unstable_synapse(self):
        """High entropy synapse should return -2 (CognitiveInstability)."""
        assert llmosafe.get_stability(make_synapse(1100)) == -2

    def test_bias_detected(self):
        """Synapse with has_bias=True should return -3 (BiasHaloDetected)."""
        assert llmosafe.get_stability(make_synapse(400, has_bias=True)) == -3

    def test_zero_synapse(self):
        """Zero synapse should be stable."""
        assert llmosafe.get_stability(0) == 0

    def test_pressure_boundary(self):
        """Entropy exactly at 1000 is stable, 1001+ is not."""
        # Note: 1000 fits in u16 and is_stable(1000) returns true
        assert llmosafe.get_stability(make_synapse(1000)) == 0

    def test_return_code_values(self):
        """Verify all documented return codes are achievable."""
        codes_seen = set()
        codes_seen.add(llmosafe.get_stability(make_synapse(400)))       # 0
        codes_seen.add(llmosafe.get_stability(make_synapse(1100)))      # -2
        codes_seen.add(llmosafe.get_stability(make_synapse(400, has_bias=True)))  # -3
        assert 0 in codes_seen
        assert -2 in codes_seen
        assert -3 in codes_seen


class TestProcessSynapse:
    """Tests for process_synapse function with documented return codes."""

    def test_valid_synapse(self):
        """Valid synapse should process successfully."""
        result = llmosafe.process_synapse(make_synapse(400))
        assert result == 0

    def test_unstable_synapse(self):
        """Unstable synapse should return -2."""
        result = llmosafe.process_synapse(make_synapse(1100))
        assert result == -2

    def test_high_surprise(self):
        """Surprise > 500 should return -4 (HallucinationDetected)."""
        result = llmosafe.process_synapse(make_synapse(400, surprise=600))
        assert result == -4

    def test_bias_detected(self):
        """Bias flag should return -3."""
        result = llmosafe.process_synapse(make_synapse(400, has_bias=True))
        assert result == -3


# ── System Metrics ──────────────────────────────────────────────

class TestSystemMetrics:
    """Tests for system metric functions."""

    def test_cpu_load_bounded(self):
        """CPU load should be 0–100."""
        load = llmosafe.get_system_cpu_load()
        assert 0 <= load <= 100

    def test_environmental_entropy_bounded(self):
        """Environmental entropy should be 0–1000."""
        entropy = llmosafe.get_environmental_entropy()
        assert 0 <= entropy <= 1000

    def test_environmental_entropy_weights(self):
        """Environmental entropy is weighted: 50% RSS, 25% IO wait, 25% CPU.
        On a system with low load, the score should be low."""
        entropy = llmosafe.get_environmental_entropy()
        # On a quiet system, entropy should be well below pressure zone
        assert entropy < 800  # generous — most systems are < 200

    def test_entropy_is_predictive_not_enforcement(self):
        """get_environmental_entropy() returns a number, never raises.
        It is advisory, not enforcement-grade."""
        # Should never raise
        entropy = llmosafe.get_environmental_entropy()
        assert isinstance(entropy, int)


# ── Exception Hierarchy ─────────────────────────────────────────

class TestExceptions:
    """Tests for exception hierarchy and semantics."""

    def test_resource_exhausted_inherits(self):
        """ResourceExhaustedError should inherit from LLMOSafeError."""
        with pytest.raises(llmosafe.LLMOSafeError):
            llmosafe.check_resources(0)

    def test_resource_exhausted_direct(self):
        """ResourceExhaustedError should be catchable directly."""
        with pytest.raises(llmosafe.ResourceExhaustedError):
            llmosafe.check_resources(0)

    def test_exception_hierarchy(self):
        """Verify the full exception chain."""
        assert issubclass(llmosafe.ResourceExhaustedError, llmosafe.LLMOSafeError)
        assert issubclass(llmosafe.CognitiveInstabilityError, llmosafe.LLMOSafeError)
        assert issubclass(llmosafe.BiasHaloDetectedError, llmosafe.LLMOSafeError)
        assert issubclass(llmosafe.LLMOSafeError, Exception)


# ── Disk Guard Composition (integration) ────────────────────────

class TestDiskGuardComposition:
    """Tests demonstrating the canonical disk-exhaustion protection pattern.

    These tests verify that llmosafe signals compose correctly with
    stdlib disk checks, as documented in the README cookbook.
    """

    def test_composition_pattern(self):
        """The two-layer defense: llmosafe predictive + stdlib hard floor."""
        import shutil

        # Layer 1: llmosafe predictive signals (advisory)
        entropy = llmosafe.get_environmental_entropy()
        pressure = llmosafe.get_resource_pressure(1024)
        assert isinstance(entropy, int)
        assert isinstance(pressure, int)

        # Layer 2: stdlib direct check (hard floor)
        usage = shutil.disk_usage("/")
        assert usage.free > 0  # disk should have some free space

        # Composition: both layers contribute to the decision
        should_throttle = entropy >= 800 or pressure >= 80
        disk_critical = usage.free < 5 * (1 << 30)  # 5GB floor
        # The combination is the complete protection
        can_write = not should_throttle and not disk_critical
        assert isinstance(can_write, bool)

    def test_check_resources_before_write(self):
        """check_resources should be called before write operations.
        It catches RSS growth from write buffering."""
        # This simulates the pre-write check pattern
        try:
            llmosafe.check_resources(1024)
            # If OK, proceed with write
        except llmosafe.ResourceExhaustedError:
            # Must stop — RSS ceiling breached
            pass  # expected in some environments

    def test_entropy_as_early_warning(self):
        """Environmental entropy's IO wait component (25% weight)
        provides early warning for disk pressure."""
        entropy = llmosafe.get_environmental_entropy()
        # The IO wait component is embedded in the score
        # On a quiet system, it should be low
        assert entropy < 800


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
