"""
llmosafe — Predictive resource-pressure instrumentation and runtime guardrails.

Signal classes:

  DIRECT GUARANTEES (enforcement-grade — raise exceptions):
    check_resources()       → RSS memory ceiling
    get_stability()         → cognitive entropy threshold

  PREDICTIVE SIGNALS (advisory — compose into your policy):
    get_environmental_entropy() → weighted composite (RSS 50%, IO wait 25%, CPU 25%)
    get_resource_pressure()     → RSS as % of ceiling
    get_system_cpu_load()       → CPU load %

  BEHAVIORAL SIGNALS:
    calculate_halo()       → manipulation detection via dual-path sifter
                            (classifier + keyword bias, returns entropy [0, 65535])

For disk exhaustion protection, compose llmosafe signals with
shutil.disk_usage(). See README for the canonical cookbook.

Example:
    >>> from llmosafe import check_resources, get_environmental_entropy
    >>> check_resources(1024)  # 1GB RSS ceiling
    0
    >>> get_environmental_entropy()  # 0-1000, IO wait is key for disk
    15

"""

from llmosafe._llmosafe import (
    BiasHaloDetectedError,
    CognitiveInstabilityError,
    CognitivePipeline,
    LLMOSafeError,
    ResourceExhaustedError,
    calculate_halo,
    check_resources,
    combined_risk_bits,
    get_body_pressure,
    get_classifier_score,
    get_decision,
    get_detection_flags,
    get_entropy,
    get_environmental_entropy,
    get_kernel_output,
    get_oov_ratio,
    get_pid_state,
    get_resource_pressure,
    get_stability,
    get_stages_executed,
    get_step_count,
    get_surprise,
    get_system_cpu_load,
    memory_stats,
    process_synapse,
)

__version__: str = "0.7.1"

__all__ = [
    "BiasHaloDetectedError",
    "CognitiveInstabilityError",
    # Classes
    "CognitivePipeline",
    # Exceptions
    "LLMOSafeError",
    "ResourceExhaustedError",
    # Functions
    "calculate_halo",
    "check_resources",
    "combined_risk_bits",
    "get_body_pressure",
    "get_classifier_score",
    "get_decision",
    "get_detection_flags",
    "get_entropy",
    "get_environmental_entropy",
    "get_kernel_output",
    "get_oov_ratio",
    "get_pid_state",
    "get_resource_pressure",
    "get_stability",
    "get_stages_executed",
    "get_step_count",
    "get_surprise",
    "get_system_cpu_load",
    "make_synapse",
    "memory_stats",
    "parse_synapse",
    "process_synapse",
]


# ── Synapse constructor ────────────────────────────────────────


def make_synapse(entropy: int, surprise: int = 0, has_bias: bool = False) -> int:
    """Construct a synapse_bits value for get_stability() / process_synapse().

    The synapse encodes cognitive state in a 64-bit integer:

        Bits [0:15]  → raw_entropy   (u16, 0-65535)
        Bits [16:31] → raw_surprise  (u16, 0-65535)
        Bit  [32]    → has_bias      (0 or 1)
        Bits [33:44] → position      (u12)
        Bits [45:60] → timestamp     (u16)
        Bits [61:68] → cascade_depth (u8)

    For most usage, only entropy, surprise, and has_bias matter.

    **v0.7.0 thresholds:**
    Entropy is now in classifier probability space [0, 65535]:
      0-40000  = stable, 40000-50000 = pressure, >50000 = unstable.
    Previous v0.6.x keyword-range thresholds (0-1000) are obsolete.

    Args:
        entropy:  Cognitive entropy score (0-65535).
        surprise: Surprise level (0-65535).
        has_bias: Whether bias was detected in the input.

    Returns:
        64-bit synapse value suitable for get_stability() or process_synapse().

    Example:
        >>> from llmosafe import make_synapse, get_stability
        >>> get_stability(make_synapse(entropy=500))
        0
        >>> get_stability(make_synapse(entropy=41000))
        -2
        >>> get_stability(make_synapse(entropy=500, has_bias=True))
        -3

    """
    return (entropy & 0xFFFF) | ((surprise & 0xFFFF) << 16) | ((int(has_bias) & 0x1) << 32)


def parse_synapse(synapse_bits: int) -> dict[str, int | bool]:
    """Parse a synapse_bits value into its component fields.

    Inverse of make_synapse().

    Args:
        synapse_bits: 64-bit synapse value.

    Returns:
        Dict with keys: entropy, surprise, has_bias.

    Example:
        >>> parse_synapse(make_synapse(entropy=400, surprise=100, has_bias=True))
        {'entropy': 400, 'surprise': 100, 'has_bias': True}

    """
    return {
        "entropy": synapse_bits & 0xFFFF,
        "surprise": (synapse_bits >> 16) & 0xFFFF,
        "has_bias": bool((synapse_bits >> 32) & 0x1),
    }
