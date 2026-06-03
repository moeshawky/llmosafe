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
    calculate_halo()       → manipulation pattern detection in text

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
    calculate_halo,
    check_resources,
    get_resource_pressure,
    get_stability,
    get_system_cpu_load,
    get_environmental_entropy,
    process_synapse,
    LLMOSafeError,
    ResourceExhaustedError,
    CognitiveInstabilityError,
    BiasHaloDetectedError,
)

__version__ = "0.6.2"

__all__ = [
    # Functions
    "calculate_halo",
    "check_resources",
    "get_resource_pressure",
    "get_stability",
    "get_system_cpu_load",
    "get_environmental_entropy",
    "process_synapse",
    "make_synapse",
    "parse_synapse",
    # Exceptions
    "LLMOSafeError",
    "ResourceExhaustedError",
    "CognitiveInstabilityError",
    "BiasHaloDetectedError",
]


# ── Synapse constructor ────────────────────────────────────────

def make_synapse(entropy: int, surprise: int = 0, has_bias: bool = False) -> int:
    """Construct a synapse_bits value for get_stability() / process_synapse().

    The synapse encodes cognitive state in a 64-bit integer:

        Bits [0:15]  → raw_entropy   (u16, operational range 0–1000)
        Bits [16:31] → raw_surprise  (u16, 0–65535)
        Bit  [32]    → has_bias      (0 or 1)
        Bits [33:44] → position      (u12)
        Bits [45:60] → timestamp     (u16)
        Bits [61:68] → cascade_depth (u8)

    For most usage, only entropy, surprise, and has_bias matter.

    Args:
        entropy:  Cognitive entropy score (0–1000).
                  0–800 = stable, 800–1000 = pressure, >1000 = unstable.
        surprise: Surprise level (0–65535). Rejects if > 500 in process_synapse().
        has_bias: Whether bias was detected in the input.

    Returns:
        64-bit synapse value suitable for get_stability() or process_synapse().

    Example:
        >>> from llmosafe import make_synapse, get_stability
        >>> get_stability(make_synapse(entropy=400))
        0
        >>> get_stability(make_synapse(entropy=1100))
        -2
        >>> get_stability(make_synapse(entropy=400, has_bias=True))
        -3
    """
    return (
        (entropy & 0xFFFF)
        | ((surprise & 0xFFFF) << 16)
        | ((int(has_bias) & 0x1) << 32)
    )


def parse_synapse(synapse_bits: int) -> dict:
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
