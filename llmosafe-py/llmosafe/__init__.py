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
    DesignAssuranceLevel,
    LLMOSafeError,
    PressureLevel,
    PySynapse,
    ResourceExhaustedError,
    SafetyDecision,
    calculate_halo,
    calculate_halo_signal_legacy,
    calculate_utility,
    check_resources,
    combined_risk_bits,
    get_bias_breakdown,
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

__version__: str = "0.7.7"

# Nice names for Python users
Synapse = PySynapse

__all__ = [
    "DETECTION_FLAGS_MASK",
    "FLAG_ADVERSARIAL",
    "FLAG_ANOMALY",
    "FLAG_DECAYING",
    "FLAG_DRIFTING",
    "FLAG_LOW_CONFIDENCE",
    "FLAG_STUCK",
    "PRESSURE_THRESHOLD",
    "STABILITY_THRESHOLD",
    "STAGE_BODY",
    "STAGE_DETECTION",
    "STAGE_KERNEL",
    "STAGE_MEMORY",
    "STAGE_MONITOR",
    "STAGE_SIFT",
    "BiasHaloDetectedError",
    "CognitiveInstabilityError",
    "CognitivePipeline",
    "DesignAssuranceLevel",
    "LLMOSafeError",
    "PressureLevel",
    "ResourceExhaustedError",
    "SafetyDecision",
    "Synapse",
    "calculate_halo",
    "calculate_halo_signal_legacy",
    "calculate_utility",
    "check_resources",
    "combined_risk_bits",
    "get_bias_breakdown",
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


# ── Constants (mirror of Rust public API) ─────────────────────

# Entropy thresholds (classifier space [0, 65535])
STABILITY_THRESHOLD: int = 50000
PRESSURE_THRESHOLD: int = 40000

# Pipeline stage bitmasks (PipelineResult.stages_executed)
STAGE_SIFT: int = 0x01
STAGE_MEMORY: int = 0x02
STAGE_KERNEL: int = 0x04
STAGE_DETECTION: int = 0x08
STAGE_MONITOR: int = 0x10
STAGE_BODY: int = 0x20  # only set when process_with_pressure() was used

# Detection flags (packed into Synapse reserved bits 0-5 and PipelineResult.detection_flags)
FLAG_STUCK: int = 0x01
FLAG_DRIFTING: int = 0x02
FLAG_LOW_CONFIDENCE: int = 0x04
FLAG_DECAYING: int = 0x08
FLAG_ANOMALY: int = 0x10
FLAG_ADVERSARIAL: int = 0x20
DETECTION_FLAGS_MASK: int = 0x3F

# ── Synapse constructor / parser (full 128-bit layout) ─────────


def make_synapse(
    entropy: int,
    surprise: int = 0,
    has_bias: bool = False,
    *,
    position: int = 0,
    timestamp: int = 0,
    cascade_depth: int = 0,
    anchor_hash: int = 0,
    detection_flags: int = 0,
    oov_ratio: int = 0,
) -> int:
    """Construct a full 128-bit synapse value.

    Layout (MSB → LSB):
        [Entropy:16][Surprise:16][Bias:1][Position:12][Timestamp:16]
        [Cascade:8][AnchorHash:31][Reserved:28]
        Reserved sub-layout (bits 0-27 of the reserved field):
            [OOV:8 (bits 6-13)][FLAGS:6 (bits 0-5)][upper reserved:14]

    For normal use only entropy/surprise/has_bias + optional detection_flags/oov_ratio matter.

    Thresholds (v0.7+ classifier space):
        0-40000  = stable zone
        40000-50000 = pressure zone
        >50000 = CognitiveInstability (get_stability / validate)

    Args:
        entropy: raw_entropy (0-65535)
        surprise: raw_surprise (0-65535)
        has_bias: bias flag
        position: position field (0-4095, advanced, rarely needed)
        timestamp: timestamp field (0-65535, advanced, rarely needed)
        cascade_depth: cascade depth (0-255, advanced, rarely needed)
        anchor_hash: anchor hash (0-2^31-1, advanced, rarely needed)
        detection_flags: 0-0x3F (FLAG_* values OR-ed together)
        oov_ratio: 0-255 (0=0%, 255=100%)

    Returns:
        Integer suitable for get_stability(), process_synapse(), combined_risk_bits().

    The returned value is a Python int (unlimited precision) and is passed
    through to the u128 C-ABI functions.

    """
    e = entropy & 0xFFFF
    s = surprise & 0xFFFF
    b = 1 if has_bias else 0
    pos = position & 0xFFF
    ts = timestamp & 0xFFFF
    cd = cascade_depth & 0xFF
    ah = anchor_hash & 0x7FFFFFFF

    # Lower 64 bits: entropy(16) | surprise(16) | bias(1) | pos(12) | ts(16) | cd(3 of 8)
    # We pack what fits cleanly; the bitfield crate handles the exact layout on the Rust side.
    # For round-tripping we keep the construction simple and let from_raw_u128 do the work.
    lower = (
        (e)
        | (s << 16)
        | (b << 32)
        | (pos << 33)
        | (ts << 45)
        | ((cd & 0x07) << 61)  # low 3 bits of cascade into lower word
    )

    # Upper 64 bits start with remaining cascade bits + anchor_hash + reserved
    # Cascade high 5 bits | anchor_hash(31) | reserved(28)
    upper = ((cd >> 3) & 0x1F) | (ah << 5)

    # Reserved field (28 bits) layout inside the upper word:
    # bits 0-5  = detection flags
    # bits 6-13 = oov_ratio
    reserved = ((oov_ratio & 0xFF) << 6) | (detection_flags & 0x3F)
    upper |= (reserved & 0x0FFFFFFF) << (5 + 31)  # after 5 cascade + 31 hash

    return (upper << 64) | lower


def parse_synapse(synapse_bits: int) -> dict[str, int | bool]:
    """Parse a (possibly 128-bit) synapse into its fields.

    Returns a dict with the main observable fields plus raw lower/upper words.
    """
    lower = synapse_bits & ((1 << 64) - 1)
    upper = (synapse_bits >> 64) & ((1 << 64) - 1)

    entropy = lower & 0xFFFF
    surprise = (lower >> 16) & 0xFFFF
    has_bias = bool((lower >> 32) & 0x1)

    # Best-effort extraction (exact bit positions are enforced by the Rust bitfield)
    position = (lower >> 33) & 0xFFF
    timestamp = (lower >> 45) & 0xFFFF
    cascade_low = (lower >> 61) & 0x7
    cascade_high = upper & 0x1F
    cascade_depth = (cascade_high << 3) | cascade_low
    anchor_hash = (upper >> 5) & 0x7FFFFFFF

    reserved = (upper >> (5 + 31)) & 0x0FFFFFFF
    detection_flags = reserved & 0x3F
    oov_ratio = (reserved >> 6) & 0xFF

    return {
        "entropy": entropy,
        "surprise": surprise,
        "has_bias": has_bias,
        "position": position,
        "timestamp": timestamp,
        "cascade_depth": cascade_depth,
        "anchor_hash": anchor_hash,
        "detection_flags": detection_flags,
        "oov_ratio": oov_ratio,
        "lower64": lower,
        "upper64": upper,
    }
