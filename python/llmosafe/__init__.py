"""
llmosafe - Safety-critical cognitive safety library for AI agents.

This package provides Python bindings to the llmosafe Rust library,
offering formal primitives for building safety-critical AI agents.

Architecture:
    - Tier 0: Resource Body (physical safety)
    - Tier 1: Deterministic Kernel (formal law)
    - Tier 2: Cognitive Working Memory (stateful safety)
    - Tier 3: Perceptual Sifter (boundary safety)

Example:
    >>> from llmosafe import calculate_halo, check_resources
    >>>
    >>> # Check for bias in text
    >>> halo = calculate_halo("The expert recommendation is proven and certified.")
    >>> print(f"Bias score: {halo}")
    >>>
    >>> # Check resource limits
    >>> result = check_resources(1024)  # 1GB ceiling
    >>> if result == 0:
    ...     print("Resources OK")
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

__version__ = "0.6.0"
__all__ = [
    "calculate_halo",
    "check_resources",
    "get_resource_pressure",
    "get_stability",
    "get_system_cpu_load",
    "get_environmental_entropy",
    "process_synapse",
    "LLMOSafeError",
    "ResourceExhaustedError",
    "CognitiveInstabilityError",
    "BiasHaloDetectedError",
]
