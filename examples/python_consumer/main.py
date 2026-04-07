#!/usr/bin/env python3
"""
Example: Python bindings for llmosafe via ctypes
This demonstrates how to call llmosafe from Python using the C-ABI.
Requirements:
  cargo build --release --features ffi
  LD_LIBRARY_PATH=./target/release python3 examples/python_consumer/main.py
"""
import ctypes
import os
import sys

# Load the shared library
# Adjusted path to match directory structure
lib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'target', 'release', 'libllmosafe.so')
if not os.path.exists(lib_path):
    print(f"Library not found at {lib_path}")
    print("Build with: cargo build --release --features ffi")
    sys.exit(1)

llmosafe = ctypes.CDLL(lib_path)

# Define function signatures
llmosafe.llmosafe_process_synapse.argtypes = [ctypes.c_uint64]
llmosafe.llmosafe_process_synapse.restype = ctypes.c_int32

llmosafe.llmosafe_calculate_halo.argtypes = [ctypes.c_char_p]
llmosafe.llmosafe_calculate_halo.restype = ctypes.c_uint16

llmosafe.llmosafe_check_resources.argtypes = [ctypes.c_uint32]
llmosafe.llmosafe_check_resources.restype = ctypes.c_int32

llmosafe.llmosafe_get_resource_pressure.argtypes = [ctypes.c_uint32]
llmosafe.llmosafe_get_resource_pressure.restype = ctypes.c_uint8

llmosafe.llmosafe_get_stability.argtypes = [ctypes.c_uint64]
llmosafe.llmosafe_get_stability.restype = ctypes.c_int32

llmosafe.llmosafe_get_system_cpu_load.argtypes = []
llmosafe.llmosafe_get_system_cpu_load.restype = ctypes.c_uint8

llmosafe.llmosafe_get_environmental_entropy.argtypes = []
llmosafe.llmosafe_get_environmental_entropy.restype = ctypes.c_uint16

def calculate_halo(text: str) -> int:
    """Calculate halo signal for text."""
    return llmosafe.llmosafe_calculate_halo(text.encode('utf-8'))

def check_resources(ceiling_mb: int) -> int:
    """Check resources against ceiling."""
    return llmosafe.llmosafe_check_resources(ceiling_mb)

def get_resource_pressure(ceiling_mb: int) -> int:
    """Get resource pressure percentage."""
    return llmosafe.llmosafe_get_resource_pressure(ceiling_mb)

def get_stability(synapse_bits: int) -> int:
    """Check synapse stability."""
    return llmosafe.llmosafe_get_stability(synapse_bits)

def process_synapse(synapse_bits: int) -> int:
    """Process a synapse through the safety pipeline."""
    return llmosafe.llmosafe_process_synapse(synapse_bits)

def get_system_cpu_load() -> int:
    """Get system CPU load percentage."""
    return llmosafe.llmosafe_get_system_cpu_load()

def get_environmental_entropy() -> int:
    """Get environmental entropy score."""
    return llmosafe.llmosafe_get_environmental_entropy()

def main():
    print("=== llmosafe Python Bindings Demo ===\n")

    # Halo signal
    print("--- Halo Signal ---\n")
    safe = "This is a normal sentence."
    print(f"Safe text: \"{safe}\"")
    print(f"Halo signal: {calculate_halo(safe)}\n")

    biased = "The expert provided an official recommendation."
    print(f"Biased text: \"{biased}\"")
    print(f"Halo signal: {calculate_halo(biased)}")
    print("(Higher = more bias detected)\n")

    # Resource checking
    print("--- Resource Checking ---\n")
    ceiling_mb = 1024
    print(f"Memory ceiling: {ceiling_mb} MB")
    print(f"Check result: {check_resources(ceiling_mb)} (0=ok)\n")
    print(f"Resource pressure: {get_resource_pressure(ceiling_mb)}%\n")

    # Stability
    print("--- Stability Check ---\n")
    print(f"Stable synapse (entropy=400): {get_stability(400)}")
    print(f"Unstable synapse (entropy=1100): {get_stability(1100)}")
    print("(0=stable, -2=cognitive instability)\n")

    # System metrics
    print("--- System Metrics ---\n")
    print(f"CPU load: {get_system_cpu_load()}%")
    print(f"Environmental entropy: {get_environmental_entropy()}\n")

    print("=== Demo Complete ===")

if __name__ == "__main__":
    main()
