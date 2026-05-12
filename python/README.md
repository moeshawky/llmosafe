# llmosafe Python Bindings

Safety-critical cognitive safety library for AI agents.

## Installation

### From PyPI (recommended)

```bash
pip install llmosafe
```

### From source

```bash
# Install maturin (build tool)
pip install maturin

# Build and install
cd python
maturin develop --release

# Or build wheel
maturin build --release
pip install dist/llmosafe-*.whl
```

## Quick Start

```python
from llmosafe import calculate_halo, check_resources, get_stability

# Check for cognitive bias in text
text = "The expert recommendation is proven and certified."
bias_score = calculate_halo(text)
print(f"Bias score: {bias_score}")  # Higher = more bias detected

# Check resource limits
try:
    check_resources(1024)  # 1GB ceiling
    print("Resources OK")
except llmosafe.ResourceExhaustedError:
    print("Memory limit exceeded!")

# Check cognitive stability
stability = get_stability(synapse_bits=400)
if stability == 0:
    print("Cognitive state stable")
else:
    print(f"Instability detected: {stability}")
```

## API Reference

### Bias Detection

```python
calculate_halo(text: str) -> int
```

Calculate the "halo signal" (bias score) for text. Detects:
- Authority bias (expert, official, certified)
- Social proof (popular, trending, consensus)
- Scarcity (limited, exclusive, rare)
- Urgency (now, fast, deadline)
- Emotional appeal (love, fear, miracle)
- Expertise signaling (sophisticated, cutting-edge)

Returns: Bias score (0 = no bias, higher = more bias patterns detected)

### Resource Management

```python
check_resources(ceiling_mb: int) -> int
```

Check if current memory usage is within ceiling.

Returns: 0 if OK, raises `ResourceExhaustedError` if exceeded.

```python
get_resource_pressure(ceiling_mb: int) -> int
```

Get current memory pressure as percentage (0-100).

### Stability Checking

```python
get_stability(synapse_bits: int) -> int
```

Check if cognitive state (synapse) is stable.

Returns: 0 if stable, -2 if cognitive instability, -3 if bias halo detected.

### System Metrics

```python
get_system_cpu_load() -> int
```

Get current CPU load percentage (0-100).

```python
get_environmental_entropy() -> int
```

Get environmental entropy score (0-1000, higher = more entropy).

### Advanced

```python
process_synapse(synapse_bits: int) -> int
```

Process a cognitive state update through the full safety pipeline.

Returns: 0 if successful, negative error code otherwise.

## Exceptions

All exceptions inherit from `llmosafe.LLMOSafeError`:

- `ResourceExhaustedError`: Memory ceiling exceeded
- `CognitiveInstabilityError`: Cognitive entropy threshold exceeded
- `BiasHaloDetectedError`: Bias pattern detected in input

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest llmosafe/tests -v

# Type checking
mypy llmosafe

# Build wheel
maturin build --release
```

## License

MIT License - see LICENSE file.
