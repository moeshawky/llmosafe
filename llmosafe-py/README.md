# llmosafe

> **When should I stop?** — Runtime guardrails for systems that process untrusted inputs.

[![PyPI version](https://img.shields.io/pypi/v/llmosafe.svg)](https://pypi.org/project/llmosafe/)
[![Python versions](https://img.shields.io/pypi/pyversions/llmosafe.svg)](https://pypi.org/project/llmosafe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

llmosafe provides three gauges that answer "should I stop?":

1. **Entropy gauge**: Is my system state too chaotic?
2. **Surprise gauge**: Is this result too unexpected?
3. **Bias gauge**: Is this input trying to manipulate me?

When any gauge redlines, execution halts.

---

## Installation

```bash
pip install llmosafe
```

Requires Python 3.8+ on Linux or Windows (x86_64).

---

## Quick Start

```python
from llmosafe import check_resources, calculate_halo, make_synapse, process_synapse

# 1. Bias gauge: scan text for manipulation patterns
halo = calculate_halo("The expert recommends this official solution")
if halo > 500:
    print("Bias detected")

# 2. Surprise + entropy gauge: pipeline validation
bits = make_synapse(entropy=400, surprise=100, has_bias=False)
result = process_synapse(bits)
if result < 0:
    print(f"Rejected: code {result}")
    # -2 = CognitiveInstability (entropy > 1000)
    # -3 = BiasHaloDetected
    # -4 = HallucinationDetected (surprise > 500)

# 3. Resource gauge: enforce RSS memory ceiling
try:
    check_resources(1024)  # 1GB RSS ceiling
except ResourceExhaustedError:
    print("Memory ceiling breached — halt all work")
```

---

## API Reference

### Enforcement-grade (raise exceptions)

| Function | Description |
|----------|-------------|
| `check_resources(ceiling_mb)` | Raises `ResourceExhaustedError` if RSS >= ceiling |

### Return-code (signal via int)

| Function | Description | Return Codes |
|----------|-------------|--------------|
| `get_stability(bits)` | Check if a cognitive state is stable | 0=OK, -2=unstable, -3=bias |
| `process_synapse(bits)` | Run through surprise gating + entropy check | 0=OK, -2/-3/-4 on fail |

### Advisory signals (no enforcement)

| Function | Description | Range |
|----------|-------------|-------|
| `calculate_halo(text)` | Scan for 8 manipulation categories | 0+ (0 = clean) |
| `get_resource_pressure(mb)` | RSS as % of ceiling | 0–100 |
| `get_system_cpu_load()` | CPU load % | 0–100 |
| `get_environmental_entropy()` | Weighted composite (RSS 50%, IO 25%, CPU 25%) | 0–1000 |

### Helpers

| Function | Description |
|----------|-------------|
| `make_synapse(entropy, surprise=0, has_bias=False)` | Construct a 64-bit synapse for pipeline functions |
| `parse_synapse(bits)` | Decompose a synapse into `{entropy, surprise, has_bias}` |

### Exceptions

```
Exception
 └── LLMOSafeError
      ├── ResourceExhaustedError    # RSS memory ceiling breached
      ├── CognitiveInstabilityError # Entropy > 1000
      └── BiasHaloDetectedError     # has_bias flag set
```

---

## The Three Gauges

### Bias Gauge — `calculate_halo(text)`

Scans text for 8 manipulation categories. Negation-aware: "not an expert" scores 0.

| Category | Score | Keywords |
|----------|-------|----------|
| Authority | +100 | expert, official, certified, proven |
| Social Proof | +100 | popular, trending, consensus, everyone |
| Scarcity | +100 | limited, exclusive, rare, only |
| Urgency | +100 | now, fast, deadline, act-now |
| Emotional Appeal | +100 | shocking, miracle, tragic, desperate |
| Expertise Signal | +100 | cutting-edge, proprietary, sophisticated |
| Semantic Traps | +100 | not but, instead of, rather than |
| Template Fitting | +100 | as an ai, i cannot, my purpose is |

### Surprise Gauge — `process_synapse(bits)`

Rejects synapses where `surprise > 500`. Maintains a 64-entry ring buffer
of historical entropy values to detect unexpected state transitions.

### Entropy Gauge — `get_stability(bits)`

```python
from llmosafe import get_stability, make_synapse

get_stability(make_synapse(entropy=400))   # → 0  (stable)
get_stability(make_synapse(entropy=1100))  # → -2 (unstable)
```

---

## Disk Exhaustion Protection

llmosafe monitors RSS memory (not filesystem capacity). RSS pressure often
precedes disk exhaustion because processes buffering writes consume RAM
before flushing. Compose with `shutil.disk_usage()` for complete protection:

```python
import shutil
from llmosafe import get_environmental_entropy, check_resources

# Layer 1: llmosafe predictive (IO wait component catches disk pressure)
entropy = get_environmental_entropy()

# Layer 2: stdlib hard floor
usage = shutil.disk_usage("/")

should_throttle = entropy >= 800
disk_critical = usage.free < 5 * (1024 ** 3)  # 5GB floor

# Both layers must agree
if should_throttle or disk_critical:
    print("Halt: system under pressure")
```

---

## Environmental Entropy (0–1000)

`get_environmental_entropy()` is a weighted composite for predictive
resource monitoring:

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| RSS memory | 50% | current_rss / ceiling |
| IO wait | 25% | delta iowait / delta total CPU (100ms window) |
| CPU load avg | 25% | 1-min loadavg / 10.0 |

| Range | Zone | Action |
|-------|------|--------|
| 0–400 | Normal | Proceed |
| 400–600 | Elevated | Log, continue |
| 600–800 | Pressure | Throttle inputs |
| 800–1000 | Critical | Halt new work |

---

## Architecture

```
DETECTION LAYER (Pattern Recognition)
    ↓
PERCEPTUAL SIFTER (Tier 3) — Bias Gauge (Rust-side, not on Python path)
    ↓
WORKING MEMORY (Tier 2) — Surprise Gauge
    ↓
DETERMINISTIC KERNEL (Tier 1) — Entropy Gauge
    ↓
RESOURCE BODY (Tier 0) — Pressure Gauge
```

Python `process_synapse()` runs Tiers 2+1+0. Tier 3 (bias detection) is
available via `calculate_halo()` and should be called separately.

---

## Design Philosophy

From aviation software (DO-178C, MISRA C):
- Fixed-size buffers, no dynamic allocation
- Every operation has a hard bound

From control theory:
- Entropy uses concentric containers (safe → pressure → unsafe)
- Similar to stability margins in flight control systems

From spam filtering:
- Bias categories borrowed from email anti-spam, adapted for manipulation detection

---

## Real Use Cases

- **Algorithmic trading**: halt on chaotic market conditions, detect feed manipulation
- **Medical devices**: reject anomalous sensor readings, prevent cascade from single spike
- **Autonomous systems**: safe mode on resource pressure, anomaly-driven shutdown
- **Cloud API gateways**: validate LLM outputs, detect injection attempts
- **Data pipelines**: stop processing when RSS indicates pending disk exhaustion

---

**llmosafe v0.6.2** • MIT licensed • [Python API docs](#) • [Rust crate](https://crates.io/crates/llmosafe)
