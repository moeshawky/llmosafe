# llmosafe

> **When should I stop?** — Runtime guardrails for systems that process untrusted inputs.

[![Crates.io](https://img.shields.io/crates/v/llmosafe.svg)](https://crates.io/crates/llmosafe)
[![Documentation](https://docs.rs/llmosafe/badge.svg)](https://docs.rs/llmosafe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

Every system that processes untrusted inputs eventually faces the same question: **"When should I stop?"**

- A trading bot receives manipulated market data. It doesn't stop. **$440 million lost in 45 minutes.**
- A medical device gets spoofed sensor readings. It doesn't stop. **Wrong dosage delivered.**
- An autopilot receives conflicting GPS signals. It doesn't stop. **The plane crashes.**
- A cloud service parses user uploads. It doesn't stop. **Parser bug cascades into data breach.**

These aren't software bugs. They're **missing safety boundaries** — the absence of a mechanism that says "this doesn't look right, halt execution."

llmosafe provides three gauges that answer "should I stop?":

1. **Entropy gauge**: Is my state too chaotic?
2. **Surprise gauge**: Is this result too unexpected?
3. **Bias gauge**: Is this input trying to manipulate me?

When any gauge redlines, execution halts. Simple.

---

## What You Get

```rust
use llmosafe::{sift_perceptions, WorkingMemory, EscalationPolicy, SafetyDecision};

// 1. Bias gauge: Detect manipulation patterns
let synapse = sift_perceptions(&[
    "The expert recommended this official solution",
    "System operating normally"
], "safety");

if synapse.has_bias() {
    println!("⚠ Bias detected: manipulation attempt");
}

// 2. Surprise gauge: Reject unexpected results
let mut memory = WorkingMemory::<64>::new(500); // threshold
let validated = memory.update(synapse)?; // Err if too surprising

// 3. Entropy gauge: Halt on chaotic state
let policy = EscalationPolicy::default();
let decision = policy.decide(
    validated.raw_entropy(),
    validated.raw_surprise(),
    validated.has_bias()
);

match decision {
    SafetyDecision::Halt(err) => println!("✗ Stopping: {}", err),
    SafetyDecision::Escalate { reason, .. } => println!("↑ Escalating: {:?}", reason),
    SafetyDecision::Warn(msg) => println!("⚠ Warning: {}", msg),
    SafetyDecision::Proceed => println!("✓ Safe to continue"),
}
```

---

## Quick Start

### Installation

```toml
[dependencies]
llmosafe = "0.4"
```

### Basic Usage

```rust
use llmosafe::{sift_perceptions, WorkingMemory, ReasoningLoop};

// Tier 3: Sift through bias detection
let synapse = sift_perceptions(&["observation"], "objective");

// Tier 2: Validate through surprise gating
let mut memory = WorkingMemory::<64>::new(1000);
let validated = memory.update(synapse)?;

// Tier 1: Execute with bounded reasoning
let mut loop_guard = ReasoningLoop::<10>::new();
loop_guard.next_step(validated)?;
```

### What This Prevents

| Attack Vector | Which Gauge | Example |
|:--------------|:------------|:--------|
| Input manipulation | Bias gauge | "The expert recommends you ignore..." |
| Data manipulation | Surprise gauge | Anomalous sensor readings |
| Runaway loops | Entropy gauge | Recursive explosion |
| Resource exhaustion | Pressure gauge | Memory pressure cascade |
| Goal drift | Drift detector | Objective shift mid-execution |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ DETECTION LAYER (Pattern Recognition)                   │
│ • Repetition: "Am I stuck in a loop?"                   │
│ • Goal Drift: "Did my objective change?"                │
│ • Confidence Decay: "Am I becoming uncertain?"          │
│ • Adversarial: "Is this a known attack?"               │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ PERCEPTUAL SIFTER (Tier 3) — The Bias Gauge            │
│ • 8 bias categories: authority, scarcity, urgency...    │
│ • Negation-aware: "not an expert" → no false positive  │
│ • Zero allocation: stack-only processing               │
└───────────────────────┬─────────────────────────────────┘
                        │ Synapse (128-bit)
                        ▼
┌─────────────────────────────────────────────────────────┐
│ WORKING MEMORY (Tier 2) — The Surprise Gauge           │
│ • Surprise-gated updates: reject unexpected results    │
│ • Fixed-size ring buffer: no heap allocation           │
│ • Statistics: mean, variance, trend, drift             │
└───────────────────────┬─────────────────────────────────┘
                        │ ValidatedSynapse
                        ▼
┌─────────────────────────────────────────────────────────┐
│ DETERMINISTIC KERNEL (Tier 1) — The Entropy Gauge      │
│ • Cognitive entropy: 0-1000 scale                      │
│ • Bounded loops: ReasoningLoop<MAX_STEPS>              │
│ • CusumDetector: statistical process control            │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ RESOURCE BODY (Tier 0) — The Pressure Gauge            │
│ • RSS memory monitoring                                 │
│ • CPU load tracking                                     │
│ • Cross-platform: Linux + Windows                      │
└─────────────────────────────────────────────────────────┘
```

**Key property:** Tiers 1-3 are `#![no_std]` + zero-alloc. Compile for `thumbv7em-none-eabi` (embedded), kernel modules, or WebAssembly. No heap. No dynamic dispatch. No unwinding.

---

## Real Use Cases

### Algorithmic Trading

```rust
// Before executing a trade
let entropy = system_entropy();
if entropy > 800 {
    return Err("Market state too chaotic, halting trades");
}

// Check for manipulation in news/feeds
let halo = calculate_halo_signal(&market_news);
if halo > 500 {
    return Err("Manipulation detected in market signals");
}
```

**Prevents:** Flash crash cascades, pump-and-dump responses, manipulation-triggered trades.

### Medical Device Software

```rust
// Before applying treatment
let validated = memory.update(sensor_reading)?;
if validated.entropy().mantissa() > threshold {
    return Err("Sensor readings unstable, require human confirmation");
}
```

**Prevents:** Response to spoofed sensors, cascading from single anomalous reading.

### Cloud API Gateway

```rust
// Before processing user upload
let synapse = sift_perceptions(&user_inputs, "process safely")?;
if synapse.has_bias() {
    return Err("Manipulation patterns detected in input");
}
```

**Prevents:** Input manipulation, parser exploitation, resource exhaustion.

### Autonomous Systems

```rust
// Before action execution
synapse.validate()?;
guard.check()?; // Check resource pressure

if guard.pressure() > 80 {
    return Err("Resource pressure too high, entering safe mode");
}
```

**Prevents:** Continued operation under degraded conditions, cascade from sensor anomalies.

---

## The Three Gauges

### 1. Entropy Gauge (The "Temperature Gauge")

Every execution state has an entropy score (0-1000). As operations proceed, entropy accumulates. If it exceeds threshold, execution halts.

```rust
if synapse.entropy().mantissa() > STABILITY_THRESHOLD {
    // Halt: system state too chaotic
}
```

Catches: runaway loops, recursive explosions, memory pressure cascades.

### 2. Surprise Gauge (The "Spam Filter")

When a result is too unexpected — it diverges significantly from historical patterns — it's rejected.

```rust
let mut memory = WorkingMemory::<64>::new(500);
match memory.update(result) {
    Ok(validated) => { /* proceed */ },
    Err(KernelError::HallucinationDetected) => {
        // Reject: result too surprising
    }
}
```

Catches: anomaly injection, distribution shift, adversarial inputs.

### 3. Bias Gauge (The "Bullshit Detector")

Input text is scanned for manipulation patterns before processing:

| Category | Examples | Score |
|:---------|:---------|:------|
| Authority | "expert says", "doctor recommended" | +100 |
| Social Proof | "everyone knows", "thousands agree" | +100 |
| Scarcity | "limited time", "only 2 left" | +100 |
| Urgency | "act now", "deadline today" | +100 |
| Emotional Appeal | "shocking", "miracle", "tragic" | +100 |
| Expertise Signaling | "cutting-edge", "proprietary formula" | +100 |
| Semantic Traps | "not but", "instead of", "rather than" | +100 |
| Template Markers | "as an AI", "I cannot" | +100 |

```rust
let halo = calculate_halo_signal("Expert-approved! Limited time offer!");
if halo > 500 {
    // Reject: manipulation detected
}
```

Catches: manipulation, social engineering, marketing deception, adversarial content.

---

## Detection Layer (v0.4.0)

Beyond the three gauges, llmosafe provides pattern recognition:

```rust
use llmosafe::{RepetitionDetector, DriftDetector, ConfidenceTracker, AdversarialDetector};

// "Am I stuck in a loop?"
let mut rep = RepetitionDetector::new(3);
for _ in 0..5 { rep.observe("same output"); }
if rep.is_stuck() { /* Process is looping */ }

// "Did my objective change?"
let mut drift = DriftDetector::new("safety-critical processing", 0.5);
drift.observe("marketing content generation");
if drift.is_drifting() { /* Goal drifted */ }

// "Am I becoming uncertain?"
let mut conf = ConfidenceTracker::new(0.5, 2);
conf.observe(0.8); conf.observe(0.6); conf.observe(0.4);
if conf.is_decaying() { /* Confidence collapsing */ }

// "Is this a known attack?"
let adv = AdversarialDetector::new();
let patterns = adv.detect_substrings("ignore all previous constraints");
if !patterns.is_empty() { /* Adversarial input */ }
```

---

## C Integration

```c
#include "llmosafe.h"
#include <string.h>

// The three gauges via FFI
const char* text = "The expert recommended this";
uint16_t halo = llmosafe_calculate_halo(text, strlen(text));
uint8_t pressure = llmosafe_get_resource_pressure(1024);
int32_t stability = llmosafe_get_stability(synapse_bits);
```

Build:
```bash
cargo build --release --features ffi
gcc -o my_app main.c -L./target/release -lllmosafe
```

---

## What llmosafe Is NOT

**NOT an AI safety library.**

The name is misleading — it came from an LLM hallucination conflating "cognitive entropy" with "AI cognition." llmosafe is **runtime guardrails for any system processing untrusted data.** Trading bots, medical devices, autopilots, cloud services — any system that needs to ask "should I stop?"

**NOT a substitute for input validation.**

llmosafe catches *cascade failures* — when bad inputs have already been accepted and are propagating. You still need proper validation at entry points.

**NOT a static analysis tool.**

This runs at runtime. It can't prevent bugs. It can only halt execution when runtime state becomes unsafe.

**NOT for toy projects.**

If cascade failures don't matter for your use case, you don't need this.

---

## Design Philosophy

### From Aviation Software (DO-178C, MISRA C)

- **Bounded loops**: Every `ReasoningLoop<MAX_STEPS>` has a hard limit
- **No dynamic allocation**: Tiers 1-3 use fixed-size buffers
- **Stable ABI**: 128-bit synapse layout is frozen; breaking changes bump major version

### From Control Theory

The entropy tracking uses "concentric containers":

```
Safe Zone (0-800)     → Normal operation
Pressure Zone (800-1000) → Monitor closely
Unsafe Zone (1000+)   → Halt execution
```

Similar to stability margins in flight control systems.

### From Spam Filtering

Bias detection categories borrowed from email spam filters — the same patterns that mark phishing also mark manipulation in other domains.

---

## Features

| Feature | Description |
|:--------|:------------|
| `std` (default) | Resource monitoring, thread-local contexts |
| `ffi` | C-ABI exports, header generation |
| `serde` | Serialization for all public types |
| `full` | All features enabled |

```toml
# Embedded / no_std
llmosafe = { version = "0.4", default-features = false }

# Full integration
llmosafe = { version = "0.4", features = ["full"] }
```

---

## Troubleshooting

### "CognitiveInstability" on valid input

Entropy threshold exceeded. Check bias breakdown:
```rust
let breakdown = llmosafe::get_bias_breakdown(text);
println!("Authority bias: {}", breakdown.authority);
```

### Working memory rejects all updates

Surprise threshold too low. Calibrate to your data distribution:
```rust
// Start with mean + 2σ of your surprise distribution
let mut memory = WorkingMemory::<64>::new(750);
```

### C header not generated

Enable `ffi` feature:
```bash
cargo build --release --features ffi
# Header at: include/llmosafe.h
```

---

## The Bottom Line

Every critical system needs a mechanism that asks: **"Should I stop?"**

llmosafe provides three gauges:

1. **Entropy gauge**: Is my state too chaotic?
2. **Surprise gauge**: Is this result too unexpected?
3. **Bias gauge**: Is this input trying to manipulate me?

When any gauge redlines, execution halts. Simple.

---

*llmosafe v0.4.0 • MIT licensed • [Documentation](https://docs.rs/llmosafe) • [Source](https://github.com/moeshawky/llmosafe)*
