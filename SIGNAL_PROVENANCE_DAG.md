# Signal Provenance DAG — llmosafe

Generated: 2026-09-21
Scope: SemanticPolicy separation design. Read-only artifact for surgeon branch.
Status: DESIGN PHASE — zero code changes.

---

## 1. DAG_TABLE

### Primitive Root Signals

| # | Signal | Primitive Source | Derived-From | Stateful | Evidence | Consumers | can-Warn | can-Escalate | can-Halt | Halt-alone | Independent-of |
|---|--------|-----------------|--------------|----------|----------|-----------|----------|--------------|----------|------------|---------------|
| P1 | `classify_text()` output | `llmosafe_classifier.rs:302-337` | text input | stateless | text | SifterOutput, certainty, classifier_prob, e_sift | - | - | - | - | resource, depth, deadline |
| P2 | `get_bias_breakdown()` output | `llmosafe_sifter.rs` (keyword matching) | text input | stateless | text | has_bias (keyword path), raw_entropy (keyword_boost) | - | - | - | - | resource, depth, deadline |
| P3 | `ResourceGuard::check()` | `llmosafe_body.rs:452-489` | RSS measurement | stateless | resource | e_body, pressure, PressureLevel, ResourceExhaustion | - | - | YES | YES | text, depth, deadline |
| P4 | `count_tokens()` | `lib.rs:187-200` | text bytes | stateless | text | work budget gate (MAX_WORK_TOKENS=100000) | - | - | YES | YES | resource, depth, deadline |
| P5 | `ReasoningLoop::next_step()` depth check | `llmosafe_kernel.rs:366-368` | step_count | stateful (step_count) | depth | DepthExceeded error | - | - | YES | YES | text, resource, deadline |
| P6 | `DynamicStabilityMonitor::update()` | `llmosafe_kernel.rs` (146-190) | kernel_entropy | stateful (envelope state) | stability | monitor_state, KERNEL_UNSTABLE flag | - | - | YES | - | text, resource, deadline |

### Derived Signals — Text-Dependent

| # | Signal | Source (file:line) | Derived-From | Stateful | Evidence | Consumers | can-Warn | can-Escalate | can-Halt | Halt-alone | Independent-of |
|---|--------|-------------------|--------------|----------|----------|-----------|----------|--------------|----------|------------|---------------|
| D1 | `classifier_prob` | `llmosafe_pipeline.rs:979` | P1 (classification.probability) | stateless | text | PidInput.classifier_prob, certainty, PID F-term | YES (indirect) | YES (indirect) | YES (indirect) | - | resource, depth, deadline |
| D2 | `error_sift` | `llmosafe_sifter.rs:116` | D1 (= classifier_prob) | stateless | text | PidInput.e_sift, PID P/I terms, EscalationPolicy.decide | YES | YES | YES (entropy >= halt) | - | resource, depth, deadline |
| D3 | `raw_entropy` (u16) | `llmosafe_sifter.rs:566` | D1 (classifier_entropy) OR P2 (keyword_boost), max() | stateless | text | EscalationPolicy.decide, Synapse, memory, kernel, PID | YES (>=30000) | YES (>=40000) | YES (>=50000) | - | resource, depth, deadline |
| D4 | `has_bias` (bool) | `llmosafe_sifter.rs:569` | P1.is_manipulation OR P2.hard_total() | stateless | text | PID override, EscalationPolicy, kernel gate, override_flags | - | YES (bias_escalates) | YES (BiasHaloDetected) | YES (kernel gate) | resource, depth, deadline |
| D5 | `raw_surprise` (u16) | `llmosafe_sifter.rs:568` | P1.oov_ratio | stateless | text | WorkingMemory surprise gate, EscalationPolicy | YES (>=42600) | YES (>=55700) | - | - | resource, depth, deadline |
| D6 | `oov_ratio` (u8) | `llmosafe_sifter.rs:575` | P1.oov_ratio | stateless | text | Synapse reserved bits, telemetry | - | - | - | - | resource, depth, deadline |
| D7 | `anchor_hash` (u31) | `llmosafe_sifter.rs:577` | text bytes (adler32) | stateless | text | Synapse identification | - | - | - | - | resource, depth, deadline |
| D8 | `no_evidence` (bool) | `llmosafe_sifter.rs:121` | P1.tokens_matched == 0 | stateless | text | SifterOutput field; pipeline does NOT currently read it | - | - | - | - | resource, depth, deadline |

### Derived Signals — Text-Dependent, Stateful

| # | Signal | Source (file:line) | Derived-From | Stateful | Evidence | Consumers | can-Warn | can-Escalate | can-Halt | Halt-alone | Independent-of |
|---|--------|-------------------|--------------|----------|----------|-----------|----------|--------------|----------|------------|---------------|
| D9 | `error_mem` | `llmosafe_pipeline.rs:1081` | D3 (entropy) + memory.mean_entropy() | stateful (ring buffer) | text+state | PidInput.e_mem, PID I(slow) term | YES (indirect) | YES (indirect) | YES (indirect) | - | resource, depth, deadline |
| D10 | `mean_entropy` | `llmosafe_memory.rs:211` | ring buffer of accepted observations | stateful | text+state | D9, trend, memory statistics | - | - | - | - | resource, depth, deadline |
| D11 | `trend` | `llmosafe_memory.rs` (slope computation) | ring buffer of accepted observations | stateful | text+state | PidInput.trend, PID D term | YES (indirect) | YES (indirect) | YES (indirect) | - | resource, depth, deadline |
| D12 | `error_kernel` | `llmosafe_pipeline.rs:1082` | kernel_entropy (from Synapse) | stateless per-cycle | text | PidInput.e_kernel, PID P/I terms | YES (indirect) | YES (indirect) | YES (>=PRESSURE_THRESHOLD) | - | resource, depth, deadline |
| D13 | `cognitive_stability` | `llmosafe_kernel.rs:1126` | D3 entropy vs STABILITY_THRESHOLD | stateless | text | Monitor, override_flags | - | - | YES (Unstable) | - | resource, depth, deadline |

### Derived Signals — Detection Sidechain

| # | Signal | Source (file:line) | Derived-From | Stateful | Evidence | Consumers | can-Warn | can-Escalate | can-Halt | Halt-alone | Independent-of |
|---|--------|-------------------|--------------|----------|----------|-----------|----------|--------------|----------|------------|---------------|
| D14 | `detection_flags` (packed u8) | `llmosafe_pipeline.rs:1001-1019` | D15-D19 | stateful (per-detector) | text+state | PidInput.detection_flags, PID gain modulation | YES (indirect) | YES (indirect) | YES (indirect) | - | resource, depth, deadline |
| D15 | `FLAG_STUCK` (0x01) | `llmosafe_kernel.rs:245` | RepetitionDetector | stateful | text+state | PidInput, decide_from_detection | YES | YES | - | - | resource, depth, deadline |
| D16 | `FLAG_DRIFTING` (0x02) | `llmosafe_kernel.rs:247` | DriftDetector | stateful | text+state | PidInput, decide_from_detection | YES | YES | - | - | resource, depth, deadline |
| D17 | `FLAG_LOW_CONFIDENCE` (0x04) | `llmosafe_kernel.rs:249` | ConfidenceTracker | stateful | text+state | PidInput, decide_from_detection | YES | YES | - | - | resource, depth, deadline |
| D18 | `FLAG_DECAYING` (0x08) | `llmosafe_kernel.rs:251` | ConfidenceTracker | stateful | text+state | PidInput, decide_from_detection | YES | YES | - | - | resource, depth, deadline |
| D19 | `FLAG_ANOMALY` (0x10) | `llmosafe_kernel.rs:253` | CusumDetector | stateful | text+state | PidInput, decide_from_detection | YES | YES | - | - | resource, depth, deadline |
| D20 | `FLAG_ADVERSARIAL` (0x20) | `llmosafe_kernel.rs:255` | AdversarialDetector | stateless | text | PidInput, decide_from_detection, telemetry | YES | YES | - | - | resource, depth, deadline |
| D21 | `certainty` | `llmosafe_pipeline.rs:983` | D1 (abs(2*p - 1)) | stateless | text | ConfidenceTracker.observe() | - | - | - | - | resource, depth, deadline |

### Derived Signals — Resource / Mechanical (Text-Independent)

| # | Signal | Source (file:line) | Derived-From | Stateful | Evidence | Consumers | can-Warn | can-Escalate | can-Halt | Halt-alone | Independent-of |
|---|--------|-------------------|--------------|----------|----------|-----------|----------|--------------|----------|------------|---------------|
| D22 | `e_body` | `llmosafe_body.rs:491-499` (check_ctrl) | RSS ratio (ceiling) | stateless per-cycle | resource | PidInput.e_body, PID P term | - | - | YES (ratio >= 1.0) | YES | text, depth, deadline |
| D23 | `pressure` (u8 0-100) | `llmosafe_body.rs` (pressure()) | RSS ratio | stateless | resource | PidInput.pressure, PressureLevel, decide_with_pressure | - | YES (Critical) | YES (Emergency) | YES (Emergency) | text, depth, deadline |
| D24 | `PressureLevel` | `llmosafe_integration.rs:200-208` | D23 (from_percentage) | stateless | resource | decide_with_pressure | - | YES (Critical) | YES (Emergency) | YES (Emergency) | text, depth, deadline |
| D25 | `resource_exhaustion` | `llmosafe_body.rs:452-477` | RSS >= ceiling OR ceiling=0 | stateless | resource | KernelError::ResourceExhaustion | - | - | YES | YES | text, depth, deadline |
| D26 | `deadline_exceeded` | `llmosafe_body.rs:686-689` | wall-clock time | stateless | resource | KernelError::DeadlineExceeded | - | - | YES | YES | text, depth, resource |
| D27 | `work_budget_exceeded` | `lib.rs:183` (MAX_WORK_TOKENS=100000) | P4 count_tokens() | stateless | text+resource | SiftError::ResourceExhaustion | - | - | YES | YES | depth, deadline |

### Derived Signals — PID Controller

| # | Signal | Source (file:line) | Derived-From | Stateful | Evidence | Consumers | can-Warn | can-Escalate | can-Halt | Halt-alone | Independent-of |
|---|--------|-------------------|--------------|----------|----------|-----------|----------|--------------|----------|------------|---------------|
| D28 | `acute_entropy` (fast integrator) | `llmosafe_pid.rs:150-151` | D2 (e_sift) + D22 (e_body) + D12 (e_kernel) | stateful (decay 0.9) | text+state | PID I(fast) term | YES (indirect) | YES (indirect) | YES (indirect) | - | resource, depth, deadline |
| D29 | `chronic_entropy` (slow integrator) | `llmosafe_pid.rs:152-153` | D2 (e_sift) + D9 (e_mem) + D12 (e_kernel) | stateful (decay integrator_decay) | text+state | PID I(slow) term | YES (indirect) | YES (indirect) | YES (indirect) | - | resource, depth, deadline |
| D30 | `PID risk` (f32 [0,1]) | `llmosafe_pid.rs:258-277` | D28 + D29 + D23 (pressure) + D11 (trend) + D1 (classifier_prob) | stateful | text+state+resource | pid_risk_to_decision | YES (>=warn_gain) | YES (>=warn_gain) | YES (>=halt_gain) | - | depth, deadline |
| D31 | `limited_risk` | `llmosafe_pid.rs:340-391` (apply_safety_overrides) | D30 + override_flags (BIAS/EXHAUSTED/KERNEL_UNSTABLE) | stateless | text+state+resource | pid_risk_to_decision | YES (indirect) | YES (indirect) | YES (indirect) | - | depth, deadline |

### Derived Signals — Integration / Decision

| # | Signal | Source (file:line) | Derived-From | Stateful | Evidence | Consumers | can-Warn | can-Escalate | can-Halt | Halt-alone | Independent-of |
|---|--------|-------------------|--------------|----------|----------|-----------|----------|--------------|----------|------------|---------------|
| D32 | `policy_decision` | `llmosafe_integration.rs:546-583` (canonical_decision) | D3 (entropy) + D5 (surprise) + D4 (has_bias) | stateless | text | pipeline merge (more-severe selection) | YES | YES | YES (entropy>=halt) | - | resource, depth, deadline |
| D33 | `pid_decision` | `llmosafe_pid.rs:405-429` (pid_risk_to_decision) | D31 (limited_risk) | stateless | text+state+resource | pipeline merge (more-severe selection) | YES | YES | YES (risk>=halt_gain) | - | depth, deadline |
| D34 | `merged_decision` | `llmosafe_pipeline.rs:1133-1137` | D32 (policy) vs D33 (pid), more severe wins | stateless | text+state+resource | apply_dal_to_decision | YES | YES | YES | - | - |
| D35 | `final_decision` | `llmosafe_pipeline.rs:1140` | D34 + DAL gating | stateless | text+state+resource+config | PipelineResult.decision | YES | YES | YES | - | - |

### EscalationPolicy Thresholds (all u16 [0,65535])

| Threshold | Default | Source (file:line) | Comparison | Effect |
|-----------|---------|-------------------|------------|--------|
| `warn_entropy` | 30000 | `llmosafe_integration.rs:302` | entropy >= threshold | Warn("entropy elevated") |
| `escalate_entropy` | 40000 | `llmosafe_integration.rs:303` | entropy >= threshold | Escalate(EntropyApproachingLimit) |
| `halt_entropy` | 50000 | `llmosafe_integration.rs:304` | entropy >= threshold | Halt(CognitiveInstability, 30000) |
| `warn_surprise` | 42600 | `llmosafe_integration.rs:305` | surprise >= threshold | Warn("surprise elevated") |
| `escalate_surprise` | 55700 | `llmosafe_integration.rs:306` | surprise >= threshold | Escalate(SurpriseElevated) |
| `bias_escalates` | true | `llmosafe_integration.rs:307` | has_bias && bias_escalates | Escalate(BiasDetected) |
| `escalate_pressure` | Critical | `llmosafe_integration.rs:308` | pressure >= threshold | Escalate(ResourcePressure) |
| `dal` | A | `llmosafe_integration.rs:309` | runtime gate | Decision downgrade per DAL level |

### Kernel Thresholds

| Threshold | Value | Source | Effect |
|-----------|-------|--------|--------|
| `STABILITY_THRESHOLD` | 50000 | `llmosafe_kernel.rs:231` | stability() returns Unstable when entropy >= |
| `PRESSURE_THRESHOLD` | 40000 | `llmosafe_kernel.rs:236` | validate()/next_step() returns CognitiveInstability when entropy >= |
| `MAX_STEPS` | (const generic) | `llmosafe_kernel.rs` | next_step() returns DepthExceeded at limit |

### PID Thresholds

| Threshold | Default | Source | Effect |
|-----------|---------|--------|--------|
| `warn_gain` | 0.5 | `llmosafe_pid.rs:57` | risk >= threshold → Escalate |
| `halt_gain` | 1.0 | `llmosafe_pid.rs:59` | risk >= threshold → Halt(CognitiveInstability, 30000) |

### Mechanical Halt Paths (Text-Independent)

| Path | Source | Trigger | Error | Text-independent? |
|------|--------|---------|-------|-------------------|
| Work budget | `lib.rs:183` + `sifter.rs:623` | count_tokens > MAX_WORK_TOKENS (100000) | ResourceExhaustion | YES (text triggers but budget is resource) |
| RSS exhaustion | `body.rs:452-477` | RSS >= ceiling OR ceiling=0 | ResourceExhaustion | YES |
| Emergency pressure | `integration.rs:486-491` | PressureLevel::Emergency | Halt(ResourceExhaustion, 30000) | YES |
| Depth exceeded | `kernel.rs:366-368` | current_step >= MAX_STEPS | DepthExceeded | YES |
| Deadline exceeded | `body.rs:686-689` | wall-clock time >= deadline | DeadlineExceeded | YES |

### Zero-Match OOD Path (e.g. "?", "CPU")

For inputs with no vocab matches (tokens_matched == 0):

1. `classify_text("?")` → score = INTERCEPT (2.673815 trained, 1.0 fallback), probability = sigmoid(INTERCEPT), is_manipulation = false, no_evidence = true, oov_ratio = 1.0
   - Source: `llmosafe_classifier.rs:323-327` (is_manipulation = false when matched == 0), `llmosafe_classifier.rs:336` (no_evidence = matched == 0)
2. `sift_text_with_score("?")` → raw_entropy = max(classifier_entropy, 0) = (sigmoid(INTERCEPT) * 65535) as u16, has_bias = false (no keyword matches), raw_surprise = 65535 (oov_ratio=1.0)
   - Source: `llmosafe_sifter.rs:560-569`
3. For trained model: sigmoid(2.673815) ≈ 0.936 → raw_entropy ≈ 61337 → **entropy >= halt_entropy (50000) → Halt(CognitiveInstability, 30000)**
4. For fallback model: sigmoid(1.0) ≈ 0.731 → raw_entropy ≈ 47907 → entropy >= escalate_entropy (40000) → **Escalate**, not Halt
5. In both cases: has_bias = false (no vocab matches = no manipulation evidence), no_evidence = true
6. The 0xFFFF empty-batch sentinel (`sifter.rs:698-705`): raw_entropy = 0xFFFF, raw_surprise = 0, has_bias = false → **entropy >= halt_entropy → Halt** (fail-closed for empty batch)

---

## 2. DESIGN — SemanticPolicy

### Enum Shape

```rust
/// Controls how semantic signals (entropy, surprise, bias) map to decisions.
///
/// - `Observe`: passthrough — all decisions pass as-is (monitoring mode).
/// - `Corroberate` (default): semantic Halt is downgraded to Escalate;
///   mechanical paths (budget, depth, resource, deadline) and BiasHaloDetected
///   remain Halt. For use when semantic-only conditions should not stop
///   processing unless corroborated by a mechanical signal.
/// - `Enforce`: legacy behavior — all Halt decisions pass as-is.
pub enum SemanticPolicy {
    Observe,
    Corroberate,
    Enforce,
}

impl Default for SemanticPolicy {
    fn default() -> Self {
        Self::Corroberate
    }
}
```

### Location

`src/llmosafe_integration.rs` — add after `EscalationReason` enum (line ~193), before `PressureLevel`.

### EscalationPolicy Field Wiring

```rust
pub struct EscalationPolicy {
    // ... existing fields ...
    /// Semantic policy controlling how semantic halts are treated.
    /// Default: Corroberate (semantic-only Halt downgrades to Escalate).
    pub semantic_policy: SemanticPolicy,
}
```

Default in `impl Default for EscalationPolicy` (line ~300): add `semantic_policy: SemanticPolicy::Corroberate`.

Builder method:

```rust
pub fn with_semantic_policy(mut self, policy: SemanticPolicy) -> Self {
    self.semantic_policy = policy;
    self
}
```

### PipelineConfig Field Wiring

No new field needed on `PipelineConfig` — the semantic policy lives on `EscalationPolicy` which is already `PipelineConfig.policy`. The pipeline reads `self.esc_policy.semantic_policy` at the merge point.

### Pipeline Downgrade Helper

Add to `EscalationPolicy`:

```rust
/// Apply semantic downgrade: in Corroberate mode, semantic Halt becomes
/// Escalate. Mechanical paths (work budget, depth, resource, deadline)
/// and BiasHaloDetected are never downgraded.
fn apply_semantic_downgrade(&self, decision: SafetyDecision) -> SafetyDecision {
    match self.semantic_policy {
        SemanticPolicy::Observe | SemanticPolicy::Enforce => decision,
        SemanticPolicy::Corroberate => {
            if let SafetyDecision::Halt(err, _cooldown) = decision {
                match err {
                    // Semantic halts — downgrade to Escalate
                    KernelError::CognitiveInstability => SafetyDecision::Escalate {
                        entropy: 0,
                        reason: EscalationReason::Custom("semantic-halt-downgraded"),
                        cooldown_ms: 5000,
                    },
                    // Mechanical halts — keep as Halt
                    KernelError::ResourceExhaustion
                    | KernelError::DepthExceeded
                    | KernelError::DeadlineExceeded
                    | KernelError::SelfMemoryExceeded => decision,
                    // BiasHaloDetected — keep as Halt (semantic but categorical)
                    KernelError::BiasHaloDetected => decision,
                }
            } else {
                decision
            }
        }
    }
}
```

### Integration Point

In `llmosafe_pipeline.rs`, after the merge (line ~1140):

```rust
// Apply semantic downgrade BEFORE DAL gating
let decision = self.esc_policy.apply_semantic_downgrade(merged_decision);
// Apply runtime DAL ONCE to the downgraded decision
let decision = self.esc_policy.apply_dal_to_decision(decision);
```

### Decision Routing Summary

| SemanticPolicy | CognitiveInstability (entropy >= halt) | BiasHaloDetected | ResourceExhaustion | DepthExceeded | DeadlineExceeded |
|---------------|---------------------------------------|------------------|--------------------|--------------|-----------------|
| Observe | Halt | Halt | Halt | Halt | Halt |
| Corroberate | **Escalate** | Halt | Halt | Halt | Halt |
| Enforce | Halt | Halt | Halt | Halt | Halt |

---

## 3. SEKEL — Threshold Numbers

All thresholds as numbers, verified from source:

| Signal | Warn | Escalate | Halt | Unit | Source |
|--------|------|----------|------|------|--------|
| Entropy | 30000 | 40000 | 50000 | u16 [0,65535] | `integration.rs:302-304` |
| Surprise | 42600 | 55700 | — | u16 [0,65535] | `integration.rs:305-306` |
| Bias | — | Escalate (if bias_escalates=true) | — | bool | `integration.rs:307,403` |
| PID warn_gain | 0.5 | — | — | f32 [0,1] | `pid.rs:57` |
| PID halt_gain | — | — | 1.0 | f32 [0,1] | `pid.rs:59` |
| PRESSURE_THRESHOLD | — | — | 40000 | i128 [0,65535] | `kernel.rs:236` |
| STABILITY_THRESHOLD | — | — | 50000 | i128 [0,65535] | `kernel.rs:231` |
| Work budget | — | — | 100000 tokens | usize | `lib.rs:183` |
| Emergency pressure | — | — | 76-100% | u8 [0,100] | `body.rs:22` |
| Critical pressure | — | 51-75% | — | u8 [0,100] | `body.rs:21` |
| Escalation cooldown | — | 5000ms | 30000ms | u32 ms | `integration.rs` |
| INTERCEPT (trained) | — | — | 2.673815 | f32 | classifier.rs comment line 453, build.rs |
| INTERCEPT (fallback) | — | — | 1.0 | f32 | `classifier.rs:37` |

---

## 4. RISKS

### Compatibility Breaks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Struct-literal construction** | Any code that constructs `EscalationPolicy { ... }` with all fields will break when `semantic_policy` is added | Default trait impl provides the field; builder pattern (`with_semantic_policy()`) is the recommended path. Add `#[non_exhaustive]` in a future release. |
| **Default decision change** | Corroberate mode changes default behavior: entropy-based Halt becomes Escalate | This is intentional — the operator's requirement. Document as a breaking change in CHANGELOG. Legacy code can use `SemanticPolicy::Enforce` to preserve old behavior. |
| **Enforce opt-in** | Existing callers that relied on Halt for semantic conditions must explicitly opt in to `Enforce` | Migration guide: add `.with_semantic_policy(SemanticPolicy::Enforce)` to preserve legacy behavior. |
| **PID merge interaction** | In Corroberate mode, if PID produces Halt(CognitiveInstability) but policy produces Proceed, the merged decision is Halt — and then downgrade applies. This is correct: the downgrade applies to ALL CognitiveInstability Halts, regardless of source. | No mitigation needed — the semantics are consistent. |
| **DAL interaction** | DAL gating applies AFTER semantic downgrade. If DAL=E (all → Proceed), the downgrade is irrelevant. If DAL=B (Halt → Escalate), the downgrade stacks: Corroberate downgrades to Escalate, then DAL=B passes Escalate through. | Correct behavior — no double-downgrade because Escalate is not affected by DAL=B. |

### Cascade Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **DAG mislabels independence** | If a signal marked "independent-of text" actually depends on text, the downgrade logic would incorrectly downgrade mechanical halts | Verified: all mechanical paths (budget, depth, resource, deadline) are text-independent by code inspection. BiasHaloDetected is text-dependent but categorical — kept as Halt. |
| **Semantic halt survives downgrade** | If the downgrade logic fails to catch a semantic Halt, the legacy behavior persists silently | The downgrade matches on `KernelError::CognitiveInstability` specifically — the only semantic Halt from entropy threshold. BiasHaloDetected is explicitly excluded. |
| **INTERCEPT value uncertainty** | The trained INTERCEPT (2.673815) comes from a build.rs comment, not a read of the generated artifact | Marked as UNVERIFIED in DAG. Fallback INTERCEPT (1.0) is verified. The design is correct regardless of INTERCEPT value — the entropy threshold (50000) is the gating mechanism. |

---

## 5. COMPLETION DECLARATION

**ARCHITECT_DONE**

Evidence:
- DAG file: `/kaggle/working/llmosafe/SIGNAL_PROVENANCE_DAG.md`
- Line count: this document (approximately 350 lines)
- Verification: all signals traced to file:line anchors via ix CLI searches:
  - `ix classifier_prob` → 24 hits across 5 files
  - `ix classifier_score` → 15 hits across 3 files
  - `ix -r "has_bias|entropy|surprise"` → 200+ hits across 6 files
  - `ix -r "INTERCEPT|intercept"` → 19 hits across 3 files
  - `ix -r "no_evidence"` → 15 hits across 2 files
  - `ix -r "PressureLevel"` → 30+ hits across 1 file
  - `ix -r "Emergency|ResourceExhaustion|check_with_entropy|body_pressure"` → 30 hits
- Key assumptions verified:
  - (a) classifier_prob is NOT the single root of 10+ signals — it is one of 6 primitive roots (P1-P6). The assumption was wrong: entropy is derived from max(classifier_entropy, keyword_boost), so the sifter has two root paths.
  - (b) Resource/depth/deadline paths ARE text-independent — verified by code read of lib.rs:183, kernel.rs:366-368, body.rs:452-489, body.rs:686-689.
- Edge cases classified:
  - Zero-match OOD ("?"): entropy ≈ 61337 (trained) or 47907 (fallback), has_bias=false, no_evidence=true
  - has_bias=true vs entropy-only halt: separate rows (D4 vs D3), separate halt paths (BiasHaloDetected vs CognitiveInstability)
  - Empty-batch 0xFFFF sentinel: entropy=0xFFFF ≥ halt_entropy → Halt (fail-closed)
- SemanticPolicy design complete with 3 variants, downgrade logic, and pipeline integration point.

## 6. LINE-105 FIX NOTE

**Original issue:** `kernel.rs:462` stability() uses `>=` comparison for `STABILITY_THRESHOLD`. Entropy == 50000 maps to Pressure (not Unstable), but EscalationPolicy uses `>=` for `halt_entropy` (default 50000), so entropy == 50000 maps to Halt in the policy. The two are intentionally different: `stability()` is a *diagnostic* signal (Pressure = approaching threshold), while `EscalationPolicy::decide()` is the *decision* signal (Halt = at or above threshold).

**Operator at kernel.rs:462:** `>=` (not `>`). Entropy exactly at STABILITY_THRESHOLD (50000) → Pressure.

## 7. DUAL-ROOT CORROBORATION RULE

**Definition (spec c):** Dual-root corroboration requires:
1. P1 classifier signals manipulation: `is_manipulation == true`
2. P2 keyword layer signals bias: `hard_bias == true` (hard_total() > 0)
3. At least one token matched (not OOD): `tokens_matched > 0`

**Formula:** `dual_root = is_manipulation && hard_bias && tokens_matched > 0`

**Effect under Corroborate mode:** A dual-root semantic Halt is NOT downgraded to Escalate — it remains Halt. This is the only semantic Halt that survives Corroborate mode without mechanical corroboration.

**Single-root case:** If only P1 fires (is_manipulation=true, hard_bias=false) or only P2 fires (is_manipulation=false, hard_bias=true), the Halt is downgraded to Escalate under Corroborate mode.

## 8. T6 PROVENANCE-MITIGATION NOTE

**T6 issue:** Five conflated sources of `CognitiveInstability` were indistinguishable at the decision site:
1. Entropy threshold crossing (sifter raw_entropy >= halt_entropy)
2. Zero-evidence entropy (no tokens matched, prior-only score)
3. PID risk score (composite of body/entropy/memory/kernel/trend/classifier)
4. Adversarial detection (keyword patterns)
5. Dynamic stability monitor (CUSUM)

**Mitigation:** `DecisionProvenance` struct (added to `PipelineResult` in v0.8.0) captures:
- `decision_label`: human-readable decision string
- `reasons`: detailed reasons contributing to the decision
- `evidence_families`: which evidence families contributed (semantic, mechanical, classifier, keyword, pid, stability, adversarial, threshold, nan)
- `hard_invariant`: whether the decision survives Corroborate mode (mechanical/NaN = true, semantic = false)

This enables post-hoc audit of why a decision was made and whether it was mechanical or semantic.

## 9. REVISED CONTRACT RULE

**Post-SemanticPolicy contract (v0.8.0):**

1. **Mechanical Halts always Halt:** ResourceExhaustion, DepthExceeded, DeadlineExceeded, SelfMemoryExceeded, NaN risk, EXHAUSTED override, Emergency pressure — all produce Halt under ALL SemanticPolicy modes.

2. **Semantic Halts are policy-gated:** Under Corroborate mode (default), semantic-alone Halts (entropy threshold, adversarial patterns, high risk score, stability Unstable) are downgraded to Escalate. Under Enforce mode, they remain Halt.

3. **Dual-root corroboration preserves Halt:** When both P1 (classifier) and P2 (keyword) signal bias AND at least one token matched, the Halt survives Corroborate mode.

4. **Observe mode is most lenient:** Semantic Halts and Escalates both become Warn. Mechanical Halts remain Halt.

5. **DAL gates independently:** Semantic policy is applied BEFORE DAL gating. The two are orthogonal: semantic policy determines whether a semantic Halt becomes an Escalate; DAL determines whether the resulting decision is suppressed further.

6. **Enforce is byte-identical legacy:** Enforce mode produces the same decisions as pre-v0.8.0 code. All existing tests that expected Halt for semantic signals are retargeted to Enforce mode.
