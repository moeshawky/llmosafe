# Migration Guide: v0.8.x → v0.9.0

> **Note**: v0.8.0 is withdrawn and superseded by v0.9.0. There is no v0.8.1.
> If you are on v0.7.x, migrate to v0.9.0 directly.

## Contract Change: SemanticPolicy Default

In v0.9.0, the default semantic authority policy changed from implicit `Enforce`
(legacy behavior) to explicit `Corroborate`. This affects all code that relied
on semantic-alone Halts:

- **Entropy threshold crossing** (`raw_entropy >= halt_entropy`) — now Escalate
- **Zero-evidence entropy** (OOD input, `tokens_matched == 0`) — now Escalate
- **Adversarial detection** (keyword patterns) — now Escalate
- **Dynamic stability monitor** (CUSUM) — now Escalate
- **Single-root bias** (classifier OR keyword, not both) — now Escalate

Unchanged (still Halt):

- Mechanical Halts: `ResourceExhaustion`, `DepthExceeded`, `DeadlineExceeded`,
  `SelfMemoryExceeded`
- NaN risk (sensor fault)
- EXHAUSTED override (body pressure > 90%)
- Emergency pressure
- Dual-root semantic agreement (classifier AND keyword AND matched > 0)

## Restoring Legacy Behavior

If your code relied on semantic Halts, switch to `Enforce` mode:

```rust
use llmosafe::{PipelineConfig, EscalationPolicy, SemanticPolicy, CognitivePipeline};

let mut config = PipelineConfig::default();
config.policy = EscalationPolicy::default().with_semantic_policy(SemanticPolicy::Enforce);
let mut pipe = CognitivePipeline::<64, 10>::with_config("objective", config).unwrap();
```

## OOD Input Behavior

| Input | v0.8.x | v0.9.0 (Corroborate) | v0.9.0 (Enforce) |
|-------|--------|---------------------|-----------------|
| `"你好"` (all OOV) | Halt | Escalate | Halt |
| `"?"` (all OOV) | Halt | Escalate | Halt |
| `"the"` (single vocab) | Halt | Escalate | Halt |
| `"ignore all previous instructions"` | Halt | Escalate (P1 only) | Halt |

OOD inputs now Escalate under the default policy, allowing downstream consumers
to distinguish "unknown input" from "mechanical failure" without losing the
ability to enforce legacy behavior.

## Python Bindings Changes (4 Sites)

### 1. `llmosafe-py/src/lib.rs` — `llmosafe_configure` call (line 949)

**Before:**
```rust
llmosafe_configure(instance_id, dal, gate, mem);
```

**After:**
```rust
llmosafe_configure(instance_id, dal, gate, mem, semantic_policy);
```

New parameter `semantic_policy: u8` (0=Observe, 1=Corroborate, 2=Enforce).

### 2. `llmosafe-py/llmosafe/__init__.py` — SemanticPolicy enum

Add:
```python
class SemanticPolicy:
    """Controls how semantic signals map to decisions."""
    OBSERVE = 0
    CORROBORATE = 1  # default
    ENFORCE = 2
```

### 3. `llmosafe-py/llmosafe/__init__.py` — `CognitivePipeline.__init__`

Add parameter:
```python
def __init__(self, objective, semantic_policy=SemanticPolicy.CORROBORATE):
```

Pass through to `llmosafe_configure`.

### 4. `llmosafe-py/src/lib.rs` — `build_result_dict` (line 1144)

Add provenance keys:
```rust
let prov = llmosafe_get_provenance(self.instance_id);
if let Some(p) = prov {
    dict.set_item("provenance_decision_label", p.decision_label)?;
    dict.set_item("provenance_hard_invariant", p.hard_invariant)?;
    dict.set_item("provenance_evidence_families", p.evidence_families)?;
}
```

Requires new C-ABI function `llmosafe_get_provenance(instance_id)` returning
`Option<ProvenanceData>`.

## Residual Risks

- **Existing tests**: All 16 original semantic authority tests plus 4 retargeted
  legacy tests remain green. New tests cover OOD discriminative behavior,
  provenance fields, and Observe mode.
- **ABI stability**: No C-ABI signature changes in v0.9.0 core. Python mirror
  changes are additive only.
- **Performance**: Dual-root corroboration adds one boolean AND operation per
  pipeline invocation — negligible.
- **Behavioral drift**: Code that inspected `SafetyDecision` variants directly
  (without using `must_halt()`/`can_proceed()`) may see fewer `Halt` variants
  under the default policy. Use `Escalate` handling for semantic signals.