# llmosafe v0.5.0 Design Decision

**Date**: 2026-04-09
**Author**: GLM5 (nvidia/z-ai/glm5)
**Status**: Approved for Implementation

## Problem Statement

RECOMMENDATIONS.md identified 7 critical issues with llmosafe v0.4.10:

1. Escalate is advisory (can be ignored) → led to RAM DDoS incident
2. Halt loops forever (sleeps 2s, retries) → never actually exits
3. Manual interpretation by callers is error-prone → ixd bugs
4. No self-protection for daemons → daemon becomes the problem
5. Magic sleep values (2000ms) → no rationale
6. Status reporting requires stderr parsing → not machine-readable
7. raw_entropy() returns bitfield → API surprise

## Cross-Domain Insights

Analyzed analogous problems in:
- **Circuit breakers**: State machine with forced transitions, automatic cooldown
- **Traffic lights**: Colors have legal enforcement, not just advisory
- **Radiation safety**: Zones with physical enforcement, not levels
- **USB power negotiation**: Request/grant protocol with back-off timing
- **Elevator safety**: Hardware-enforced interlocks, not just signals

**Key Insight**: The problem is not "severity levels" - it's **ENFORCEMENT MECHANISMS** and **RECOVERY PATHS**.

## Selected Approach

Combine three design patterns:
1. **Exit variant**: Unrecoverable state requiring termination
2. **check_blocking()**: Auto-enforced back-pressure (makes safe path easy)
3. **Self-protection**: Separate ceiling for daemon's own RSS

## Design Principles

### Essential Complexity Preserved
- Different safety levels DO have different semantics (Proceed ≠ Escalate ≠ Halt ≠ Exit)
- Self-monitoring IS hard (daemon watching itself requires separate ceiling)
- Timing IS domain-specific (cooldown depends on entropy level)
- Status reporting IS needed (machine-readable, not stderr parsing)

### Accidental Complexity Removed
- Manual interpretation in callers → replaced with check_blocking()
- Magic sleep values (2000ms) → replaced with recommended_cooldown_ms()
- Daemon self-protection scattered → consolidated in self_ceiling_bytes

### What We Chose NOT to Build
1. DSL for policies (overkill for threshold-based decisions)
2. Async-native methods (spawn_blocking is sufficient, no_std compatible)
3. Config files (caller owns policy construction)
4. Auto-termination (too dangerous, caller must decide)

## API Changes

### SafetyDecision Enum

```rust
pub enum SafetyDecision {
    Proceed,
    Warn(&'static str),
    Escalate {
        entropy: u16,
        reason: EscalationReason,
        cooldown_ms: u32  // NEW
    },
    Halt(KernelError, cooldown_ms: u32),  // CHANGED: added cooldown
    Exit(KernelError),  // NEW: unrecoverable
}

impl SafetyDecision {
    pub fn is_blocking(&self) -> bool;
    pub fn should_exit(&self) -> bool;
    pub fn recommended_cooldown_ms(&self) -> u32;
    pub fn status_label(&self) -> &'static str;
}
```

### ResourceGuard Enhancements

```rust
impl ResourceGuard {
    pub fn new_with_self_limit(memory_ceiling: usize, self_ceiling: usize) -> Self;

    /// Blocks until resources are safe, automatically honoring cooldowns
    /// ⚠ BLOCKING: Use spawn_blocking in async contexts
    pub fn check_blocking(&self) -> Result<Synapse, KernelError>;

    /// Same as check_blocking() but with deadline
    pub fn check_with_deadline(&self, deadline: Instant) -> Result<Synapse, KernelError>;
}
```

### New KernelError Variants

```rust
pub enum KernelError {
    // ... existing variants
    SelfMemoryExceeded,  // Daemon's own RSS exceeded limit
    DeadlineExceeded,    // Timeout waiting for safe state
}
```

## Implementation Priority

| Priority | Feature | Rationale |
|----------|---------|-----------|
| HIGH | Exit variant + recommended_cooldown_ms() | Core safety, prevents infinite loops |
| HIGH | check_blocking() + check_with_deadline() | Enforcement, makes safe path easy |
| MEDIUM | self_ceiling_bytes | Daemon protection, prevents self-DDoS |
| LOW | status_label() | Monitoring convenience |

## Testing Requirements

1. Test check_blocking() with mock time
2. Test self_ceiling_bytes enforcement
3. Test all SafetyDecision transitions
4. Test async compatibility with spawn_blocking
5. Property test cooldown values are bounded

## Migration Guide

### Before (v0.4.10)
```rust
let decision = policy.decide(entropy, surprise, has_bias);
match decision {
    SafetyDecision::Proceed => { /* continue */ }
    SafetyDecision::Halt(err) => {
        std::thread::sleep(Duration::from_millis(2000));  // Magic number!
        // What now? Loop forever?
    }
    _ => {}
}
```

### After (v0.5.0)
```rust
let synapse = guard.check_blocking().map_err(|e| {
    eprintln!("Cannot continue safely: {:?}", e);
    std::process::exit(1);
})?;
// One function call. Safe by default. No interpretation bugs.
```

## Version Compatibility

- **Breaking change**: New Exit variant, Halt signature changed
- **Requires**: v0.5.0 (semver major version bump)
- **Timeline**: Implement in v0.5.0 branch, merge after testing

## Success Criteria

- [ ] All RECOMMENDATIONS.md issues addressed
- [ ] No new heap allocations in core types
- [ ] no_std compatibility preserved
- [ ] Async compatibility documented
- [ ] 100% test coverage on new code paths
- [ ] Clippy clean with warnings as errors
