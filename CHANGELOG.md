# Changelog

All notable changes to this project will be documented in this file.

## [0.5.0] - 2026-04-09

### Breaking Changes
- `SafetyDecision::Halt` now has signature `Halt(KernelError, u32)` - second parameter is cooldown_ms
- C-ABI error codes: new codes `-6` (SelfMemoryExceeded), `-7` (DeadlineExceeded)
- Match statements on `SafetyDecision::Halt` must now handle 2-tuple

### Added
- **SafetyDecision::Exit(KernelError)**: Unrecoverable error requiring immediate termination
  - Distinct from Halt (pause+retry) - Exit means abort
  - `should_exit()` helper returns true for Exit variant

- **check_blocking()**: Blocks until resources are safe, honoring cooldowns
  - Automatically respects Escalate/Halt cooldown periods
  - Returns `Err(Exit)` on termination, `Ok(Synapse)` when safe
  - ⚠ BLOCKING: Do not call in async contexts without spawn_blocking

- **check_with_deadline(Instant)**: Same as check_blocking with timeout
  - Returns `Err(DeadlineExceeded)` if deadline passes

- **SafetyDecision helper methods**:
  - `is_blocking()`: Returns true for Escalate, Halt, Exit
  - `should_exit()`: Returns true for Exit only
  - `recommended_cooldown_ms()`: Returns cooldown value (0 for Proceed/Warn/Exit)
  - `status_label()`: Returns machine-readable string ("safe", "warning", "escalate", "halt", "exit")

- **KernelError variants**:
  - `SelfMemoryExceeded`: Daemon's RSS exceeded self-imposed ceiling
  - `DeadlineExceeded`: Timeout while waiting for safe state

- **Escalate variant now has cooldown_ms field**:
  - `Escalate { entropy, reason, cooldown_ms }`
  - Method `recommended_cooldown_ms()` returns this value

### Changed
- All `EscalationPolicy::decide()` methods now populate `cooldown_ms` field
- Default cooldown is `0` (can be tuned per policy)

### Fixed
- C-ABI bindings updated for new KernelError variants
- All pattern matches updated for Halt(KernelError, u32) signature
- Tower middleware example updated for Exit variant

### Migration Guide
```rust
// Before v0.5.0
match decision {
    SafetyDecision::Halt(err) => Err(err),
}

// After v0.5.0
match decision {
    SafetyDecision::Halt(err, _cooldown) => Err(err),
    SafetyDecision::Exit(err) => Err(err),
}
```

## [0.4.9] - 2026-04-09

### Changed
- Updated package metadata (description, keywords) for crates.io
- Version bump from 0.4.2 to 0.4.9

### Note
- This release continues the stable API from v0.4.2
- Bias detection uses keyword matching (no ML model dependencies)
- 89 tests passing
- No breaking changes

## [0.5.0-alpha] - 2026-04-08

### Added
- **MetastabilityCounters**: Per-session event frequency tracking via AtomicU8
  - Tracks: bias_count, entropy_spike_count, cascade_count, surprise_count
  - FFI function: `llmosafe_get_metastability()`
  - Wrapping behavior: 255→0 (safe for frequency counters)

- **Deterministic Reaping**: Session cleanup via FFI
  - FFI function: `llmosafe_reap_stale(current_tick, timeout_ticks)`
  - Returns count of reaped sessions

- **Phrase Matching in Sifter**: 88 new bias detection phrases
  - Authority: "having spent", "decades working", "years in"
  - Social Proof: "what most people", "successful entrepreneurs"
  - Scarcity: "hard to come by", "not widely available"
  - Urgency: "the longer you wait", "window close on"
  - Emotional: "what drives real", "beneath the surface"
  - Expertise: "underlying mechanism", "feedback loops"
  - Semantic Traps: "what appears to be", "contrary to initial"
  - Template: "based on analysis", "what evidence suggests"
  - Phrases score +150 (vs +100 for words)

### Changed
- Phrase matching runs alongside word matching in `get_bias_breakdown`
- No breaking API changes

### Architecture
- Tier 2 trait abstraction prepared for plug-and-play ML model integration
- Supports: mock, local GGUF, remote API, ONNX backends
- Configuration via environment variables

## [0.4.2] - Previous
- Initial release with ContextRegistry, SynapseABI v2, basic sifter
