# llmosafe v0.5.x Recommendations

## 1. Escalate Must Be Binding

Currently `SafetyDecision::Escalate` is advisory — the caller can ignore it and proceed.

```rust
SafetyDecision::Escalate { entropy, reason } → ixd "throttles" 500ms → still calls builder.update()
```

Problem: This is how ixd caused RAM DDoS. llmosafe correctly detected high memory, but the daemon's handling made things worse.

**Recommendation**: Add `SafetyDecision::is_blocking() -> bool`. When true, the caller MUST stop. If they call `builder.update()` anyway, it's a programming error (caught by clippy lint or runtime debug_assert). Or add `check_strict()` that returns `Result<SafetyDecision, UnhandledEscalation>` forcing compile-time enforcement.

**Simpler approach**: `EscalationPolicy::decide_strict()` that fails to compile if the caller doesn't match all variants and handle Escalate with a `continue`.

## 2. Halt Should Exit, Not Sleep

Currently:
```rust
SafetyDecision::Halt(err) → sleep(2000ms) → continue
```

Problem: Halt loops forever if memory stays high. The daemon never exits — it just pauses 2s forever, consuming resources to check memory, never actually recovering.

**Recommendation**: Add `SafetyDecision::should_exit() -> bool` or a variant `SafetyDecision::Exit(KernelError)` that unambiguously means "done, please terminate cleanly". Halt should mean "pause AND assess". Exit means "you cannot continue safely, exit now".

## 3. ResourceGuard::check_blocking() — Self-Throttling Guard

Currently ixd manually interprets SafetyDecision. This is error-prone and why the Escalate bug happened.

**Recommendation**:
```rust
impl ResourceGuard {
    /// Blocks until memory is safe, then returns.
    /// Automatically honors Halt/Escalate by waiting (not looping).
    pub fn check_blocking(&self) -> Result<Synapse, KernelError>;

    /// Same but with a timeout — returns Err if can't proceed within duration.
    pub fn check_with_deadline(&self, deadline: Instant) -> Result<Synapse, KernelError>;
}
```

With `check_blocking()`, ixd calls ONE function and gets automatic back-pressure:
```rust
let synapse = guard.check_blocking().map_err(|e| {
    eprintln!("ixd: cannot continue safely: {:?}", e);
    std::process::exit(1);
})?;
```

## 4. Built-In Self-RSS Limit for Daemons

ixd starts a build, memory fills, daemon **becomes** the problem. Currently no protection against this.

**Recommendation**:
```rust
ResourceGuard::auto_with_self_limit(ceiling: f32, self_ceiling_bytes: u64)
```

When `check()` sees YOUR RSS > self_ceiling_bytes, it returns `Halt(KernelError::SelfMemoryExceeded)` regardless of system memory state. This protects against the daemon eating itself.

## 5. Mandatory Grace Period / Recommended Cooldown

Currently ixd hardcodes `sleep(2000ms)` for Halt with no rationale for the 2000 number.

**Recommendation**: `SafetyDecision` should carry `recommended_cooldown_ms() -> u32`. ixd calls:
```rust
std::thread::sleep(Duration::from_millis(decision.recommended_cooldown()));
```
No more magic numbers in ixd. llmosafe owns the timing wisdom.

## 6. Beacon/Status Reporting Integration

ixd writes to `beacon.json` to report status to external monitors. This is how users/other tools know what's happening.

**Recommendation**: Add `SafetyDecision::status_label() -> &'static str` so monitoring tools can read "safety halt" without parsing stderr. Or a `SafetyDecision::write_beacon(path: &Path)` that atomically writes a machine-readable status file.

## 7. Errata: raw_entropy() Returns Bitfield, Needs u16::from()

For users implementing custom handling:
```rust
let raw = synapse.raw_entropy();
let entropy_val = u16::from(raw);  // MUST convert, not direct use
```

Document this prominently. The type is a bitfield, not a u16.

---

## Summary Table

| Issue | Current | llmosafe Fix |
|-------|---------|--------------|
| Escalate ignored | Caller can proceed | `is_blocking()` or `check_strict()` |
| Halt loops forever | 2s sleep, retry | `should_exit()` variant |
| Manual interpretation | ixd must match variants | `check_blocking()` auto-honors |
| Daemon becomes the problem | No self-protection | `self_ceiling_bytes` in ResourceGuard |
| Magic sleep values | `2000u64` in ixd | `recommended_cooldown_ms()` |
| Status reporting | Parse stderr | `status_label()` or `write_beacon()` |

## Core Principle

**Make the safe behavior the path of least resistance.** With `check_blocking()` and `should_exit()`, an ixd could be:

```rust
let synapse = guard.check_blocking().map_err(|e| {
    eprintln!("ixd: cannot continue safely: {:?}", e);
    std::process::exit(1);
})?;
```

One function call. Safe by default. No interpretation bugs possible.
