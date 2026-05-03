# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.3] - 2026-04-23

### 🔒 Security

#### HIGH: FFI Panic DoS via Mutex Poisoning (CWE-388)

**Problem:** The `process_state_update` FFI function used `.expect()` on `GLOBAL_MEMORY.lock()`. If a thread previously panicked while holding the lock, subsequent FFI calls would panic, causing the host application to crash (Denial of Service).

**Solution:** Replaced `.expect()` with explicit `match` to return error code `-6` (SelfMemoryExceeded) when the mutex is poisoned.

```rust
// Before (vulnerable)
let mut memory = GLOBAL_MEMORY.lock().expect("memory lock poisoned");

// After (safe)
let mut memory = match GLOBAL_MEMORY.lock() {
    Ok(guard) => guard,
    Err(_) => return -6, // Return error code instead of panicking
};
```

### ⚡ Performance

#### Bolt: Cache halo signal in sift_perceptions

**Problem:** `calculate_halo_signal(best_obs)` was called again after the search loop to determine `has_bias`, despite having been computed for every observation inside the loop.

**Solution:** Cache the `halo` value alongside `best_obs` when it updates, avoiding redundant O(N) recalculation.

## [0.5.2] - 2026-04-14

### 🔒 Security

#### Critical: Out-of-Bounds Read Vulnerability (CWE-125)

**Problem:** The `llmosafe_calculate_halo` FFI function used `CStr::from_ptr` which relies on null-terminator scanning. A malicious caller could pass an unterminated string, causing the function to read past allocated memory boundaries — potentially exposing sensitive data or causing crashes.

**Solution:** Added an explicit `len` parameter that bounds all memory reads.

```c
// Before (vulnerable)
uint16_t llmosafe_calculate_halo(const char *text);

// After (safe)
uint16_t llmosafe_calculate_halo(const char *text_ptr, uintptr_t len);
```

**Impact:** All FFI consumers (C, Python, other languages) must now provide explicit string lengths. This is a **breaking change** for the C-ABI, but necessary for memory safety.

**Migration:**
```c
// C consumers
uint16_t halo = llmosafe_calculate_halo(text, strlen(text));
```
```python
# Python consumers
encoded = text.encode('utf-8')
halo = lib.llmosafe_calculate_halo(encoded, len(encoded))
```

---

### ⚡ Performance

#### Negation Tracking: 40% Faster Bias Detection

**What changed:** Replaced an array-based sliding window with a Time-To-Live (TTL) counter for tracking negation words.

**Why it matters:** Previously, every word required checking the preceding 3 words for negation — causing redundant string trimming and array operations. The TTL counter tracks state in a single integer, eliminating the inner loop entirely.

| Metric | Before | After |
|--------|--------|-------|
| Complexity | O(N × 3) | O(N) |
| String ops | 3× per word | 1× per word |
| Bench time | ~1.09µs | ~0.82µs |

#### Utility Calculation: 4× Faster for Long Objectives

**What changed:** Added a 64-element stack-allocated cache for trimmed objective words.

**Why it matters:** Previously, every word in the observation caused the entire objective string to be re-split and re-trimmed. Now, objective words are trimmed once and cached for O(1) lookup.

| Metric | Before | After |
|--------|--------|-------|
| Complexity | O(N × M) | O(N + M) |
| Memory | Heap allocations | Stack only |
| Bench improvement | — | ~865ms → ~200ms (100K iterations) |

---

### 📋 Breaking Changes

| Change | Impact | Action Required |
|--------|--------|-----------------|
| `llmosafe_calculate_halo` signature | C-ABI consumers | Pass explicit `len` parameter |
| `cbindgen.toml` | Build system | `usize_is_size_t = true` added |

---

### 📝 Files Changed

```
.jules/bolt.md              |  Performance documentation
.jules/sentinel.md          |  Security incident log
cbindgen.toml               |  Added usize_is_size_t config
examples/c_consumer/main.c  |  Updated to use explicit length
examples/python_consumer/main.py |  Updated ctypes signatures
include/llmosafe.h          |  Regenerated C header
src/lib.rs                  |  C-ABI function signature fix
src/llmosafe_sifter.rs      |  TTL counter + stack cache
```

---

### ✅ Verification

- **89 tests passing** (including new edge cases)
- **Clippy clean** (zero warnings)
- **C consumer builds and runs** (gcc + LD_LIBRARY_PATH)
- **Python bindings verified** (ctypes integration)
- **Benchmarks confirmed** performance improvements

---

## [0.5.1] - 2026-04-14

### Fixed
- Changed `unwrap()` to `expect()` in test code for better error messages

---

## [0.5.0] - 2026-04-09

### Breaking Changes
- `SafetyDecision::Halt` now has signature `Halt(KernelError, u32)` - second parameter is cooldown_ms
- C-ABI error codes: new codes `-6` (SelfMemoryExceeded), `-7` (DeadlineExceeded)
- Match statements on `SafetyDecision::Halt` must now handle 2-tuple

### Added
- **SafetyDecision::Exit(KernelError)**: Unrecoverable error requiring immediate termination
- **check_blocking()**: Blocks until resources are safe, honoring cooldowns
- **check_with_deadline(Instant)**: Same as check_blocking with timeout
- **SafetyDecision helper methods**: `is_blocking()`, `should_exit()`, `recommended_cooldown_ms()`, `status_label()`
- **KernelError variants**: `SelfMemoryExceeded`, `DeadlineExceeded`

### Changed
- All `EscalationPolicy::decide()` methods now populate `cooldown_ms` field

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

---

## [0.4.9] - 2026-04-09

### Changed
- Updated package metadata (description, keywords) for crates.io
- 89 tests passing, no breaking changes

---

## [0.4.2] - Previous

Initial stable release with:
- ContextRegistry for session management
- SynapseABI v2 (128-bit layout)
- Basic sifter with bias detection
- Working memory with surprise gating
- Resource monitoring (Tier 0)
