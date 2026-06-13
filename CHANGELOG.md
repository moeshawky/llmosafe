# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.4] — 2026-06-13

### Added
- **Maat systemic order audit** — Ran Maat audit (PASS, weight 4,248), found 2 clustered findings
- **Mutex poisoning observability** — Added `tracing::warn!` on all `PoisonError::into_inner()` recovery paths
  - Created `lock_arena()` helper in C-ABI module (replaces 19 inline patterns)
  - Created `lock_memory()` helper in cognitive_memory module
  - `tracing` added as optional dependency gated behind `std` feature
- **Legacy function deprecation** — Added `#[deprecated(since = "0.8.0")]` to keyword-based functions
  - `calculate_halo_signal()` — keyword-only (18.6% accuracy), use `sift_perceptions()` (93.4%)
  - `get_bias_breakdown()` — keyword-only, use `sift_perceptions()` instead
  - Internal callers marked with `#[allow(deprecated)]` for backward compatibility
- **Branch hygiene** — Extracted useful code from 3 stale branches before deletion
  - i128 trend optimization tests
  - Integration policy tests (231 lines)
  - hash_str invariant tests (53 lines)
- **Python bindings audit** — Verified no mocks/stubs, all functions call real Rust FFI

### Changed
- **Python version sync** — `__init__.py` version bumped 0.7.3 → 0.7.4 to match Cargo.toml
- **Clippy compliance** — Added `#[allow(deprecated)]` to public re-exports in `lib.rs`

### Fixed
- **C-ABI UTF-8 truncation UB** — `store_objective` now uses `is_char_boundary()` backtracking
  to prevent multi-byte character splitting before `from_utf8_unchecked`. (#127)
- **System metrics silent failures** — Fixed silent failures in system metrics parsing. (#123)

### Removed
- **Stale branches** — Deleted 120 remote branches and 3 local branches (`devel`, `bolt-memory-trend-i128`, `fix-spin-loop-dos`)

## [0.7.3] — 2026-06-07

### Fixed

- **PID cascade: wire full 4-tier control** — Memory and kernel error channels
  were hardcoded to 0.0 in `PidInput::new()`, making the PID effectively 2-tier
  (body + sift) instead of the documented 4-tier cascade. `e_mem` and `e_kernel`
  are now computed from WorkingMemory statistics and kernel entropy respectively.
  The PID formula in `compute_pid_score_inner()` now reads `e_body`, `e_mem`,
  and `e_kernel` as additional I-term channels via multi-channel blend.

- **Pressure pre-gate implemented** — `process_with_pressure()` documentation
  claimed a pre-SIFT pressure gate that didn't exist in code. Critical and
  Emergency pressure levels now gate through `EscalationPolicy` before the
  SIFT stage runs, returning early on blocking decisions.

- **KERNEL_UNSTABLE override activated** — The `OverrideFlags::KERNEL_UNSTABLE`
  flag was defined and tested but never set by any production code path. Now
  wired from `DynamicStabilityMonitor` state to `apply_safety_overrides()`.

- **rustdoc: fix bracket-escaping warnings** — `[0,1]` and `[0,100]` ranges in
  module docs were parsed as intra-doc links. Escaped with backticks.

### Changed

- **Python package hygiene** — Version synced across all manifests, `__all__`
  sorted and deduplicated, missing docstring args added, type annotations
  improved, mypy Python version bumped to 3.10, stale wheel removed.

- **WD-40 repo cleanup** — `.gitignore` hardened with missing patterns
  (`output/`, `training_metrics.jsonl`, `.antigravitycli/`, `llmosafe-py/dist/`).
  Removed stale tracked documents (`RECOMMENDATIONS.md`, `DESIGN_DECISION_v0.5.0.md`).
  Fixed AGENTS.md paradox (it is human-authored DNA, now correctly tracked).
  Removed dead reference from kernel.rs doc comment.

- **dal feature documented** — `Cargo.toml` `[features]` now explains that
  `dal` gates DO-178C Design Assurance Level safety overrides.

- **C-ABI blocking documented** — `llmosafe_get_environmental_entropy()` now
  warns about ~100ms blocking from `/proc/stat` reads in its doc comment.

### Added

- **Lint justification comments** — 22 test-scoped lint allows in `lib.rs`
  and 1 module-level allow in `detection.rs` now carry DO-178C justification
  comments matching sibling modules.

- **Prepublish recon** — Repo maintenance workflow audit (CAM + CBP + AD + AP
  phases) completed. 10 findings across 4 failure categories resolved.
  Audit workpapers preserved in `.audit/workpapers/`.

## [0.7.1] — 2026-06-05

### Added

- **`ResourceGuard::for_testing()` constructor** — injection point for deterministic
  entropy/pressure values, enabling test coverage of blocking-loop success paths
  previously gated by live OS measurements

- **`DesignAssuranceLevel` wired into `EscalationPolicy`** — DAL tiers (A–E) now gate
  decision severity at runtime. Halt→Escalate→Warn→Proceed downgrading follows
  DO-178C partitioning. Compile-time `dal` feature gates `apply_safety_overrides`
  hard halts vs advisory passthrough.

- **`PidInput` struct wired** — replaces 7-arg `compute_pid_score_pure` and
  8-arg `compute_pid_score` signatures with a single typed aggregate. Removes
  `#[allow(clippy::too_many_arguments)]` hack.

- **`process_safe()` on `CognitivePipeline`** — pre-call `ResourceGuard` gate
  with deadline fallback. If resources are safe, runs full pipeline; if deadline
  expires, falls back to `process_with_pressure()`.

- **`PipelineConfig.use_detection_gate` toggle** — alternative non-PID decision
  path using `DetectionResult` + `EscalationPolicy.decide_from_detection()` with
  first-match-wins severity ordering. Lighter-weight than full PID.

- **Exposure layer** — internal state now queryable through accessors, C-ABI,
  and Python bindings:
  - `classifier_score` (raw logit before sigmoid)
  - `pid_state` (acute/chronic entropy, pressure norm)
  - `memory_stats()` (mean, variance, trend, drift)
  - `kernel_output` + `body_pressure`
  - `combined_risk_bits()` (OOV ratio × detection flags 2D risk space)

- **Python `CognitivePipeline` pyclass** — wraps the C-ABI arena pipeline.
  5-stage process (SIFT→MEMORY→KERNEL→detectors→PID), 6 detectors,
  DAL gating, 13-field result dict. Constructor accepts `dal_level`,
  `use_detection_gate`, `memory_depth` with `llmosafe_configure` C-ABI.

### Changed

- **`dal_a`/`dal_e` feature flags merged** → single `dal` feature. Without
  `dal`, `apply_safety_overrides` is a no-op passthrough. With `dal`, hard
  halts enforced (BIAS/EXHAUSTED/KERNEL_UNSTABLE).

- **`sift_observation` fixed** — now includes keyword-bias backstop matching
  `sift_text`'s dual-path (classifier + keyword) behavior.

- **`sift_perceptions` deprecation message** corrected to point to `sift_text()`
  instead of the obsolete `sift_observation()`.

- **Python v0.7.0 threshold alignment** — tests updated to match kernel
  constants: `PRESSURE_THRESHOLD=40000`, `STABILITY_THRESHOLD=50000`,
  HallucinationDetected surprise threshold `58000`.

### Removed

- **`GainSchedule` struct** — strict subset of `PidConfig` with zero production
  callers.
- **`Setpoint` struct** — zero-field const-generic phantom type, never referenced.
- **`sift_observation_inner`** — private single-caller wrapper, inlined into
  `sift_observation`.

### Fixed

- **Missing test coverage for `check_blocking()` and `check_with_deadline()`** —
  three new cross-module invariant tests cover Proceed path, retry exhaustion with
  sustained pressure, and immediate deadline-expired error.

- **`dal` feature enabled in Python Cargo.toml** — safety overrides were silently
  disabled in Python builds (feature was missing from dependency declaration).

- **6 missing Python exports** added to `__init__.py` and `__all__`.

- **7 standalone pyfunctions wired** — `get_decision`, `get_entropy`,
  `get_surprise`, `get_detection_flags`, `get_oov_ratio`,
  `get_stages_executed`, `get_step_count`.

- **Python package standards** — `py.typed` marker, `LICENSE` file,
  `.gitignore`, type hints on `__version__` and `parse_synapse`.

## [0.7.0] — 2026-06-04

### Added

- **TF-IDF classifier** (`llmosafe_classifier`): streaming FNV-1a tokenizer (unigrams + bigrams), binary search in sorted vocab array, 256-entry sigmoid LUT, zero-alloc, no_std. Replaces keyword-based halo scoring with learned weights from 42,845 real samples (ShieldLM + neuralchemy + deepset). 93.4% accuracy, 91.0% F1 on held-out data.
- **build.rs vocabulary generation**: compiles `vocab_model.bin` into embedded `VOCAB` const array. Validates sort order, hash uniqueness, and NaN. Fail-closed fallback on model corruption.
- **Training pipeline** (`tools/train_tfidf_classifier.py`): mutual information feature selection, boolean TF-IDF, logistic regression (sklearn), JSONL input, binary model output.

### Fixed

- **Entropy formula corrected** (`sifter.rs`): replaced `65535*(1-p)` with binary entropy `65535*4*p*(1-p)`. Old formula assigned maximum entropy to safe-confident text and zero entropy to dangerous-confident text, inverting the stability gate. Binary entropy peaks at p=0.5 (true uncertainty) and drops to 0 at both extremes.
- **Entropy composition**: `saturating_add` changed to `max` in `sift_text()`. Classifier entropy and keyword-bias boost no longer double-count — the greater of the two pathways determines entropy. `PipelineResult.classifier_prob` now correctly recovers pure classifier probability via `entropy / 65535`.
- **`sift_bias_flag_matches_breakdown` invariant**: description and check both updated to `has_bias == (classifier.is_manipulation || bias_breakdown.total() > 0)`, resolving self-contradiction with dual-path `sift_text()` OR-logic.
- **Surprise thresholds recalibrated**: `warn_surprise` 300→42600, `escalate_surprise` 500→55700. Old thresholds were calibrated for keyword-halo [0,1000] range; classifier uses probability*65535 [0,65535].
- **Body→policy mismatch**: `check_blocking()` and `check_with_deadline()` now call `decide_with_pressure()` instead of `decide()`, adding resource pressure signal. Body entropy [0,1000] was ~50x below policy thresholds [30000,50000], making resource pressure invisible.
- **DriftDetector empty-objective fix**: `observe()` returns early, `is_drifting()` returns false when objective is empty. Previously caused perpetual drift.
- **Dead code removed**: `!negated` check in emphasis scoring (redundant with prior `continue`), `KERNEL_UNSTABLE` override in `process_ctrl()` (masked by `PRESSURE_THRESHOLD < STABILITY_THRESHOLD`).
- **`sift_perceptions()` dual-path**: deprecated function now uses `sift_text()` (classifier + keywords) per observation, selecting by highest `raw_entropy()`, instead of classifier-only inner path.
- **`calculate_utility` excess-word fallback**: expanded objective cache to 128 words; pre-collects excess words once instead of re-splitting per unmatched observation word.
- **`phrase_matches` zero-allocation**: iterator-based comparison replaces `Vec<&str>` allocation per multi-word phrase match.
- **rustdoc warnings**: fixed unresolved link brackets in `SifterOutput`, `classify_text`, and `BodyOutput` doc comments.

### Added

- **AdversarialDetector wired** (`CognitivePipeline`): instantiated in `with_config()`, called during detection stage via `is_adversarial()`. New `FLAG_ADVERSARIAL = 0x20` and updated `DETECTION_FLAGS_MASK = 0x3F`. Reset in both `reset_detectors()` and `reset_full()`. New invariant `pipeline_adversarial_wired`.
- **`classifier_score` on `PipelineResult`**: raw logistic regression logit (`ClassificationResult.score`) preserved through pipeline for diagnostic access. `sift_text_with_score()` added as `pub(crate)` internal helper; `sift_text()` delegates to it (backward-compatible).

### Changed

- **STABILITY_THRESHOLD**: 1000→50000 for classifier probability space [0,65535]
- **EscalationPolicy thresholds recalibrated**: `warn_entropy` 600→30000, `escalate_entropy` 800→40000, `halt_entropy` 1000→50000
- **`raw_surprise` carries `classifier_score`**: `probability * 65535` stored in surprise field
- **`has_bias` from classifier**: `is_manipulation = score > THRESHOLD`, not keyword breakdown
- **Synapse reserved field layout**: detection flags expanded to 6 bits (bits 0-5), OOV ratio shifted to bits 6-13, clear mask updated to `0x3FFF`.
- **C-ABI `calculate_halo`**: now routes through dual-path `sift_text()` instead of keyword-only `calculate_halo_signal()`, returning combined classifier + keyword entropy.
- **4 invariants.toml entries updated**: threshold references, `best_halo`→`is_manipulation`, `last_entropy()`→`mean_entropy()`, pressure zone range
- **`.gitignore` hardened**: excludes audit analysis artifacts, generated training data, model binaries, `.moeinclude`

## [0.6.0] - 2026-05-28

### Added

- **`SiftedSynapse::from_synapse()`** — public constructor replacing `new()` for external use. `new()` is now `pub(crate)` to prevent bypassing the sifter pipeline ([GH-1a2dcab](https://github.com/moeshawky/llmosafe/commit/1a2dcab)).
- **`ResourceGuard::check_with_entropy()`** — reuses a previously-measured entropy value to prevent TOCTOU in `check_blocking()`.
- **`ResourceGuard::check_blocking_with_max_retries()`** — configurable max retries for blocking resource checks. `check_blocking()` now defaults to 3 retries and returns `DeadlineExceeded` instead of spinning indefinitely.
- **`EscalationPolicy::decide_from_detection()`** — maps `DetectionResult` fields (stuck, drifting, decaying, adversarial) to `SafetyDecision` severity levels.
- **`From<StabilityResult> for CognitiveStability`** conversion.
- **Multi-word phrase matching** — `SEMANTIC_TRAPS` and `TEMPLATE_FITTING` now detect phrases ("as an ai", "instead of", "rather than") via Phase 2 token-window matching. Single-word detection unchanged for no_std.
- **New `EscalationReason` variants**: `StuckAgent`, `GoalDriftDetected`, `ConfidenceDecaying`, `AdversarialDetected`.
- **Shadow validators** — `debug_assert!` checks at sift→memory boundary enforcing 8 CMIT invariants.
- **`WorkingMemory::SIZE > 0`** compile-time assertion.
- **Cross-Module Invariant Tracing (CMIT)** test suite: 21 property-based and fault-injection tests (`tests/cross_module_invariants.rs`).
- **`invariants.toml`** — 18 documented cross-module invariants across the perception, resource, decision, and typestate chains.
- **`AdversarialDetector::hash_lowercase()`** — FNV-1a hash with ASCII lowercase folding (no allocation), removing the dependency on `RepetitionDetector::hash_str`.

### Fixed

- **Decision priority inversion**: `decide()` and `decide_with_pressure()` now check Halt conditions BEFORE Escalate. Previously, high entropy with bias could return Escalate instead of Halt ([GH-1a1b05c](https://github.com/moeshawky/llmosafe/commit/1a1b05c)).
- **Stability threshold off-by-one**: `Synapse::stability()` now uses `>` (not `>=`) for `STABILITY_THRESHOLD=1000`. Entropy=1000 is Stable, 1001+ is Unstable — consistent with `invariants.toml` and `CognitiveEntropy::is_stable()`.
- **Trend temporal ordering**: `WorkingMemory::trend()` now walks the ring buffer in temporal order (oldest→newest) instead of physical index, producing correct regression slopes after wraparound.
- **DynamicStabilityMonitor divide-by-zero**: `get_thresholds()` returns `(u32::MAX, 0, 0)` when `!seen`, preventing overflow on first update.
- **Trend denominator zero guard**: Returns `0.0` instead of NaN when all buffer values are identical.
- **Negation TTL window**: Extended from 3→6 tokens. `"not a very well known expert"` now correctly suppresses authority bias on "expert".
- **TOCTOU in `check_blocking()`**: Now calls `check_with_entropy()` instead of `check()`, reusing the entropy value that was approved by the policy decision.
- **`check_blocking` bounded retry**: No longer spins indefinitely under sustained pressure. Default 3 retries, then returns `DeadlineExceeded`.
- **EMOTIONAL_APPEAL keywords pruned**: Removed high-frequency words (`love`, `joy`, `happy`, `sad`, `angry`, `exciting`, `amazing`, `beautiful`, `emotional`, `appealing`) that triggered on everyday speech. Retained fearmongering/hyperbolic terms.
- **`"while"` removed from `SEMANTIC_TRAPS`**: False positive on everyday speech.
- **FFI `llmosafe_check_resources` and `llmosafe_get_stability`** now handle `DeadlineExceeded` error code (-7).

### Changed

- **`SiftedSynapse::new()` now `pub(crate)`** — use `SiftedSynapse::from_synapse()` for external construction. This enforces the sifter pipeline boundary: production code must route through `sift_perceptions()`. Direct construction via `from_synapse()` is intended for testing and crate-internal use.
- **`ResourceGuard::auto()` now fail-closed on non-Linux**: When `/proc/meminfo` is unreadable (macOS, Windows, containers), ceiling defaults to `0` (always `ResourceExhaustion`) instead of `usize::MAX/2` (unlimited). This is a safety improvement — callers should explicitly set a ceiling.
- **`EscalationPolicy::decide()` reordered** for correct severity: Halt → Escalate → Warn (previously checked Bias Escalate before entropy Halt).
- **`EscalationPolicy::decide()` threshold semantics**: Halt uses `>` (1001+), Escalate/Warn use `>=` (800+/400+). Consistent with stability threshold semantics.
- All 41 test sites updated from `SiftedSynapse::new()` to `SiftedSynapse::from_synapse()`.

### Security

- **`check_blocking` bounded retry**: Prevents infinite spin under sustained resource pressure (DoS hardening).
- **`ResourceGuard::auto()` fail-closed**: Non-Linux platforms no longer default to unlimited resources.

### Verified

- 237 tests (97 unit + 140 integration/edge/CMIT) — all pass
- `cargo check` + `cargo check --no-default-features` — clean (std and no_std)
- `cargo clippy -- -D warnings` — clean
- 17 audit findings addressed (see `1a1b05c`)

---

## [0.5.5] - 2026-05-13

### Added

- **Python bindings**: PyO3-based Python package with maturin build system
  - `calculate_halo()`, `check_resources()`, `get_stability()`, and 4 other FFI functions
  - Exception classes: `LLMOSafeError`, `ResourceExhaustedError`, `CognitiveInstabilityError`, `BiasHaloDetectedError`
  - Full pytest test suite (19 tests)

### Fixed

- **Fixed:** Zero cooldown on all `SafetyDecision::Halt` and `SafetyDecision::Escalate` return sites (was `0ms`, now `30000ms` and `5000ms` respectively) — previously caused immediate re-entry into evaluation loop on halt/escalation
- **Fixed:** Missing cooldown on `decide_from_stability()` Halt path (`CognitiveInstability` variant)
- **Fixed:** `test_small_ceiling` now properly catches `ResourceExhaustedError` exception
- **Fixed:** Removed AI fingerprint ("Research Grounds" comment) from kernel docs
- **Fixed:** Removed internal `.jules/` development artifacts from git tracking

### Changed

- `EscalationPolicy::decide()` now returns `5000ms` cooldown for all `Escalate` variants (bias, entropy, surprise)
- `EscalationPolicy::decide()` now returns `30000ms` cooldown for `Halt` variant
- `EscalationPolicy::decide_from_stability()` now returns `30000ms` cooldown for `CognitiveInstability`
- `.gitignore`: Added `.jules/` directory (internal development artifacts)

## [0.5.4] - 2026-04-23

### Fixed

- Updated version metadata (no code changes)

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
