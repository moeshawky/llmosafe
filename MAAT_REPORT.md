# Maat Systemic Order Audit — llmosafe

*Weighed: 2026-06-13 | Scope: Full codebase (src/, Cargo.toml, invariants.toml) | Threshold: PASS*

---

## Summary

| Metric | Value |
|--------|-------|
| Total findings | 2 (clustered from 3 raw confessions) |
| Weighted violations | 4,248 |
| Domain violations | 2/42 |
| Domain scores | TRUTH: 0/7, VISIBILITY: 1/7, COHERENCE: 0/7, STRUCTURE: 0/7, VITALITY: 1/7, CONTRACT: 0/7 |
| Status | **PASS** |

---

## Domain: TRUTH (0/7)

All 7 confessions: **NO_EVIDENCE_UNDER_SCOPE**

| ID | Clinical Term | Finding | Evidence |
|----|---------------|---------|----------|
| 1 | split_brain_authority | NO_EVIDENCE_UNDER_SCOPE | No env vars, no split config paths. Two Mutexes (PIPELINE_ARENA, GLOBAL_MEMORY) serve distinct purposes. |
| 2 | import_time_state_capture | NO_EVIDENCE_UNDER_SCOPE | No module-level env reads. Static state initialized at compile time via `Mutex::new()`. |
| 3 | default_impersonates_config | NO_EVIDENCE_UNDER_SCOPE | `PipelineConfig::default()` provides real defaults with builder pattern for override. |
| 4 | version_fragmentation | NO_EVIDENCE_UNDER_SCOPE | Single version declaration: `Cargo.toml` line 3: `version = "0.7.4"`. No conflicting declarations found. |
| 5 | stale_commentary | NO_EVIDENCE_UNDER_SCOPE | No TODO/FIXME/HACK/XXX/BUG markers found in source files. |
| 6 | orphaned_import | NO_EVIDENCE_UNDER_SCOPE | `cargo clippy` passes clean. No unused imports detected. |
| 7 | config_loading_bypass | NO_EVIDENCE_UNDER_SCOPE | No env var reads, no file-based config loading. Configuration via `PipelineConfig` struct only. |

**Structural note:** The TRUTH domain is exceptionally clean. The codebase uses compile-time initialization (`static PIPELINE_ARENA: Mutex<...> = Mutex::new(...)`) rather than runtime config loading. This is appropriate for a `no_std`-compatible safety library.

---

## Domain: VISIBILITY (1/7)

### Confession 9: `silent_failure_returns` — **GUILTY**

**Declaration:** "I have not allowed silent None/empty/default returns on failure"

**Evidence (strong):**
- `lib.rs:293,351,371,396,429,462,573,620,649,681,695,709,723,737,751,784,802,817,853` — 19 instances of `unwrap_or_else(std::sync::PoisonError::into_inner)` in C-ABI module
- Pattern: `PIPELINE_ARENA.lock().unwrap_or_else(std::sync::PoisonError::into_inner)` — recovers from poisoned mutex by returning the inner state
- No logging, no error signal, no metric emission when poisoning occurs

**Weight calculation:**
- Invariant risk: 8 (Caller makes decisions on corrupted state)
- Blast radius: 7 (All C-ABI consumers affected)
- Latentness: 9 (Return value looks valid; no error marker)
- Time amplification: 7 (Silent failures compound)
- **Weight: 3,528**

**Counter-hypothesis:**
- evidence_against: FFI safety requires not panicking across the C boundary (UB per Rust ABI). `PoisonError::into_inner()` is the idiomatic Rust recovery pattern.
- benign_explanation: The C-ABI module is designed to never panic. Recovery from poisoned locks is intentional fail-safe behavior. The system continues with potentially stale (but not corrupt) state because the mutex was already locked when the poisoning thread panicked.
- needed_to_confirm: Verify whether mutex poisoning actually occurs in production. Instrument a log statement to detect poisoning frequency.

**Scar tissue assessment:** The `unwrap_or_else(PoisonError::into_inner)` pattern is **SCAR TISSUE** — ugly but preserves the invariant that FFI calls never panic. The finding is NOT the recovery pattern itself, but the LACK OF LOGGING when recovery occurs.

---

### Confession 10: `unlogged_degradation` — **GUILTY** (clustered with Confession 9)

**Declaration:** "I have not allowed warning-worthy conditions to pass without a log"

**Evidence (strong):** Same locations as Confession 9. Mutex poisoning is a warning-worthy condition that executes without any log emission.

**Weight calculation:**
- Invariant risk: 5 (Doesn't break correctness, breaks awareness)
- Blast radius: 7 (Operations loses visibility)
- Latentness: 8 (No signal — degradation invisible)
- Time amplification: 8 (Undetected degradation compounds)
- **Weight: 2,240**

**Cluster note:** False independence guard triggered. Confessions 9 and 10 describe the SAME wound — mutex poisoning recovery without logging. Counted as ONE finding with weight = max(3528, 2240) = 3,528.

---

### Confessions 8, 11-14: **NO_EVIDENCE_UNDER_SCOPE**

| ID | Clinical Term | Finding |
|----|---------------|---------|
| 8 | bare_except_suppression | NO_EVIDENCE_UNDER_SCOPE — Not applicable in Rust (no bare except) |
| 11 | subprocess_output_loss | NO_EVIDENCE_UNDER_SCOPE — No subprocess calls in library |
| 12 | silent_storage_corruption | NO_EVIDENCE_UNDER_SCOPE — No database operations |
| 13 | silent_empty_database_creation | NO_EVIDENCE_UNDER_SCOPE — No database operations |
| 14 | partial_result_mask_failure | NO_EVIDENCE_UNDER_SCOPE — C-ABI returns specific error codes (-9, -1 to -8) |

---

## Domain: COHERENCE (0/7)

All 7 confessions: **NO_EVIDENCE_UNDER_SCOPE**

| ID | Clinical Term | Finding | Evidence |
|----|---------------|---------|----------|
| 15 | cache_source_divergence | NO_EVIDENCE_UNDER_SCOPE | No caches with invalidation issues. |
| 16 | import_time_state_read | NO_EVIDENCE_UNDER_SCOPE | `GLOBAL_MEMORY` initialized at compile time via `Mutex::new()`. |
| 17 | stale_detection_proliferation | NO_EVIDENCE_UNDER_SCOPE | 5 detectors (repetition, drift, confidence, adversarial, CUSUM) guard different resources. |
| 18 | orphaned_lock_file | NO_EVIDENCE_UNDER_SCOPE | No file-based locks. |
| 19 | generation_counter_drift | NO_EVIDENCE_UNDER_SCOPE | `NEXT_GENERATION` AtomicU64 with Relaxed ordering; Mutex provides synchronization. |
| 20 | phantom_database_entity | NO_EVIDENCE_UNDER_SCOPE | No database. |
| 21 | session_state_leakage | NO_EVIDENCE_UNDER_SCOPE | `llmosafe_destroy()` clears arena slots. In-process memory dies with process. |

---

## Domain: STRUCTURE (0/7)

All 7 confessions: **NO_EVIDENCE_UNDER_SCOPE** (with scar tissue notes)

| ID | Clinical Term | Finding | Notes |
|----|---------------|---------|-------|
| 22 | patch_over_crack | NO_EVIDENCE_UNDER_SCOPE | No repeated defensive patterns addressing same root cause. |
| 23 | module_size_creep | **SCAR TISSUE** | 4 files >500 lines: lib.rs (1185), llmosafe_pipeline.rs (1286), llmosafe_pid.rs (1211), llmosafe_kernel.rs (1196). Size justified by safety-critical complexity and comprehensive documentation. |
| 24 | defensive_conversion_proliferation | **UNVERIFIABLE** | `unwrap_or_else(PoisonError::into_inner)` appears 20+ times. Idiomatic Rust pattern; central wrapper would be less clear. |
| 25 | cwd_dependency | NO_EVIDENCE_UNDER_SCOPE | No CWD reads. |
| 26 | subsystem_proliferation | NO_EVIDENCE_UNDER_SCOPE | 4-tier architecture is intentional design. |
| 27 | format_variation | NO_EVIDENCE_UNDER_SCOPE | No format variations. |
| 28 | concern_duplication | NO_EVIDENCE_UNDER_SCOPE | `calculate_halo_signal()` is thin wrapper around `get_bias_breakdown().total()` — intentional API. |

**Scar tissue note (Confession 23):** The module sizes exceed 500 lines, but splitting would create circular dependencies or require exposing internal implementation details. The files have clear section boundaries and are coherent modules. This is load-bearing complexity, not rot.

---

## Domain: VITALITY (1/7)

### Confession 31: `deprecated_interface_persistence` — **GUILTY**

**Declaration:** "I have not allowed deprecated interfaces to persist without sunset"

**Evidence (medium):**
- `llmosafe_sifter.rs:19-20`: "The legacy keyword-based `calculate_halo_signal()` and `get_bias_breakdown()` remain for backward compatibility"
- `llmosafe_sifter.rs:401`: "Legacy keyword-based halo signal. For backward compatibility"
- Both functions are exported as public API in `lib.rs:166`
- No `#[deprecated]` attribute on either function
- No timeline for removal documented

**Weight calculation:**
- Invariant risk: 4 (Deprecated code accumulates maintenance burden)
- Blast radius: 5 (External consumers may depend on legacy API)
- Latentness: 6 (Functions work; just suboptimal)
- Time amplification: 6 (More consumers adopt deprecated API over time)
- **Weight: 720**

**Counter-hypothesis:**
- evidence_against: The functions are documented as legacy with "Prefer sift_perceptions() for new code"
- benign_explanation: Backward compatibility requires keeping these functions until a major version bump
- needed_to_confirm: Check if any external crates depend on these functions

---

### Confessions 29-30, 32-35: **NO_EVIDENCE_UNDER_SCOPE**

| ID | Clinical Term | Finding |
|----|---------------|---------|
| 29 | orphaned_import_persistence | NO_EVIDENCE_UNDER_SCOPE |
| 30 | dead_test_persistence | NO_EVIDENCE_UNDER_SCOPE — 390 tests pass |
| 32 | stale_doc_persistence | NO_EVIDENCE_UNDER_SCOPE |
| 33 | mechanism_persistence | NO_EVIDENCE_UNDER_SCOPE |
| 34 | feature_flag_persistence | NO_EVIDENCE_UNDER_SCOPE — 5 feature flags are clean |
| 35 | migration_artifact_persistence | NO_EVIDENCE_UNDER_SCOPE |

---

## Domain: CONTRACT (0/7)

All 7 confessions: **NO_EVIDENCE_UNDER_SCOPE** (with scar tissue notes)

| ID | Clinical Term | Finding | Notes |
|----|---------------|---------|-------|
| 36 | schema_drift | NO_EVIDENCE_UNDER_SCOPE | No schema. |
| 37 | type_boundary_violation | **SCAR TISSUE** | C-ABI uses unsafe blocks with SAFETY comments. Type contracts documented. |
| 38 | lying_error_message | NO_EVIDENCE_UNDER_SCOPE | Error codes well-documented in lib.rs comments. |
| 39 | version_mismatch | NO_EVIDENCE_UNDER_SCOPE | Single version in Cargo.toml. |
| 40 | response_contract_violation | NO_EVIDENCE_UNDER_SCOPE | C-ABI returns specific error codes consistently. |
| 41 | resource_contract_violation | NO_EVIDENCE_UNDER_SCOPE | ResourceGuard checks RSS/CPU with documented bounds. |
| 42 | lifetime_contract_violation | **SCAR TISSUE** | `store_objective()` creates `&'static str` from `Box`. SAFETY comment explains lifetime contract via field declaration order. |

**Scar tissue note (Contract 37/42):** The C-ABI module contains 6 `#[allow(clippy::not_unsafe_ptr_arg_deref)]` annotations. These are correct for FFI functions that receive raw pointers — the safety contract is documented in SAFETY comments. This is scar tissue that preserves the Rust↔C contract.

---

## Structural Diagnosis

The llmosafe codebase exhibits **exceptionally strong systemic order** for a safety-critical library. The two findings share a single root cause:

**The FFI safety trade-off:** The C-ABI module must never panic across the Rust↔C boundary (UB per Rust ABI). This forces `unwrap_or_else(PoisonError::into_inner)` recovery from poisoned mutexes. The recovery is correct for safety (prevents UB) but creates an observability gap (no logging when recovery occurs). This is a **known architectural trade-off**, not accidental disorder.

**The backward compatibility pressure:** The legacy `calculate_halo_signal()` and `get_bias_breakdown()` functions persist without deprecation warnings. This reflects the tension between API stability and code hygiene in a published library.

**What the codebase does WELL:**
- Compile-time initialization eliminates config-loading disorders
- Typestate pattern (`SiftedSynapse → ValidatedSynapse`) prevents illegal state transitions
- Comprehensive `invariants.toml` documents cross-module contracts
- DO-178C/MISRA-grade lint configuration catches common failure modes
- C-ABI error codes are consistently documented

---

## Heatmap

| Domain | Density | Locations |
|--------|---------|-----------|
| VISIBILITY | High | `lib.rs:170-853` (C-ABI module — 19 mutex recovery sites) |
| VITALITY | Low | `llmosafe_sifter.rs:401-417` (legacy API functions) |

---

## Unknowns

1. **Runtime mutex poisoning frequency:** Does mutex poisoning actually occur in production? The recovery pattern exists but its activation rate is unknown.
2. **External consumer dependency:** Are `calculate_halo_signal()` and `get_bias_breakdown()` used by downstream crates? Removal timeline depends on this.
3. **Module size maintainability:** Do the 4 large files (>500 lines) cause actual review difficulties in practice? The size is justified by complexity, but empirical evidence is needed.

---

## Next Instruments

1. **Runtime monitoring:** Add `tracing::warn!` or `eprintln!` in the mutex poisoning recovery path to detect activation frequency.
2. **Deprecation campaign:** Add `#[deprecated(since = "0.8.0", note = "use sift_perceptions()")]` to `calculate_halo_signal()` and `get_bias_breakdown()`.
3. **Module decomposition study:** If module sizes become a maintenance burden, consider splitting `lib.rs` C-ABI module into a separate `c_abi.rs` file.

---

## Final Verdict

```yaml
maat_report:
  codebase: llmosafe
  weighed: 2026-06-13
  scope: Full codebase (src/, Cargo.toml, invariants.toml)
  weight: 4248
  threshold: PASS
  status: PASS
  
  domains:
    - name: TRUTH
      score: 0/7
    - name: VISIBILITY
      score: 1/7
    - name: COHERENCE
      score: 0/7
    - name: STRUCTURE
      score: 0/7
    - name: VITALITY
      score: 1/7
    - name: CONTRACT
      score: 0/7
  
  finding_count:
    guilty: 2
    scar_tissue: 3
    unverifiable: 1
    no_evidence: 36
  
  weight: 4248
  status: PASS
```

---

*"The heart of the codebase is weighed — not for sins, but for its capacity to tell the truth about itself under change, failure, time, and operation."*
