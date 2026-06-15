# llmosafe v0.7.5 — Breaking Change Notification

**Date:** 2026-06-14
**Severity:** Breaking (ABI)
**Affected:** All C-ABI consumers and Python bindings

---

## Summary

llmosafe v0.7.5 changes the return signature of 6 pipeline getter functions. This is a **breaking change** to the C ABI. v0.7.4 has been yanked from both crates.io and PyPI.

If you are calling any `llmosafe_get_*` function from C, or using the Python `llmosafe` package, you **must** update your code.

---

## What Changed

Six getter functions changed from returning the value directly to using an **out-parameter** pattern with a status return code.

### Before (v0.7.4 and earlier)

```c
// Value returned directly. 0 = valid value AND error indicator (ambiguity).
uint16_t entropy = llmosafe_get_entropy(instance_id);
// BUG: entropy=0 could mean "zero entropy" or "error occurred"
```

### After (v0.7.5)

```c
// Value written to out-pointer. Return code indicates success/failure.
uint16_t entropy;
int32_t rc = llmosafe_get_entropy(instance_id, &entropy);
if (rc != 0) {
    // Handle error: rc=1 (invalid handle), rc=2 (null pointer), rc=3 (no result)
}
// entropy is guaranteed valid when rc == 0
```

---

## Affected Functions

| Function | Old Signature | New Signature |
|----------|--------------|---------------|
| `llmosafe_get_entropy` | `uint16_t (uint32_t)` | `int32_t (uint32_t, uint16_t*)` |
| `llmosafe_get_surprise` | `uint16_t (uint32_t)` | `int32_t (uint32_t, uint16_t*)` |
| `llmosafe_get_detection_flags` | `uint8_t (uint32_t)` | `int32_t (uint32_t, uint8_t*)` |
| `llmosafe_get_oov_ratio` | `uint8_t (uint32_t)` | `int32_t (uint32_t, uint8_t*)` |
| `llmosafe_get_stages_executed` | `uint8_t (uint32_t)` | `int32_t (uint32_t, uint8_t*)` |
| `llmosafe_get_step_count` | `uint32_t (uint32_t)` | `int32_t (uint32_t, uint32_t*)` |

## Return Codes

| Code | Meaning |
|------|---------|
| `0` | Success — value written to `*out` |
| `1` | Invalid handle (slot not found, uninitialized, stale, or out of bounds) |
| `2` | Null pointer passed as `*out` |
| `3` | No result available (`sift_and_process` not called yet) |

---

## Unaffected Functions

These functions are **not** changed:

- `llmosafe_get_classifier_score` — returns `double`, uses `-1.0` sentinel
- `llmosafe_get_body_pressure` — returns `uint32_t`, uses `UINT32_MAX` sentinel
- `llmosafe_get_decision` — returns `int32_t`, uses `-9` sentinel
- `llmosafe_get_pid_state` — already uses out-parameter pattern
- `llmosafe_get_memory_stats` — already uses out-parameter pattern
- `llmosafe_get_kernel_output` — already uses out-parameter pattern

---

## Why This Change

The previous API had a **sentinel ambiguity bug**: five getters used `0` as both a valid value and an error indicator. For example, `llmosafe_get_entropy` returned `0` on error, but entropy=0 is a valid measurement. Callers could not distinguish "zero entropy" from "error occurred."

This has been corrected by separating the value (written to an out-pointer) from the error status (returned as `int32_t`).

---

## Migration Guide

### C Consumers

```c
// OLD (v0.7.4)
uint16_t entropy = llmosafe_get_entropy(id);
if (entropy == 0) { /* error? or zero entropy? */ }

// NEW (v0.7.5)
uint16_t entropy;
int32_t rc = llmosafe_get_entropy(id, &entropy);
if (rc != 0) { /* definitely an error */ }
```

### Python Users

The Python bindings have been updated to raise `LLMOSafeError` on error:

```python
# OLD (v0.7.4)
entropy = llmosafe.get_entropy(pipeline.instance_id)
if entropy == 0:  # ambiguous
    pass

# NEW (v0.7.5)
try:
    entropy = llmosafe.get_entropy(pipeline.instance_id)
except llmosafe.LLMOSafeError as e:
    print(f"Error: {e}")  # "get_entropy failed for instance 0: code 1"
```

The `CognitivePipeline.process()` method continues to return a complete dictionary. If any getter fails internally, the entire `process()` call raises `LLMOSafeError` — no partial dictionaries.

---

## Updated Header

The regenerated C header (`include/llmosafe.h`) reflects the new signatures. Download the latest from the [v0.7.5 release](https://github.com/moeshawky/llmosafe/releases/tag/v0.7.5).

---

## Version Status

| Version | crates.io | PyPI | Status |
|---------|-----------|------|--------|
| v0.7.4 | Yanked | Yanked | **Do not use** — sentinel ambiguity bug |
| v0.7.5 | Published | Published | **Current** — use this version |

---

## Questions?

Open an issue at https://github.com/moeshawky/llmosafe/issues or contact the maintainers directly.
