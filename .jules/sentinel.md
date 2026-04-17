## 2024-04-13 - [Sentinel setup]
**Vulnerability:** None yet
**Learning:** Initializing sentinel.md
**Prevention:** N/A

## 2024-04-13 - [C-ABI Safety: Unbounded C-string reads]
**Vulnerability:** `llmosafe_calculate_halo` uses `CStr::from_ptr` which scans indefinitely for a null-terminator. This is vulnerable to out-of-bounds reads if the caller passes an unterminated string or a memory block without a null byte.
**Learning:** For C-ABI integration, prioritize explicit length bounds and `core::slice::from_raw_parts` to prevent out-of-bounds memory reads instead of relying on unbounded C-string null-terminator scans (like `CStr::from_ptr`).
**Prevention:** Always require an explicit length argument for string pointers in FFI boundaries to bound the memory read.

## 2024-06-20 - [CRITICAL] Prevent Out-Of-Bounds Read in C-ABI via Explicit Lengths
**Vulnerability:** Unbounded C-string reads in FFI (e.g., `llmosafe_calculate_halo` using `CStr::from_ptr`) allow out-of-bounds memory reads or segmentation faults if the string is not properly null-terminated by the caller or if the string contains invalid UTF-8 bytes mixed with no null terminator.
**Learning:** In C-ABI boundaries, relying on implicit null-terminators `\0` is unsafe and prone to memory-safety bugs, especially when strings are passed from higher-level languages (like Python) or constructed manually.
**Prevention:** Always require explicitly passed length bounds (`text_len: usize`) alongside pointers in C-ABI functions and use `core::slice::from_raw_parts` to guarantee bounded, safe memory reads. Ensure `usize` correctly maps to `size_t` via `cbindgen.toml`.

## 2026-04-17 - [CRITICAL] Prevent FFI Panic in Mutex::lock()
**Vulnerability:** The `process_state_update` C-ABI function panicked if the internal Mutex was poisoned (via `.expect()`), which causes Undefined Behavior and host crashes across FFI boundaries.
**Learning:** Panicking across FFI is unsafe. Rust Mutex poisoning should map to error states, not crashes, to maintain application resilience against DoS.
**Prevention:** Never use `.unwrap()` or `.expect()` across FFI boundaries. Explicitly match the `Mutex::lock()` result and return appropriate C-ABI error codes (like `-6` for poison).
