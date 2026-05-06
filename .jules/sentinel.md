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

## 2024-06-25 - [HIGH] Prevent FFI Panic DoS via Mutex Poisoning
**Vulnerability:** In `llmosafe_memory.rs`, the C-ABI function `process_state_update` uses `.expect("memory lock poisoned")` when locking `GLOBAL_MEMORY`. If a thread previously panicked while holding the lock, subsequent calls across the FFI boundary will panic, crashing the host application (Denial of Service).
**Learning:** Panicking across the FFI boundary leads to undefined behavior or application crashes. Fallible operations, such as locking a Mutex that might be poisoned, must be handled gracefully and return an appropriate C-ABI error code.
**Prevention:** Use explicit pattern matching (`match`) on `Mutex::lock()` instead of `.expect()` or `.unwrap()`. Return a predefined error code (e.g., `-6`) when a `PoisonError` occurs to ensure system reliability in concurrent environments.

## 2026-05-01 - [CRITICAL] Prevent FFI Out-Of-Bounds via Negative/Large Lengths
**Vulnerability:** External C callers might pass maliciously large sizes or negative integers that cast to huge unsigned values (`usize`). If passed to `core::slice::from_raw_parts`, sizes exceeding `isize::MAX` cause Undefined Behavior (UB), leading to segfaults or potential arbitrary code execution.
**Learning:** Bounding `text_len` with `0` is not enough when constructing memory slices in Rust. We must always constrain FFI length parameters to explicitly avoid lengths greater than `isize::MAX as usize`.
**Prevention:** Add `text_len > isize::MAX as usize` as an early exit condition before creating any slices with `core::slice::from_raw_parts` to prevent system crashes from malformed inputs.

## 2024-06-26 - [MEDIUM] Prevent OOM DoS in AdversarialDetector
**Vulnerability:** `AdversarialDetector::detect_substrings` calls `input.to_ascii_lowercase()`, which allocates a new `String` on the heap proportional to the input size. Passing a massive string (e.g., gigabytes) causes uncontrolled memory allocation, leading to an Out-Of-Memory (OOM) crash and Denial of Service (DoS).
**Learning:** String allocation functions like `to_ascii_lowercase()` must be bounded when processing untrusted input to prevent resource exhaustion attacks, especially in a library intended for safety-critical systems.
**Prevention:** Always enforce a maximum input length (e.g., 64KB) before performing operations that allocate memory proportional to the input size. Use `is_char_boundary` to truncate safely.
