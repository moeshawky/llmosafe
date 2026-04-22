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

## 2024-04-22 - [FFI Panic Vulnerability on Mutex Poisoning]
**Vulnerability:** The FFI-exported function `process_state_update` used `GLOBAL_MEMORY.lock().expect("memory lock poisoned")`. If the mutex was poisoned, the function would panic. Panics across the FFI boundary lead to Undefined Behavior and can crash the host application (Denial of Service).
**Learning:** Rust's standard library `Mutex` becomes poisoned if a thread panics while holding the lock. Using `.expect()` or `.unwrap()` on lock acquisition is unsafe in C-ABI exports, where exceptions cannot be properly caught by the host.
**Prevention:** Always use explicit pattern matching (`match`) on fallible operations like `Mutex::lock()` in FFI boundary code. Return a predefined C-ABI error code (e.g., `-6`) instead of allowing a panic to propagate to the caller.
