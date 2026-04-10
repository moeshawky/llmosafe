## 2026-04-10 - [C-ABI Out-of-Bounds Read in Halo Signal]
**Vulnerability:** The FFI function `llmosafe_calculate_halo` relied on `CStr::from_ptr` to parse C-strings. An attacker passing a non-null-terminated string could cause memory reads past the intended buffer.
**Learning:** Using `CStr::from_ptr` on FFI boundaries is intrinsically unsafe unless there are strong guarantees about string null-termination. It is better to require explicit length boundaries over assuming null termination.
**Prevention:** Always require an explicit `len: size_t` (or `usize` mapped to `size_t`) parameter on C-ABI string processing endpoints and construct bounds using `core::slice::from_raw_parts`.
