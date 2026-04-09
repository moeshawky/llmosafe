## 2024-04-09 - C-ABI FFI Out-of-bounds Read via Unbounded CString
**Vulnerability:** The FFI layer used `CStr::from_ptr` on untrusted input strings without verifying null termination, allowing out-of-bounds memory reads.
**Learning:** `CStr::from_ptr` is inherently dangerous in FFI contexts without length bounds, as malformed input causes unbound memory scans until a random null byte is hit.
**Prevention:** Always require an explicit `len` (or `size_t`) argument in FFI signatures and use `core::slice::from_raw_parts` to construct memory-safe slices before parsing text.
