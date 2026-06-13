## 2025-06-13 - IDOR/Stale handle masking bug in C ABI
**Vulnerability:** C ABI getter functions use `instance_id` (a handle containing a generation counter) as a raw array index. For instance IDs with generation > 0, this causes out-of-bounds check failures, defaulting responses and masking valid security/safety metrics.
**Learning:** In C ABI boundaries, handle formats that pack generation information must be explicitly unpacked before array indexing. Implicit casts to `usize` bypass this unpacking.
**Prevention:** Always use `unpack_handle` on the external handle before using it as an internal arena index, and check the generation match to prevent stale-handle access.
