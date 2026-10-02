## 2025-05-15 - Fast ASCII split in hot path
**Learning:** When parsing system files like `/proc/stat` that are guaranteed to be ASCII, `split_ascii_whitespace()` is orders of magnitude faster than `split_whitespace()`.
**Action:** Use `split_ascii_whitespace()` for system/log file parsing to reduce CPU overhead.
