## 2024-05-18 - Faster parsing of Linux system files
**Learning:** In Rust, parsing system files guaranteed to contain only ASCII data (like `/proc/stat`) using `str::split_whitespace()` incurs unnecessary performance overhead from Unicode property checks.
**Action:** Always prefer `str::split_ascii_whitespace()` over `str::split_whitespace()` for system files that are strictly ASCII-encoded.
