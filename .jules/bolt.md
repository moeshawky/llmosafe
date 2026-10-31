## 2024-05-24 - Faster String Parsing for ASCII Inputs
**Learning:** In Rust, parsing system files guaranteed to contain only ASCII data (like Linux system metrics in `/proc/stat` or `/proc/loadavg`) using `str::split_ascii_whitespace()` instead of `str::split_whitespace()` yields a significant performance improvement because it bypasses full Unicode property checks.
**Action:** Replace `split_whitespace()` with `split_ascii_whitespace()` when processing files with guaranteed ASCII contents (like `/proc/*` metrics).
