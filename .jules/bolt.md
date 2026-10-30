## 2026-10-30 - [Use split_ascii_whitespace instead of split_whitespace for ASCII system files]
**Learning:** In Rust, parsing system files guaranteed to contain only ASCII data (such as Linux system metrics in /proc/stat or /proc/loadavg) with `str::split_ascii_whitespace()` instead of `str::split_whitespace()` bypasses full Unicode property checks, yielding a significant performance improvement in hot paths.
**Action:** Replace `split_whitespace()` with `split_ascii_whitespace()` when processing known ASCII system files.
