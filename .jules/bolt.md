## 2024-05-25 - Optimize ASCII String Parsing in System Files
**Learning:** In Rust, when parsing system files guaranteed to contain only ASCII data (such as Linux system metrics in `/proc/stat` or `/proc/loadavg`), use `str::split_ascii_whitespace()` instead of `str::split_whitespace()` to bypass full Unicode property checks, yielding a significant performance improvement in hot paths.
**Action:** Replace `split_whitespace()` with `split_ascii_whitespace()` when parsing Linux metrics from `/proc` files.
