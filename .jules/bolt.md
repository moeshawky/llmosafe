## 2024-05-15 - System File Parsing Optimization
**Learning:** In Rust, parsing system files (like `/proc/stat` or `/proc/loadavg`) which are definitively guaranteed to contain only ASCII characters can be significantly bottlenecked by Unicode property checks when using `.split_whitespace()`.
**Action:** Replace `.split_whitespace()` with `.split_ascii_whitespace()` for ASCII-only data sources to bypass Unicode validation and improve parsing speed in hot paths.
