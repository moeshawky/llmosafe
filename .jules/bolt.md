## 2024-05-24 - Initial Journal
**Learning:** Initial journal creation for performance optimizations.
**Action:** Use this file to log critical performance learnings specific to this codebase.
## 2024-05-24 - Faster String Splitting for ASCII System Files
**Learning:** In Rust, `split_whitespace()` performs full Unicode property checks on every character, which is slow. For system files guaranteed to contain only ASCII data (such as Linux system metrics in `/proc/stat` or `/proc/loadavg`), `split_ascii_whitespace()` bypasses the Unicode checks and yields a significant performance improvement in hot paths.
**Action:** Use `str::split_ascii_whitespace()` instead of `str::split_whitespace()` when parsing known ASCII-only files.
