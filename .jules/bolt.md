## 2024-05-24 - ASCII System File Parsing
**Learning:** Linux system files (`/proc/stat`, `/proc/loadavg`, `/proc/meminfo`) contain only ASCII data. Using `split_whitespace()` incurs unnecessary Unicode property check overhead.
**Action:** Use `str::split_ascii_whitespace()` when parsing known ASCII system files to significantly reduce CPU overhead during hot-path metric collection.
