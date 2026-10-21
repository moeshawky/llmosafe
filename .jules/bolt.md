## 2023-10-21 - Use split_ascii_whitespace for guaranteed ASCII files
**Learning:** `split_ascii_whitespace` avoids full Unicode checks compared to `split_whitespace`, providing a measurable performance gain on system files like `/proc/stat` and `/proc/meminfo` that are guaranteed to be ASCII.
**Action:** Use `split_ascii_whitespace` when processing guaranteed ASCII strings, especially in hot paths.
