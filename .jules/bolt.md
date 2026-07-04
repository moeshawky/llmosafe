## 2025-05-18 - Optimize Streaming Token Processing
**Learning:** In hot streaming processes like DriftDetector::observe within llmosafe_detection.rs, parsing streaming string tokens into intermediate stack collections (ArrayVec) before hashing and matching introduces unnecessary iteration, bounds-checking overhead, and pushes the operation out of single-pass processing.
**Action:** Perform hashing and matching operations inline within the primary split_whitespace().take(N) iterator pass, bypassing intermediate allocations to achieve pure single-pass execution.
