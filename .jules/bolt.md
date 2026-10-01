## 2025-05-15 - ArrayVec Allocation Overhead in Hot Loops
**Learning:** In tight loops (like `DriftDetector::observe`), replacing a generic `ArrayVec` wrapper with a simple stack-allocated array (e.g., `[0u32; MAX_CONTEXT_LEN]`) and a manual length counter measurably avoids generic wrapper initialization and insertion overheads.
**Action:** Use simple array + counter for small fixed-size buffer allocations in hot loops instead of generic wrappers.
