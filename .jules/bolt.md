## 2026-09-27 - Stack Array over ArrayVec
**Learning:** In no_std Rust environments, for small temporary arrays populated within hot loops, replacing generic wrappers like ArrayVec with a simple stack-allocated array (e.g., [0u32; N]) and a manual length counter reduces initialization and insertion overhead.
**Action:** Use a plain stack array and a length counter instead of ArrayVec when performance is critical and bounds are mathematically guaranteed.
