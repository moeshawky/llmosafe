## 2024-05-15 - Integer Accumulation in Tight Loops
**Learning:** Using `i128` integer accumulations inside a loop defers `f64` conversions to the final reduction, avoiding redundant float conversions and floating-point roundoff error accumulation.
**Action:** Always prefer integer arithmetic inside loops for trend calculations or similar numeric accumulations where the source types map cleanly to integers, deferring the float conversion until necessary.
