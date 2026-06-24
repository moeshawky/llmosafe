## 2024-05-24 - Inline Token Processing
**Learning:** Allocating intermediate collections like ArrayVec during text tokenization adds measurable overhead from secondary loops and stack memory operations.
**Action:** Perform matching inline during iterator passes when processing string streams.
