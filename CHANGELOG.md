# Changelog

All notable changes to this project will be documented in this file.

## [0.4.9] - 2026-04-09

### Changed
- Updated package metadata (description, keywords) for crates.io
- Version bump from 0.4.2 to 0.4.9

### Note
- This release continues the stable API from v0.4.2
- Bias detection uses keyword matching (no ML model dependencies)
- 89 tests passing
- No breaking changes

## [0.5.0-alpha] - 2026-04-08

### Added
- **MetastabilityCounters**: Per-session event frequency tracking via AtomicU8
  - Tracks: bias_count, entropy_spike_count, cascade_count, surprise_count
  - FFI function: `llmosafe_get_metastability()`
  - Wrapping behavior: 255→0 (safe for frequency counters)

- **Deterministic Reaping**: Session cleanup via FFI
  - FFI function: `llmosafe_reap_stale(current_tick, timeout_ticks)`
  - Returns count of reaped sessions

- **Phrase Matching in Sifter**: 88 new bias detection phrases
  - Authority: "having spent", "decades working", "years in"
  - Social Proof: "what most people", "successful entrepreneurs"
  - Scarcity: "hard to come by", "not widely available"
  - Urgency: "the longer you wait", "window close on"
  - Emotional: "what drives real", "beneath the surface"
  - Expertise: "underlying mechanism", "feedback loops"
  - Semantic Traps: "what appears to be", "contrary to initial"
  - Template: "based on analysis", "what evidence suggests"
  - Phrases score +150 (vs +100 for words)

### Changed
- Phrase matching runs alongside word matching in `get_bias_breakdown`
- No breaking API changes

### Architecture
- Tier 2 trait abstraction prepared for plug-and-play ML model integration
- Supports: mock, local GGUF, remote API, ONNX backends
- Configuration via environment variables

## [0.4.2] - Previous
- Initial release with ContextRegistry, SynapseABI v2, basic sifter
