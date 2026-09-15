# SIFTER_RELEASE_GATE_V1.md — Release Acceptance Gate

**Version:** v1.0  
**Date:** 2026-09-15  
**Status:** AUTHORITATIVE-FOR-THIS-RELEASE — frozen P7 bars verbatim from CEO_RULINGS.md  
**Revision policy:** Revision only by versioned policy change (new gate version required for any bar modification)

---

## P7 Frozen Bars (Authoritative for This Release)

| # | Metric | Threshold | Source |
|---|--------|-----------|--------|
| 1 | pipeline hostile false-safe | ≤ 0.15 | CEO_RULINGS.md P7 PROVISIONAL ACCEPTANCE |
| 2 | hostile unsafe recall | ≥ 0.85 | CEO_RULINGS.md P7 PROVISIONAL ACCEPTANCE |
| 3 | clean benign false-halt | ≤ 0.05 | CEO_RULINGS.md P7 PROVISIONAL ACCEPTANCE |
| 4 | classifier-miss recovery | ≥ 0.50 | CEO_RULINGS.md P7 PROVISIONAL ACCEPTANCE |
| 5 | clean unsafe recall | ≥ 0.90 | CEO_RULINGS.md P7 PROVISIONAL ACCEPTANCE |

---

## Warning Labels

**recovery ≥ 0.50 ≠ 'system is robust'.** A recovery rate above the P7 bar is a necessary
but not sufficient condition. The E2E gate shows recovery at 0.9310 (27/29 classifier
misses recovered), but this is dominated by EscalationPolicy entropy/surprise thresholds
and PID hard invariant — NOT by the keyword bias pathway or adversarial detector.
Recovery mechanism concentration (2-3 mechanisms, ~97% single-path) is a documented
residual risk, not a strength.

**Hostile numbers in teacher-graded context:** The E2E trace set includes teacher-graded
hostile samples with false-safe rate 0.0897, unsafe recall 0.9103, and benign false-halt
0.5517 in certain sub-populations. These are teacher-graded context samples and do NOT
replace the P7 bars above. They are diagnostic only.

---

## Residual Risks

### 1. Thin Recovery Margin
Recovery is dominated by EscalationPolicy/Policy-floor (~81.48% of recoveries) and
PID hard invariant (~22.22%). Only ~3.70% comes from the repetition detector.
Keyword bias, adversarial detector, CUSUM, and confidence detectors contribute
**zero** recoveries. A scenario that bypasses EscalationPolicy entropy/surprise
thresholds could leave classifier misses unrecovered.

### 2. 8/10 Short-Neutral Escalations with NO Threshold-Change Rule
The E2E evaluation shows 8 of 10 short-neutral samples escalate without a documented
threshold-change rule. The neutral-short characterization follow-up is required to
determine whether these escalations are correct behavior or false positives.
Follow-up: **neutral-short characterization follow-up** (open follow-up #1).

### 3. Halo Work Item
- **Meaning:** `u16::MAX` (65535) is the sentinel returned by `llmosafe_calculate_halo`
  on `SiftError::ResourceExhaustion`. It is within the valid entropy range [0, 65535].
- **Collision:** Callers CANNOT distinguish "maximum entropy" from "input error" from
  the return value alone. No ABI break this task.
- **APIs affected:** `llmosafe_calculate_halo` (C-ABI), Python `CognitivePipeline`.
- **Constraints:** Caller must validate against valid entropy range [0, 65535].
- **Migration options:** Future version may add a separate error channel to distinguish
  error sentinels from valid maximum-entropy returns. Requires ABI change.

### 4. Sealed→Historical Release Holdout Rename
- **Rename rule:** `sealed/sealed_holdout.jsonl` → `historical_release_holdout_v1.jsonl`
- **New-holdout-before-next-experiment rule:** Before any new training experiment, a
  fresh holdout must be sealed and evaluated. The previous holdout becomes historical
  and must not be used for training or threshold calibration.

### 5. Deterministic Cgroup-Test Follow-up
The `resource_to_decision_chain_integrity` test fails in environments where cgroup
memory pressure causes `ResourceGuard::auto(0.5)` to compute a ceiling below current
usage. A deterministic cgroup-test follow-up is required to make this test pass
reliably across environments. Follow-up: **deterministic resource test** (open follow-up #3).

---

## Open Follow-Ups

1. **neutral-short characterization** — Determine whether 8/10 short-neutral escalations
   are correct behavior or false positives. Need characterization of neutral-short
   input distribution and escalation mechanism.
2. **deterministic resource test** — Make `resource_to_decision_chain_integrity` pass
   deterministically across cgroup environments.
3. **P7 formalization** — Formalize P7 bars in invariants or source code (currently
   only documented in CEO_RULINGS.md). This is by design for a release-scoped gate.
4. **halo ABI resolution** — Resolve the `u16::MAX` sentinel collision with a future
   ABI change that distinguishes error returns from valid maximum-entropy returns.
5. **sealed→historical_holdout_v1 rename** — Execute the rename and establish the
   new-holdout-before-next-experiment rule in the training pipeline.

---

## Rejected Architecture Guard

See `REJECTED_ARCHITECTURES.md` for the full list. The following architectures are
**REJECTED** and must not be re-entered without meeting the specified reopen conditions:
- Dual TF-IDF (Arm C/D/E): kill numbers -0.0159 F1 (C vs B), -0.0016 F1 (D vs A),
  -0.0240/-0.0662 challenge penalties. Reopen only if dual-head demonstrates >+0.02 F1
  gain over single-head on BOTH FH1 and challenge splits.
- Meta-LR (Arm B): -0.0123 F1 (non-monotonicity, challenge F1=0.2396 collapse).
- BERT-tiny/J1 judge: +0.0000 EV gain.
- BERT-small/J1 judge: +0.0059 EV, 115.1MB cost.
- Synthetic training: origin AUC 0.9999.

---

## Model Artifact

| Property | Value |
|----------|-------|
| Artifact | `tools/vocab_model.bin` |
| Size | 320,012 bytes |
| SHA256 | `0955b7b221a5aaa01d2d607db74d6ea55e0caa085438518e06bbc2c6d64150cc` |
| Determinism | Byte-identical across 2 training passes (seed 42) |
| Architecture | Arm A — Single TF-IDF 20k, count-based TF |
