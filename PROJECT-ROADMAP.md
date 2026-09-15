# PROJECT-ROADMAP.md — llmosafe

**Project roadmap:** PROJECT-ROADMAP.md — the living plan that gates scope creep and guides controlled evolution.

## Intent

Sifter release v1.0 — single 20k TF-IDF classifier with documented residual risks.
Main baseline at `3547340`. Branch `feat/sifter-20k-tfidf` contains the complete
sifter remediation. No main merge before all next gates clear.

## State (verified 2026-09-15)

- **Main baseline:** `3547340` (HEAD at merge of corrective-pass/p0-p3)
- **Classifier:** Single 20k TF-IDF, count-based TF, logistic regression
- **Status:** `READY_WITH_DOCUMENTED_RESIDUAL`
- **cargo check --all-targets --all-features:** PASS (4 commits)
- **Model artifact:** `tools/vocab_model.bin` 320,012 bytes, SHA256 `0955b7b2...`
- **Sealed holdout F1:** 0.9527 (exceeds FH1 benchmark 0.9441)

## Phases

1. **Corrective** — CLOSED (`188b6d2` on corrective-pass/p0-p3)
2. **Baseline verification** — DONE (P11 verification matrix)
3. **Sifter remediation** — COMPLETE (4 commits on `feat/sifter-20k-tfidf`)
4. **Release gate** — SIFTER_RELEASE_GATE_V1.md frozen P7 bars
5. **NEXT GATES** — See below
6. **PR/merge to main** — after all next gates clear

## Next Gates

1. **neutral-short characterization** — Determine whether 8/10 short-neutral
   escalations are correct behavior or false positives.
2. **deterministic resource test** — Make `resource_to_decision_chain_integrity`
   pass deterministically across cgroup environments.
3. **P7 formalization** — Formalize P7 bars in invariants or source code.

## Scope-Creep Defenses

- **IN:** Sifter release v1.0 gates per SIFTER_RELEASE_GATE_V1.md.
- **OUT:** Reopening P0 architecture (A1/A2/D2/D3/R1 etc.) without a counterexample;
  violated invariant escalated to CEO; no main merge before all next gates clear;
  no Python binding redesign unless ABI break is proven.

## Controlled Evolution

- Re-read this intent every 10 actions. Stop on architecture-reopen temptation.
- Drift checkpoints: after next gates, before PR.

## Frozen Reference

- `main` baseline at `3547340`
- Branch `feat/sifter-20k-tfidf` at 4 commits (7d94fcf, e76f872, 46c5f0e, <C4>)
- Source evidence: SIFTER_RELEASE_GATE_V1.md, TRAINING_REPORT.md, CEO_RULINGS.md
- Rejected architectures: `REJECTED_ARCHITECTURES.md`
