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
- **Historical release holdout v1 F1:** 0.9527 (consumed once — must not be reused for tuning; exceeds FH1 benchmark 0.9441)

## Phases

1. **Corrective** — RETIRED (historical baseline `3547340`; branch `corrective-pass/p0-p3` merged and deleted 2026-09-15)
2. **Baseline verification** — RETIRED (P11 matrix done; superseded by release gate)
3. **Sifter remediation** — COMPLETE (4 commits on `feat/sifter-20k-tfidf`)
4. **Release gate** — SIFTER_RELEASE_GATE_V1.md frozen P7 bars
5. **NEXT GATES** — See below
6. **PR/merge to main** — after all next gates clear

## Next Gates

1. **neutral-short characterization** — DONE: 8/10 short-neutral
   escalations are fail-closed by design (classifier INTERCEPT + vocab
   overlap, zero downstream-only escalation); DOCUMENTED_DEFERRED, see Residuals.
2. **deterministic resource test** — Make `resource_to_decision_chain_integrity`
   pass deterministically across cgroup environments.
3. **P7 formalization** — Formalize P7 bars in invariants or source code.

## Residuals (DOCUMENTED_DEFERRED, release-scoped)

- Thin recovery 0.5517 vs bar ≥0.50 (margin 0.0517; single-path concentration).
- 8/10 short-neutral fail-closed (DOCUMENTED_DEFERRED): INTERCEPT=2.673815 + single-unigram vocab overlap, zero downstream-only escalation; pending product safety-policy decision; behavior locked by `*_fail_closed_documented` contract tests (`tests/sifter_unified_tests.rs`).
- Halo `u16::MAX` ambiguity (DOCUMENTED_DEFERRED): caller-must-validate; regression at `src/lib.rs:1193-1223,1236-1239,2161-2173`.
- Historical release holdout v1 consumed once (F1=0.9527; new sealed holdout required before next experiment).

## Scope-Creep Defenses

- **IN:** Sifter release v1.0 gates per SIFTER_RELEASE_GATE_V1.md.
- **OUT:** Reopening P0 architecture (A1/A2/D2/D3/R1 etc.) without a counterexample;
  violated invariant escalated to CEO; no main merge before all next gates clear;
  no Python binding redesign unless ABI break is proven.
- **REJECTED (no re-entry without versioned policy decision):** Dual TF-IDF, Meta-LR,
  BERT-tiny/J1, BERT-small/J1, synthetic training; hostile-bank training prohibited (evaluation holdout only).

## Controlled Evolution

- Re-read this intent every 10 actions. Stop on architecture-reopen temptation.
- Drift checkpoints: after next gates, before PR.

## Frozen Reference

- `main` baseline at `3547340`
- Branch `feat/sifter-20k-tfidf` at 4 commits (7d94fcf, e76f872, 46c5f0e, c85c9cd)
- Source evidence: SIFTER_RELEASE_GATE_V1.md, TRAINING_REPORT.md, CEO_RULINGS.md
- Rejected architectures: SIFTER_RELEASE_GATE_V1.md guard (full text archived in `.archive/sifter-release/REJECTED_ARCHITECTURES.md`)
