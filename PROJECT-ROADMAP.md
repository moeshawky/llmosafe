# PROJECT-ROADMAP.md — llmosafe

**Project roadmap:** PROJECT-ROADMAP.md — the living plan that gates scope creep and guides controlled evolution.

## Intent

Pre-sifter corrective integrity + sifter audit on branch `corrective-pass/p0-p3`; `main` frozen at `ac56ee4`. No main merge before final validation.

## State (verified 2026-09-14)

- `188b6d2` corrective + `5106917` closeout — **BASELINE_GREEN** (5106917 = pre-sifter corrective baseline)
- 643/514/375 tests green (`--all-features` / default / `--lib`)
- `fmt --check`, `clippy --all-targets --all-features -D warnings`, `doc`, `bench --no-run`, `no_std release` — all clean
- 27 `no_mangle`; version lock 4×0.7.7
- Branch pushed, `main` untouched (rev-list: 0 behind, 2 ahead)

## Phases

1. **Corrective** — CLOSED (`188b6d2`)
2. **Baseline verification** — DONE (CEO re-measurement §6.2 CUSTODY-REPORT-PRE-SIFTER.md)
3. **Roadmap gate** — THIS STEP (rewrite complete)
4. **Sifter audit** — S3 first, then CUSUM trail, Rust budget, S2/BUG6 kill-or-retain, naming, F1 determinism, ctypes
5. **Bounded sifter remediation** — fixes on branch only
6. **Final adversarial validation** + oracle/graph pass
7. **PR/merge to main** — after phase 6 clears

## Scope-Creep Defenses

- **IN:** Sifter heuristics audit + bounded fixes on `corrective-pass/p0-p3`.
- **OUT:** Reopening P0 architecture (A1/A2/D2/D3/R1 etc.) without a counterexample; violated invariant escalated to CEO; no main merge before final validation; no Python binding redesign unless ABI break is proven.

## Controlled Evolution

- Re-read this intent every 10 actions. Stop on architecture-reopen temptation.
- Drift checkpoints: after sifter audit, after each remediation batch, before final validation, before PR.

## Frozen Reference

- `main` pinned at `ac56ee4`
- Branch `corrective-pass/p0-p3` at `5106917`
- Source evidence: CUSTODY-REPORT-PRE-SIFTER.md
