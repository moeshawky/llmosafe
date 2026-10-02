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
4. **Python wheel ABI** (NEW 2026-09-23) — `llmosafe-py` ships
   CPython-version-specific wheels (`cp312-cp312`), not `abi3`, because
   `llmosafe-py/Cargo.toml` pyo3 0.20 lacked the `abi3-pyXX` feature and
   `wheels.yml` built with `-i python3.12`. Consequence: **zero
   installable artifact on CPython 3.13 (`cp313`)** — no cp313 wheel and
   no sdist to fall back on. NEXT RELEASE must: (a) enable `abi3-py38`
   (one wheel covers 3.8–3.13+); (b) bump pyo3 to ≥0.23 so 3.13 can be
   built and tested (0.20 cannot target 3.13); (c) add the 3.13 classifier;
   (d) publish an sdist (`maturin sdist`) as a source fallback; (e) add a
   3.13 CI cell; (f) reconcile the `wheels.yml` + `PUBLISHING.md`
   "all three projects use identical copies" claim that concealed this
   divergence from ix/sniper.
   - **INTERIM (Path A, shipped 2026-09-23):** `cp38-abi3` wheels appended
     to the existing `0.9.0` on PyPI with no version bump. Known cost:
     mixed-ABI release (cp312-cp312 + cp38-abi3 coexist) and cache /
     lockfile staleness — consumers holding a `uv.lock` or pip cache must
     `uv lock --upgrade` / refresh to see the new file.
   - **NOT YET DONE — CI coverage gap:** `ci.yml` has zero Python jobs
     (no maturin / pytest / mypy / ruff). Nothing detects an ABI or
     classifier divergence from the template, and no 3.13 runtime test
     exists in CI. This remains the real open risk of the abi3 switch.
   - **VERIFIED 2026-10-02, end-to-end:** `cp38-abi3` wheels appended to
     production PyPI `0.9.0` for both arches — measured by installing from
     live PyPI under real CPython 3.13.13 (`cpython-313`):
     `uv run --python 3.13 --with llmosafe==0.9.0` → `import llmosafe` OK,
     `calculate_halo` = 53939 (identical to the 3.12 result), `process_synapse`
     = 0, `get_environmental_entropy` = 2. Resolution for 3.8 / 3.12 / 3.13
     all succeed. `0.9.0` on PyPI now carries four files: the two pre-existing
     `cp312-cp312` plus `cp38-abi3` for x86_64 and aarch64, all
     `manylinux_2_17`.
   - **Never resolved by:** re-uploading the same 0.9.0 filename (PyPI
     rejects it) or an sdist-only fix (pyo3 0.20 cannot compile against
     3.13).

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
