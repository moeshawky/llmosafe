# Agent Rules — llmosafe

## DNA vs RNA

Human-written documentation is **DNA** — structural ground truth. AI-generated summaries are **RNA** — inference calibrated against DNA. Never confuse them.

| Artifact | Who | Rule |
|----------|-----|------|
| `///` / `/** */` doc comments | Operator | **Never AI-modified.** Structural ground truth. |
| `//!` module docs | Operator | **Never AI-modified.** |
| `//` / `#` inline comments | Operator | **Never AI-modified.** Intent annotations. |
| `semantic_summary` field | Subagent | **RNA.** Regenerated each commit. Calibrated by DNA. |

| Language | Doc Comment | Module Doc | Inline | RNA Field |
|----------|------------|------------|--------|-----------|
| Rust | `///` | `//!` | `//` | `semantic_summary` |
| Python | `"""` (docstring) | `"""` (module) | `#` | `semantic_summary` |

**Rule:** DNA is the calibration layer. RNA is the inference layer. AI writes to `.annotations/` only — never to source doc comments.

---

## Subagent Annotation Pipeline

When annotations are needed, dispatch subagents in **waves** ordered by dependency (leaf → hub). Each wave's rejections feed the next wave's constraints.

### Rules

1. **Read, don't write.** Subagent reads its target entity + full dependency cluster (callers, callees, trait impls). Writes annotation proposals only — never source.
2. **Evidence required.** Every proposal must cite a line number in the code. No evidence → delete.
3. **Banned words.** Never use: `orchestrates`, `enables`, `facilitates`, `empowers`, `scalable`, `robust`, `architecture`. Describe WHAT code does, never WHY.
4. **Human gate.** Read `.annotations/` → approve/reject/edit → apply approved only. No path from proposal to DNA without operator approval.
5. **Rejection = constraint.** Rejected proposals become `DO NOT` constraints injected into the next wave's subagent prompts. The failure library grows monotonically — never remove entries, only add.

### Wave Ordering (leaf → hub)

1. Foundation types (`Synapse`, `CognitiveEntropy`, `BiasBreakdown`, `KernelError`)
2. Sifter + keyword lists
3. Detection layer (`RepetitionDetector`, `DriftDetector`, etc.)
4. Working memory
5. Kernel + reasoning loop
6. Resource body
7. Integration layer + EscalationPolicy
8. C-ABI + Python bindings

### Dispatch Format

```
[ROLE] Code Cartographer — maps WHAT code does, not WHAT it means.
[CONTEXT] target entity + callee signatures + caller context + existing comments.
[BANNED] inventing purpose, architecture claims, cross-file claims without evidence.
[OUTPUT] .annotations/[file].yaml with proposals + evidence line numbers.
```

### Output Schema

```yaml
# .annotations/llmosafe_kernel.yaml
file: src/llmosafe_kernel.rs
proposals:
  - target: "Synapse::validate (line 327)"
    semantic_summary: "Rejects synapse if bias detected or entropy exceeds stability threshold (1000)."
    evidence: [327, 328, 331]
    confidence: high
    DO_NOT:
      - "Do not claim validate() ensures surprise bounds — it doesn't check surprise."
      - "Do not describe this as 'orchestrating' or 'enabling' — it returns Result."
```

---

## Cross-Module Invariant Tracing (CMIT)

All future work must respect invariants defined in `invariants.toml`. Compound bugs live at boundaries — tests that only check final output cannot catch them. Every change that crosses a tier boundary must:

1. Check the relevant invariant in `invariants.toml`
2. Add or update a cross-module test in `tests/cross_module_invariants.rs`
3. Verify shadow validators fire in `#[cfg(debug_assertions)]` builds

---

## Bent Pyramid — 3-Strike Rule

If the same module boundary fails 3 times:
1. **Note** the pattern (not the instance)
2. **Redesign** the boundary contract
3. **Build again** with the redesign

No fourth patch at the same boundary. Escalate with evidence.

---

## Banned Patterns

| Pattern | Why |
|---------|-----|
| Writing doc comments without operator approval | DNA violation |
| Using banned words (`orchestrates`, `enables`, etc.) | Noise that hides behavior |
| Patching the same boundary 4+ times | Bent Pyramid violation |
| Adding code without a CMIT invariant check | Compound bug risk |
| Deleting entries from rejection/DO_NOT library | History loss |

---

## Release & Publish Automation

### Trichannel Release (crates.io + GitHub + PyPI)

**Trigger:** Push a version tag (`git tag vX.Y.Z && git push origin vX.Y.Z`)

**Workflow:** `.github/workflows/publish-pypi.yml`

| Channel | Automation |
|---------|------------|
| crates.io | Manual: `cargo publish` (after dry-run) |
| GitHub Releases | Manual: `gh release create vX.Y.Z` |
| PyPI + TestPyPI | **Automatic** via trusted publishing OIDC |

### PyPI Trusted Publishing Config

| Field | Value |
|-------|-------|
| Project name | `llmosafe` |
| Owner | `moeshawky` |
| Repository | `llmosafe` |
| Workflow | `publish-pypi.yml` |
| Environment | `pypi` |

### Pre-Publish Checklist (from pre-publish skill)

1. `cargo test --all-features` — all pass
2. `cargo clippy --all-targets` — clean
3. `cargo fmt --check` — clean
4. `cargo publish --dry-run` — passes
5. `maturin build --release` — x86_64 wheel builds
5. `uv build && twine check dist/*` — metadata valid
6. Verify ARM64 CI build passes (check workflow `Build wheels (aarch64)`)

### Audit Workpapers

The repo maintenance workflow (CAM → CBP → AD → AP → Prepublish) writes structured
audit evidence to `.audit/workpapers/`. These are RNA artifacts — AI-generated,
gitignored, never published. Each phase produces a YAML workpaper:

| Phase | Workpaper | Content |
|-------|-----------|---------|
| CAM | `cam-findings.yaml` | G-*/S-* failure scan with line-number evidence |
| CBP | `cbp-boundary-sweep.yaml` | Cross-module boundary map, compound bug detection |
| AD | `ad-root-cause.yaml` | Root cause analysis: file:line, assumption, violation, call chain |
| AP | `ap-fix-report.yaml` | Fix applied, test results, DNA coverage verification |
| Prepublish | `prepublish-recon.yaml` | Gate checklist: git hygiene, changelog, version, rustdoc, dry-run |

**Rules:**
- Workpapers are **evidence** for gate transitions. No phase N+1 without phase N's workpaper.
- `.audit/` and `.annotations/` are RNA directories — gitignored per `.gitignore`.
  Never commit AI-generated artifacts to the repo.
- Audit workpapers are a rolling log: each sweep overwrites the prior.
  If you need history, save manually.

### Version Bump Procedure

```bash
# 1. Bump version in all three manifests
sed -i 's/version = "X.Y.Z"/version = "X.Y.(Z+1)"/' Cargo.toml llmosafe-py/Cargo.toml llmosafe-py/pyproject.toml

# 2. Commit + tag
git add -A && git commit -m "chore: bump version to X.Y.Z"
git tag vX.Y.Z
git push origin main && git push origin vX.Y.Z

# 3. Cargo publish
cargo publish

# 4. GitHub Release
gh release create vX.Y.Z --title "vX.Y.Z" --notes "<CHANGELOG section>"

# 5. PyPI auto-publishes via CI (no manual step needed)
```
