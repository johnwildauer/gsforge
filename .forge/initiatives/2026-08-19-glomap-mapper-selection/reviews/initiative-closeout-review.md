# Review: Closeout — Restore GLOMAP Mapper Selection

## Review metadata

- **Review type:** `closeout`
- **Status:** `complete`
- **Reviewer:** `gsforge closeout reviewer`
- **Date:** `2026-08-19`
- **Reviewed artifact:** [`initiative-v1.md`](../initiative-v1.md)
- **Compared commit/diff:** Final recorded initiative state; this closeout review made no production-code or test changes.

## Scope of review

Reviewed the authoritative initiative plan, WO-001, its implementation review, execution log, relevant requirements and blueprints, and the validated source/test state. The closeout is bounded to mapper command selection and the documented automated/manual gates. It does not evaluate qualitative reconstruction superiority or authorize calibration, fallback, or unrelated cleanup.

## Findings

### Blockers

- None. WO-001 is complete, its implementation review passes with warnings, and all mandatory bounded dispatch, automated-test, and user-run A/B gates have recorded evidence.

### Warnings

- Windows Unicode output workaround, unsupported `colmap --version`, and optional focal calibration are explicitly deferred as non-fatal, out-of-scope follow-ups. The CPU Ceres/cuDSS fallback warning is likewise non-fatal and did not prevent either run.
- The recorded A/B result demonstrates command dispatch, successful completion, 185 registered cameras, and `sfm/sparse/0` for both methods. It does not demonstrate that either method is qualitatively superior.
- Repository-wide lint findings remain pre-existing/outside this bounded cleanup; scoped validation passed.

### Notes

- No durable requirement or blueprint update was authorized or needed; the initiative changes the implementation alignment gap without broadening baseline documentation.
- No production code or tests were changed during this review/closeout activity.

## Traceability and validation

- **Requirements checked:** [`REQ-RECONSTRUCTION.md`](../../../requirements/features/REQ-RECONSTRUCTION.md:18), AC-RECON-001.1 and AC-RECON-001.2.
- **Blueprints checked:** [`reconstruction.md`](../../../blueprints/features/reconstruction.md:5), [`reconstruction-pipeline.md`](../../../blueprints/components/reconstruction-pipeline.md:16), and [`gsforge-cli.md`](../../../blueprints/containers/gsforge-cli.md:5).
- **Commands/evidence checked:** 24 focused tests passed, 79 repository tests passed, scoped Ruff and format checks passed, and `git diff --check` passed. COLMAP 4.0.2 help and the same-project A/B logs are recorded in [`execution-log.md`](../execution-log.md:23).
- **Manual gates checked:** `glomap` used `global_mapper`; `colmap` used `mapper`; both accepted the selected command/options, completed, registered 185 cameras, and produced `sfm/sparse/0`.

## Decision

- **Outcome:** `closed with follow-up`
- **Required next state:** Set initiative status to `closed-with-follow-up`; keep the completed initiative and deferred warnings discoverable.
- **Human approval required:** `no`
- **Approval/evidence:** The human-run binary/help and A/B gate evidence is recorded in [`execution-log.md`](../execution-log.md:23); no qualitative superiority conclusion is required for this bounded gate.
