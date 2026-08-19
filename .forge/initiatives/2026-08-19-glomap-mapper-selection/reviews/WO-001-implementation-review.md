# Review: Implementation — WO-001 Mapper Selection

## Review metadata

- **Review type:** `implementation`
- **Status:** `complete`
- **Reviewer:** `gsforge implementation reviewer`
- **Date:** `2026-08-19`
- **Reviewed artifact:** [`WO-001-mapper-selection.md`](../work-orders/WO-001-mapper-selection.md)
- **Compared commit/diff:** Recorded implementation state in [`sfm.py`](../../../../src/gsforge/sfm.py:327) and [`test_sfm.py`](../../../../tests/test_sfm.py:67); no production-code or test edits were made by this review.

## Scope of review

Reviewed the work order, initiative acceptance criteria, execution log, mapper implementation, focused command-construction tests, requirements, and reconstruction blueprints. The review covers only the authorized mapper dispatch and its validation evidence; reconstruction quality, calibration changes, training, and unrelated lint diagnostics were excluded.

## Findings

### Blockers

- None. The global path invokes `global_mapper` with the `GlobalMapper` namespace, the incremental path invokes `mapper` with the `Mapper` namespace, and no global-only selection is present on the incremental path.

### Warnings

- Repository-wide lint remains non-zero on pre-existing diagnostics outside this cleanup and existing diagnostics elsewhere in `sfm.py`; the scoped Ruff check and format check passed and no autofixes were applied.
- Windows Rich Unicode output required UTF-8 session environment variables during the user-run validation. This is non-fatal and out of scope.
- `colmap --version` is unsupported by the target binary. COLMAP `4.0.2`, commit `d927f7e`, CUDA build, and command help were independently recorded. This is non-fatal and out of scope.
- The global run emitted a non-fatal focal-length-prior warning. Optional focal calibration/tuning is out of scope.
- A non-fatal CPU Ceres/cuDSS fallback warning was recorded. No command-contract failure resulted.

### Notes

- The A/B evidence confirms bounded execution and matching observable output counts, not qualitative reconstruction superiority. No superiority claim is made.
- No requirements, blueprints, production code, or tests were changed as part of this review.

## Traceability and validation

- **Requirements checked:** [`REQ-RECON-001`](../../../requirements/features/REQ-RECONSTRUCTION.md:18), especially AC-RECON-001.1 and AC-RECON-001.2.
- **Blueprints checked:** [`reconstruction.md`](../../../blueprints/features/reconstruction.md:5), [`reconstruction-pipeline.md`](../../../blueprints/components/reconstruction-pipeline.md:16), and [`gsforge-cli.md`](../../../blueprints/containers/gsforge-cli.md:5).
- **Commands/evidence checked:** `pixi run -e dev test tests/test_sfm.py` — 24 passed; `pixi run -e dev test` — 79 passed; scoped Ruff check and format check passed; `git diff --check` passed; COLMAP 4.0.2 help confirmed `mapper` and `global_mapper` plus their separate option namespaces.
- **Manual gates checked:** Same 185 prepared frames and same CUDA-enabled COLMAP 4.0.2 binary for both runs; `glomap` completed through `global_mapper` with 185 cameras and `sfm/sparse/0`; `colmap` completed through `mapper` with 185 cameras and `sfm/sparse/0`.

## Decision

- **Outcome:** `pass-with-warnings`
- **Required next state:** `complete`; proceed to initiative closeout with the warnings recorded as follow-ups.
- **Human approval required:** `no`
- **Approval/evidence:** Human-provided binary-help and A/B reconstruction evidence is recorded in [`execution-log.md`](../execution-log.md:23). The default implementation-review policy does not require an additional approval after this review.
