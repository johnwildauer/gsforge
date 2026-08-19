# Review: Implementation — WO-003 GLOMAP GPU Solver Compatibility

## Review metadata

- **Review type:** `implementation`
- **Status:** `blocked`
- **Reviewer:** `independent Forge implementation reviewer`
- **Date:** `2026-08-19`
- **Reviewed artifact:** [`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md)
- **Compared commit/diff:** Current workspace contents of [`sfm.py`](../../../src/gsforge/sfm.py), [`test_sfm.py`](../../../tests/test_sfm.py), [`README.md`](../../../README.md), and the initiative log; no commit identifier or VCS diff was supplied

## Scope of review

Reviewed the approved official-binary CPU-BA contract, GLOMAP command construction and output handling in [`run_mapper()`](../../../src/gsforge/sfm.py:433) and [`_run_colmap_step()`](../../../src/gsforge/sfm.py:272), both mapper-method tests, failure-state behavior in [`run_sfm()`](../../../src/gsforge/sfm.py:750), README/runtime claims, requirements, blueprints, execution evidence, and Forge gates. No production code or tests were changed, and no commands or external binaries were run.

## Findings

### Blockers

- **The required CPU-fallback diagnostic contract is not adequately implemented or tested.** The warning is emitted only when captured GLOMAP output contains the exact case-sensitive substrings checked at [`_run_colmap_step()`](../../../src/gsforge/sfm.py:316). There is no structured solver-mode result, persisted project field, or test proving that the official Ceres/cuDSS warning is classified as CPU BA. The current tests cover command flags but contain no solver-output, warning-classification, or failed-mapper-state case ([`test_sfm.py`](../../../tests/test_sfm.py:102)). This is insufficient for WO-003's explicit acceptance and automated-test requirements ([`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md:97)).

- **Runtime evidence for the claimed official-binary gate is incomplete and not reproducible from the repository.** The log claims a 4.1.1 external probe and CPU-fallback validation ([`execution-log.md`](../execution-log.md:65)), but does not retain the required `global_mapper -h` output, command/exit status, complete solver messages, camera count, sparse-model location, or isolated-project run artifact required by [`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md:97). Under project policy, manual/external gates are unverified rather than passing ([`00-overview.md`](../../../process/00-overview.md:77)).

- **Failure-state behavior is asserted by design but not proven for mapper failure.** `_run_colmap_step()` calls `log_error()` on a nonzero mapper exit at [`_run_colmap_step()`](../../../src/gsforge/sfm.py:306), and [`run_sfm()`](../../../src/gsforge/sfm.py:803) catches `SystemExit`, but no test verifies `sfm_status="failed"`, method persistence, camera count, and no unsafe automatic switch to incremental COLMAP. This is a required recovery behavior, not an optional quality check.

- **The initiative implementation gate was bypassed without recorded authorization.** Production implementation was recorded while the initiative's latest design review was `blocked` and its current gate was `planning-revision` ([`initiative-v1.md`](../initiative-v1.md:9), [`initiative-design-review.md`](initiative-design-review.md:48), [`execution-log.md`](../execution-log.md:65)). No explicit human approval or policy waiver is recorded. This blocks acceptance regardless of the fallback implementation's eventual behavior.

### Warnings

- **Truthful reporting is ephemeral and narrower than the README claim.** The implementation prints raw mapper output and a warning, but does not persist the solver mode in [`ProjectMeta`](../../../src/gsforge/project.py:63) or [`SfmResult`](../../../src/gsforge/sfm.py:93). A caller or later `info` operation cannot distinguish GPU-requested GLOMAP from CPU-BA GLOMAP. The README accurately describes the intended supported behavior ([`README.md`](../../../README.md:104)), but the requirement/blueprint set does not document the new diagnostic/state contract.

- **The implementation recognizes only two exact message fragments.** Variations in capitalization, punctuation, or a runtime message that says “falling back to CPU” without those fragments will complete without the explicit gsforge warning. Recommendation: define a stable classification rule from the official output and test both fallback and GPU-capable/no-warning paths.

- **WO-003 remains marked `draft` with an unchecked completion checklist** ([`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md:5)); the execution log's implementation-validation entry does not reconcile that state.

### Notes

- The GLOMAP command still uses `global_mapper` and explicitly requests `GlobalMapper.gp_use_gpu=1` and `GlobalMapper.ba_ceres_use_gpu=1` ([`run_mapper()`](../../../src/gsforge/sfm.py:476)). Incremental COLMAP retains the separate `mapper` namespace and does not receive global options, so no dispatch contamination was found.
- No automatic switch to incremental COLMAP was found in the reviewed path, which is consistent with the recovery boundary.
- The README's consistent-focal-length prerequisite and deferred calibration boundary are consistent with this work order and were not changed into solver behavior.

## Traceability and validation

- **Requirements checked:** [`REQ-RECONSTRUCTION.md`](../../../requirements/features/REQ-RECONSTRUCTION.md:18), especially AC-RECON-001.1 and NFR-RECON-001.
- **Blueprints checked:** [`reconstruction.md`](../../../blueprints/features/reconstruction.md:5), [`reconstruction-pipeline.md`](../../../blueprints/components/reconstruction-pipeline.md:16), and [`project-state.md`](../../../blueprints/components/project-state.md:1).
- **Process checked:** [`00-overview.md`](../../../process/00-overview.md:53) and [`05-implementation-review.md`](../../../process/05-implementation-review.md:5).
- **Commands/evidence checked:** Logged focused/full test and Ruff claims plus the 4.1.1 external-probe claim at [`execution-log.md`](../execution-log.md:65); no commands were rerun and no raw external transcript was available.
- **Manual gates checked:** Official binary help/runtime evidence, isolated CPU-fallback run, failed-mapper recovery, design approval, and branch evidence. Required evidence is incomplete or unverified.

## Decision

- **Outcome:** `blocked`
- **Required next state:** Return WO-003 and the initiative to `in-progress`/planning revision; define and test the solver-mode diagnostic/state contract, add failed-stage tests, attach reproducible external evidence, reconcile durable documentation, and obtain required design authorization before another review.
- **Human approval required:** `yes`
- **Approval/evidence:** No valid design approval or explicit gate waiver is recorded. This reviewer does not self-approve while blockers remain.
