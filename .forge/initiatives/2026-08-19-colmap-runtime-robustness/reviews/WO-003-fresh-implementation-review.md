# Fresh Independent Review — WO-003 GLOMAP CPU Fallback Contract

## Review metadata

- **Review type:** `fresh independent implementation review`
- **Status:** `blocked`
- **Reviewer:** `independent Forge architect/reviewer`
- **Date:** `2026-08-19`
- **Reviewed work order:** [`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md)
- **Reviewed implementation:** [`sfm.py`](../../../src/gsforge/sfm.py) and [`test_sfm.py`](../../../tests/test_sfm.py)
- **Validation policy:** Logged validation and external-probe claims were treated as documentary evidence only; commands and binaries were not rerun.

## Scope and evidence reviewed

This review reread the stakeholder-approved high-level CPU fallback contract, latest execution log, prior blocked review, current mapper implementation/tests, [`README.md`](../../../README.md), [`REQ-RECONSTRUCTION.md`](../../../requirements/features/REQ-RECONSTRUCTION.md), reconstruction blueprints, and the available 4.0.2/4.1.1 runtime logs. The remediation claim at [`execution-log.md`](../execution-log.md:112) was checked against the current workspace.

## Findings

### Confirmed implementation alignment

1. **The approved contract is represented without overengineering.** [`run_mapper()`](../../../src/gsforge/sfm.py:455) continues to dispatch GLOMAP through `global_mapper`, requests `GlobalMapper.gp_use_gpu=1` and `GlobalMapper.ba_ceres_use_gpu=1`, and does not add an elaborate solver telemetry model or silently switch to incremental mapping. [`_run_colmap_step()`](../../../src/gsforge/sfm.py:294) prints captured mapper output and emits a high-level CPU bundle-adjustment warning when the known official Ceres/CUDA/cuDSS fragments are present.

2. **The README accurately describes the supported boundary.** It identifies official-binary CPU BA as valid, avoids claiming GPU BA, and marks custom Ceres/Caspar/COLMAP builds as untested and unsupported at [`README.md`](../../../README.md:104). This is consistent with the revised work order and the stakeholder decision recorded at [`execution-log.md`](../execution-log.md:65).

3. **Mapper dispatch boundaries remain intact.** The incremental path retains `mapper` and the `Mapper` namespace, while the GLOMAP path uses `global_mapper` and the `GlobalMapper` namespace. Existing tests at [`test_glomap_uses_global_mapper_and_global_options()`](../../../tests/test_sfm.py:103) and [`test_colmap_uses_incremental_mapper_and_mapper_options()`](../../../tests/test_sfm.py:117) cover this boundary. No automatic fallback from GLOMAP to incremental COLMAP was found.

### Technical blockers

1. **Warning classification coverage is not adequate for the stated acceptance.** The remediation added [`test_cpu_fallback_warning_is_reported_for_glomap()`](../../../tests/test_sfm.py:180), which verifies one exact warning shape, but there is no companion test for a normal Ceres solver report/no warning, no test for the alternate known `without cuDSS support` fragment, and no test for capitalization or wording variation. The implementation remains case-sensitive and fragment-specific at [`_run_colmap_step()`](../../../src/gsforge/sfm.py:339). For the intentionally high-level contract this need not become structured telemetry, but the supported warning classification boundary still needs representative positive and negative tests.

2. **Mapper failure-state persistence is still unproven.** [`_run_colmap_step()`](../../../src/gsforge/sfm.py:325) calls `log_error()` on a nonzero mapper result and [`run_sfm()`](../../../src/gsforge/sfm.py:841) catches the resulting `SystemExit`, but no test verifies persisted `sfm_status="failed"`, selected method, zero camera count, and absence of an unsafe mapper switch. This is required by the work order's recovery contract and NFR-RECON-001, even though no production dispatch defect was found.

3. **The external gate is not reproducibly retained.** The execution log records a 4.1.1 probe and test totals at [`execution-log.md`](../execution-log.md:71), while the repository contains prior runtime logs such as [`garden-glomap-sfm-4.0.2.txt`](../../../garden-glomap-sfm-4.0.2.txt:1). However, no fresh linked artifact provides the claimed 4.1.1 `global_mapper -h` transcript, exact command/exit status, complete solver messages, camera count, sparse-model path, and isolated-project run details required by [`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md:97). The external gate is therefore unverified, not passed.

### Warnings

- Reporting is intentionally ephemeral: the solver limitation is printed/logged but not persisted in [`SfmResult`](../../../src/gsforge/sfm.py:93) or project metadata. This is acceptable under the stakeholder-approved high-level contract, but callers cannot later query the solver mode from project state.
- The detection rule is narrower than the README's general wording. It recognizes the known official messages, not every possible custom-build diagnostic. That limitation is compatible with the explicitly unsupported custom-build path, provided the tests and documentation avoid implying exhaustive classification.
- The work order remains `draft` with an unchecked completion checklist at [`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md:5), despite the validation entry in the log.

## Process-gate disposition

The initiative remains `planning-revision` at [`initiative-v1.md`](../initiative-v1.md:9), and no recorded human design approval or explicit waiver resolves the implementation-before-approval gate in [`initiative-v1.md`](../initiative-v1.md:74). Dedicated-branch creation evidence is also absent from [`execution-log.md`](../execution-log.md:9). These are independent completion blockers; this review does not self-approve them.

## Decision

- **WO-003 outcome:** `blocked` for insufficient warning-classification tests, missing mapper-failure persistence coverage, and unverified/reproducibly retained external evidence.
- **Process outcome:** `blocked` because design approval or an explicit waiver and dedicated-branch evidence remain absent.
- **Technical progress recognized:** the high-level CPU fallback contract is accurately and minimally implemented; GLOMAP selection and failure boundary are preserved; no overengineered solver telemetry or automatic mapper fallback was introduced.
- **Required next action:** add representative warning/no-warning and mapper-failure tests, retain/link the required external probe/run evidence, and obtain the missing process authorization/evidence before another review.
