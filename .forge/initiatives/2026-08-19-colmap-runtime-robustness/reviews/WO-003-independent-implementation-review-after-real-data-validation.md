# Independent Implementation Review — WO-003 After Real-Data Validation

## Review metadata

- **Review type:** independent WO-003 implementation review after human design approval and 4.1.1 real-data validation
- **Status:** `pass-with-warnings`
- **Reviewer:** independent Forge architect/reviewer
- **Date:** 2026-08-19
- **Reviewed work order:** [`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md:1)
- **Implementation:** [`sfm.py`](../../../src/gsforge/sfm.py:321)
- **Tests:** [`test_sfm.py`](../../../tests/test_sfm.py:116)
- **External evidence:** [`WO-003-external-evidence-4.1.1.md`](WO-003-external-evidence-4.1.1.md:1)
- **Prior review:** [`WO-003-final-independent-review-after-branch-switch.md`](WO-003-final-independent-review-after-branch-switch.md:1)

## Scope and validation basis

This review is limited to WO-003. I reread the revised contract, current [`_run_colmap_step()`](../../../src/gsforge/sfm.py:321), [`run_mapper()`](../../../src/gsforge/sfm.py:482), [`run_sfm()`](../../../src/gsforge/sfm.py:799), focused tests, prior WO-003 reviews, the README, the official-binary evidence artifact, the latest execution log, and the linked 4.1.1 real-data transcript. No production code or tests were modified. The recorded automated and real-data results were assessed as validation evidence and were not independently rerun in this review session.

## Technical verification

### Confirmed

1. **The intentionally high-level reporting contract is implemented.** GLOMAP remains a call to `global_mapper`; it requests `GlobalMapper.gp_use_gpu=1` and `GlobalMapper.ba_ceres_use_gpu=1`, captures mapper output, and emits a high-level CPU bundle-adjustment warning when the known CUDA/cuDSS fallback fragments are present. The implementation does not claim GPU BA, add structured solver telemetry, or substitute the incremental mapper. This matches the stakeholder-approved scope in [`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md:47).

2. **Warning and no-warning boundaries are covered.** [`test_cpu_fallback_warning_is_reported_for_glomap()`](../../../tests/test_sfm.py:200) verifies the representative CUDA warning, while [`test_normal_ceres_report_does_not_emit_cpu_fallback_warning()`](../../../tests/test_sfm.py:221) verifies that a normal successful Ceres report does not emit the fallback warning. This is appropriate high-level coverage; the contract does not require exhaustive classification of custom-build telemetry or every wording variation.

3. **Mapper failure persistence is correct.** [`test_mapper_failure_persists_failed_state_without_switching_method()`](../../../tests/test_sfm.py:331) verifies that a mapper failure persists `sfm_status=failed`, preserves `sfm_method=glomap`, records zero cameras, and does not silently retry with incremental COLMAP. The [`SystemExit`](../../../src/gsforge/sfm.py:883) boundary updates project state before re-raising the actionable failure.

4. **GLOMAP dispatch is isolated from incremental COLMAP dispatch.** [`test_glomap_uses_global_mapper_and_global_options()`](../../../tests/test_sfm.py:117) checks `global_mapper` and only the `GlobalMapper` namespace; [`test_colmap_uses_incremental_mapper_and_mapper_options()`](../../../tests/test_sfm.py:131) checks `mapper` and only the `Mapper` namespace. The current [`run_mapper()`](../../../src/gsforge/sfm.py:505) contains no automatic method switch.

5. **Official-binary evidence and real-data acceptance are present.** The evidence artifact identifies the Windows COLMAP 4.1.1 CUDA binary, commit, host GPU, driver, and CUDA runtime, and records the `global_mapper -h` GPU options and absence of a Caspar selector. The latest execution-log entry records the real command, the required UTF-8 console workaround, supported capability probes, successful GLOMAP completion, 185 registered cameras, selected `sfm/sparse/0`, persisted completed state, and the observed Ceres CUDA/cuDSS CPU fallback. The linked runtime transcript contains the fallback messages, Ceres convergence, global-pipeline completion, and 185-camera result at [`garden-glomap-sfm.txt`](../../../garden-glomap-sfm.txt:1210).

6. **The Unicode-console symptom is correctly classified outside WO-003.** The recorded `PYTHONIOENCODING=utf-8` workaround affected Windows Rich-console startup before SfM, not mapper selection or solver reporting. The initiative explicitly excludes general Windows Unicode-console behavior, so this remains a separately documented portability follow-up rather than a WO-003 technical defect or acceptance blocker.

## Warnings and boundaries

- The CPU warning detector intentionally recognizes the known official messages by bounded text fragments and is not a general solver telemetry classifier. This is a scope limitation, not a failure of the approved high-level contract.
- The external evidence artifact is concise documentary evidence rather than a complete raw-stream archive for every probe. The real-data gate is nevertheless sufficiently identified and linked for this high-level WO-003 acceptance review; future evidence should retain raw bounded streams if reproducibility requirements are raised.
- The warning reports the solver limitation transiently in the run output; it does not persist a solver-mode field in project metadata. Persistent structured solver state is outside the revised high-level contract and is not required for this disposition.

## Separate metadata / closeout disposition

The technical WO-003 implementation disposition is `pass-with-warnings`. Any stale WO-003 metadata, unchecked completion checklist, or later closeout reconciliation is separate from technical correctness and must not be presented as a mapper, warning-classification, failure-persistence, dispatch, or real-data defect. This review does not self-mark the work order complete or alter initiative metadata.

## Decision

- **Technical disposition:** `pass-with-warnings`.
- **Technical result:** The approved official-binary GLOMAP CPU-fallback reporting behavior is implemented and demonstrated by focused tests plus the recorded 4.1.1 185-frame run.
- **Required technical action:** None for the revised WO-003 scope. Preserve the current high-level boundary and do not claim GPU BA for official binaries or compatibility for custom builds.
- **Separate closeout action:** Reconcile work-order/checklist metadata and any initiative closeout records independently of this technical result.
