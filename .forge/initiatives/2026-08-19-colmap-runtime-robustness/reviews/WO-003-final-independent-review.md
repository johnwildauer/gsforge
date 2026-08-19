# Final Fresh Independent Review — WO-003 GLOMAP CPU Fallback Contract

## Review metadata

- **Review type:** `final fresh independent implementation review`
- **Status:** `blocked`
- **Reviewer:** `independent Forge architect/reviewer`
- **Date:** 2026-08-19
- **Reviewed work order:** [`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md:1)
- **Reviewed implementation:** [`sfm.py`](../../../src/gsforge/sfm.py:300) and [`test_sfm.py`](../../../tests/test_sfm.py:189)
- **Related evidence:** [`WO-003-external-evidence-4.1.1.md`](WO-003-external-evidence-4.1.1.md:1)
- **Prior review superseded for final assessment:** [`WO-003-fresh-implementation-review.md`](WO-003-fresh-implementation-review.md:1)

## Scope and validation basis

I independently reread the approved high-level CPU-fallback contract, current implementation/tests, README, requirements, reconstruction blueprints, Forge process, execution log, prior reviews, [`WO-003-external-evidence-4.1.1.md`](WO-003-external-evidence-4.1.1.md:1), and the linked 185-frame runtime log. No production code or tests were changed. No command-execution tool is available in this session; logged automated and external validation was therefore assessed, not rerun.

## Verified implementation behavior

- **High-level contract:** [`run_mapper()`](../../../src/gsforge/sfm.py:461) keeps GLOMAP on `global_mapper`, requests `GlobalMapper.gp_use_gpu=1` and `GlobalMapper.ba_ceres_use_gpu=1`, captures global-mapper output, and emits a truthful high-level CPU BA warning when the known CUDA/cuDSS messages occur. It does not introduce structured telemetry or claim GPU BA.
- **Representative warning:** [`test_cpu_fallback_warning_is_reported_for_glomap()`](../../../tests/test_sfm.py:189) verifies the representative CUDA warning produces a CPU bundle-adjustment warning. The implementation also checks the cuDSS fragment at [`_run_colmap_step()`](../../../src/gsforge/sfm.py:344).
- **Mapper failure persistence:** [`test_mapper_failure_persists_failed_state_without_switching_method()`](../../../tests/test_sfm.py:252) verifies failed state, `glomap` method persistence, zero cameras, and no automatic switch when the mapper raises the expected `SystemExit` boundary.
- **Dispatch isolation:** [`test_glomap_uses_global_mapper_and_global_options()`](../../../tests/test_sfm.py:105) and [`test_colmap_uses_incremental_mapper_and_mapper_options()`](../../../tests/test_sfm.py:120) verify separate subcommands and namespaces. No dispatch contamination or automatic GLOMAP-to-incremental fallback was found.
- **External linkage:** The 4.1.1 evidence artifact links the official binary capability claims to [`garden-glomap-sfm.txt`](../../../garden-glomap-sfm.txt:1209), including the observed CPU fallback and the 185-camera completion at [`garden-glomap-sfm.txt`](../../../garden-glomap-sfm.txt:1238).

## Remaining technical findings

### Blocker T1 — external evidence is linked but not reproducibly retained at the stated granularity

[`WO-003-external-evidence-4.1.1.md`](WO-003-external-evidence-4.1.1.md:11) records selected exit statuses and observed options, and links the runtime log, but it does not retain the actual stdout/stderr for `help`, `global_mapper -h`, the other claimed probes, or a complete 4.1.1 command transcript. The runtime linkage is to a UTF-16/encoding-corrupted log presentation, and the artifact does not identify an isolated-project command invocation, exit status, or complete solver transcript. The evidence is useful corroboration but does not meet the work order's reproducible external-gate fields. Required action: attach raw bounded probe streams and a reproducible run transcript or explicitly downgrade the gate to documentary evidence.

### Finding T2 — warning coverage remains narrow, though sufficient to establish the basic path

The one warning test uses an exact CUDA fragment. There is no explicit no-warning test for a normal successful solver report, no direct test for the cuDSS-only fragment, and no case-variation/line-break representative. The high-level contract does not require exhaustive custom-build classification, so this is a technical coverage gap rather than a demand for structured telemetry. The documentation should not imply broader detection than the exact known fragments, or the tests should add representative positive and negative boundaries.

No remaining mapper-failure persistence or dispatch-contamination defect was found in the current tests; those prior findings are resolved by the remediation.

## Process-gate findings

- The initiative remains at `planning-revision` and the design-readiness checklist lacks independent design approval and human design approval in [`initiative-v1.md`](../initiative-v1.md:84).
- Dedicated-branch creation evidence remains absent from [`execution-log.md`](../execution-log.md:9).
- The active work order remains `draft` with an unchecked completion checklist in [`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md:5).
- Logged automated and external validation at [`execution-log.md`](../execution-log.md:118) was not independently rerun in this session.

These process findings are separate from the technical findings. This review does not self-approve missing human design approval, a waiver, or branch evidence.

## Decision

- **WO-003 outcome:** `blocked` by T1; T2 remains a technical coverage finding.
- **Process outcome:** `blocked` independently of technical correctness.
- **Recognized remediation:** the stakeholder-approved high-level CPU fallback contract is represented truthfully; GLOMAP dispatch, requested GPU settings, mapper failure persistence, method mapping, and no-auto-switch behavior are aligned.
- **Required next state:** retain reproducible 4.1.1 evidence, optionally strengthen warning-boundary tests without expanding scope, and obtain the missing process approvals/evidence before completion.
