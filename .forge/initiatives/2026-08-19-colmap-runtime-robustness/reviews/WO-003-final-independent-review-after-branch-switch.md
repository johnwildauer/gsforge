# Final Independent Review — WO-003 After Branch Switch

## Review metadata

- **Review type:** final independent implementation review after branch switch and exact fixture update
- **Status:** `blocked`
- **Reviewer:** independent Forge architect/reviewer
- **Date:** 2026-08-19
- **Reviewed work order:** [`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md:1)
- **Implementation:** [`sfm.py`](../../../src/gsforge/sfm.py:300)
- **Tests:** [`test_sfm.py`](../../../tests/test_sfm.py:199)
- **External evidence:** [`WO-003-external-evidence-4.1.1.md`](WO-003-external-evidence-4.1.1.md:1)
- **Prior review:** [`WO-003-final-independent-review.md`](WO-003-final-independent-review.md:1)

## Scope and validation basis

I reread the current initiative/work order, prior implementation and final reviews, execution log, requirements, blueprints, README, Forge process, current [`sfm.py`](../../../src/gsforge/sfm.py:300), [`test_sfm.py`](../../../tests/test_sfm.py:199), the exact 4.1.1 fixture, [`WO-003-external-evidence-4.1.1.md`](WO-003-external-evidence-4.1.1.md:1), and [`garden-glomap-sfm.txt`](../../../garden-glomap-sfm.txt:1). The log records the branch switch and exact fixture validation at [`execution-log.md`](../execution-log.md:126), but no command-execution tool is available here. Logged test/lint/external results were assessed as documentary evidence and not rerun.

No production code or tests were changed by this review.

## Technical disposition

### Confirmed

- The stakeholder-approved high-level contract is represented: GLOMAP continues through `global_mapper`, requests `GlobalMapper.gp_use_gpu=1` and `GlobalMapper.ba_ceres_use_gpu=1`, captures mapper output, and emits a truthful warning for the known CUDA/cuDSS fallback in [`_run_colmap_step()`](../../../src/gsforge/sfm.py:300).
- The positive fallback-warning test is present at [`test_cpu_fallback_warning_is_reported_for_glomap()`](../../../tests/test_sfm.py:199).
- Mapper failure persistence and no automatic switch are covered by [`test_mapper_failure_persists_failed_state_without_switching_method()`](../../../tests/test_sfm.py:292). Dispatch separation is covered by [`TestRunMapper`](../../../tests/test_sfm.py:115).
- The exact 4.1.1 fixture asserts the GPU request options and absence of a Caspar selector at [`test_colmap_411_global_mapper_gpu_contract_is_captured()`](../../../tests/test_sfm.py:220). The runtime log independently links the 185-camera completion and CPU solver warnings to [`garden-glomap-sfm.txt`](../../../garden-glomap-sfm.txt:1210).

### Remaining technical findings

1. **High-level CPU fallback warning coverage is still narrow.** There is no positive/no-warning test for a normal successful Ceres solver report, no direct cuDSS-only boundary test, and no explicit assertion that ordinary success remains warning-free. The narrow test is sufficient to show one known warning path, but not full positive/no-warning boundary coverage requested by the remaining finding.
2. **External evidence linkage is useful but not reproducibly complete.** [`WO-003-external-evidence-4.1.1.md`](WO-003-external-evidence-4.1.1.md:11) records selected commands, statuses, options, environment, and links the runtime log, but does not retain raw stdout/stderr for the claimed probes, complete command metadata for every probe, or an isolated 4.1.1 run transcript with command, exit status, and complete solver streams. The linked [`garden-glomap-sfm.txt`](../../../garden-glomap-sfm.txt:1210) is strong runtime corroboration, but it is not a reproducible raw external-gate artifact.

No mapper-failure persistence or dispatch defect remains in the current source/tests. The exact fixture materially resolves the prior concern about the 4.1.1 global-mapper option contract, but it does not resolve the raw external-evidence gate or warning-boundary coverage.

## Process-gate disposition

- The initiative remains `planning-revision`; independent design approval and human design approval remain unchecked in [`initiative-v1.md`](../initiative-v1.md:84).
- Dedicated-branch creation evidence remains absent even though the branch switch is recorded at [`execution-log.md`](../execution-log.md:126).
- WO-003 remains `draft` with an unchecked completion checklist in [`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md:5).
- No human design approval or explicit waiver is recorded. These process gates are separate from technical correctness and are not self-approved under [`00-overview.md`](../../../process/00-overview.md:80).

## Decision

- **WO-003 technical outcome:** `blocked` by incomplete reproducible external evidence; warning/no-warning coverage remains an additional technical gap.
- **WO-003 process outcome:** `blocked` independently by missing design authorization/waiver and branch evidence.
- **Required next action:** retain bounded raw 4.1.1 probe/run streams with complete metadata, add the high-level warning/no-warning boundary test, then obtain the required process approvals or an explicitly recorded waiver. The mapper failure and dispatch findings are closed as verified.
