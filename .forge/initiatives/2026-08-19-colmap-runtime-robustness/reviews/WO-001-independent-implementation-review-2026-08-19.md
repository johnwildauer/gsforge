# Independent Implementation Review — WO-001 Resilient COLMAP Capability and Version Probe

## Review metadata

- **Review type:** independent implementation review after human-approved revised design
- **Status:** `blocked`
- **Reviewer:** independent Forge architect/reviewer
- **Date:** 2026-08-19
- **Branch:** `initiative/2026-08-19-colmap-runtime-robustness` as recorded in [`initiative-v1.md`](../initiative-v1.md:11)
- **Reviewed work order:** [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:1)
- **Implementation:** [`sfm.py`](../../../src/gsforge/sfm.py:180)
- **Tests:** [`test_sfm.py`](../../../tests/test_sfm.py:147)
- **Prior reviews considered:** [`WO-001-implementation-review.md`](WO-001-implementation-review.md:1), [`WO-001-fresh-implementation-review.md`](WO-001-fresh-implementation-review.md:1), [`WO-001-final-independent-review.md`](WO-001-final-independent-review.md:1), and [`WO-001-final-independent-review-after-branch-switch.md`](WO-001-final-independent-review-after-branch-switch.md:1)

## Scope and validation basis

I reviewed only WO-001. I read the current WO-001 contract, current implementation and tests, prior WO-001 reviews, the initiative execution log, revised initiative design and approval record, reconstruction requirements and blueprints, and the Forge process. WO-003 implementation was not reviewed.

No production code or tests were modified. The user-provided focused-validation result of 33 tests, the recorded full-suite/Ruff results, and external binary evidence were assessed as documentary evidence; they were not rerun in this file-inspection environment because command execution is unavailable under the repository policy in [`00-overview.md`](../../../process/00-overview.md:70).

## Acceptance assessment

### Confirmed

- **Rejected-version semantics and version invocation:** [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:180) preserves `version_status="unsupported"` when `--version` exits nonzero, even if version text is recovered. It probes both the documented `version` command and `-h`. [`test_help_is_authoritative_when_version_is_unsupported()`](../../../tests/test_sfm.py:148) asserts the `--version`, `version`, and `-h` invocations and the required unsupported state.
- **Semantic `help` gate:** The top-level `help` response is not accepted merely because it exits successfully. The implementation requires COLMAP text, an available-commands marker, `mapper`, and `global_mapper`; [`test_empty_successful_help_is_unusable()`](../../../tests/test_sfm.py:189) verifies that empty successful output is fatal. This resolves the earlier empty-help defect for the tested contract.
- **Raw evidence at the runtime boundary:** [`run_sfm()`](../../../src/gsforge/sfm.py:831) writes `project.logs_dir / "colmap-capability-probe.json"` after probing. The file contains the normalized fields and the bounded [`ColmapCapabilityProbe.raw_evidence`](../../../src/gsforge/sfm.py:111), including arguments, return codes, stdout/stderr, and captured error details where available. [`test_rejected_version_metadata_does_not_stop_pipeline()`](../../../tests/test_sfm.py:293) verifies that the runtime-boundary file is created.
- **Continuation integration:** [`test_rejected_version_metadata_does_not_stop_pipeline()`](../../../tests/test_sfm.py:293) proves that usable help plus rejected version metadata continues through the mocked extraction, matching, mapping, model selection, and completion path, while preserving the selected method.
- **Mapper-failure integration:** [`test_mapper_failure_persists_failed_state_without_switching_method()`](../../../tests/test_sfm.py:316) proves that a selected mapper failure persists `failed` state, zero cameras, and the requested `glomap` method rather than silently switching dispatch.
- **Fatal-probe state persistence:** [`test_unusable_probe_persists_failed_state()`](../../../tests/test_sfm.py:336) proves that an unavailable probe reaches the failed-state update before the fatal diagnostic exits. The production path checks `binary_available` before downstream pipeline calls at [`run_sfm()`](../../../src/gsforge/sfm.py:853).
- **Scope boundary:** The reviewed WO-001 changes do not replace mapper dispatch or add focal-calibration behavior. The global/incremental forms remain in [`run_mapper()`](../../../src/gsforge/sfm.py:462), and calibration remains outside WO-001 as required by [`initiative-v1.md`](../initiative-v1.md:18).

### Remaining blockers

1. **Top-level `-h` is invoked but not semantically validated or represented in the normalized command contract.** [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:220) calls [`run_probe()`](../../../src/gsforge/sfm.py:185) for `-h` but discards its result. The test at [`test_help_is_authoritative_when_version_is_unsupported()`](../../../tests/test_sfm.py:148) proves invocation only. A binary can therefore pass semantic `help` while top-level `-h` fails or returns unusable content without affecting the normalized result; the work-order requirement to inspect and record `-h` behavior is only partially satisfied.

2. **The required per-command `unavailable` state is still lost.** A launch failure returns `status="unavailable"` from [`run_probe()`](../../../src/gsforge/sfm.py:185), but the subcommand mapping at [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:264) maps every non-success/non-unsupported outcome to `unknown`. The fatal top-level-help return at [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:229) also omits the required command entries. This fails the normalized four-state contract in [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:27).

3. **Fatal integration coverage does not assert downstream suppression.** [`test_unusable_probe_persists_failed_state()`](../../../tests/test_sfm.py:336) verifies persistence and uses `log_error` to terminate, but does not mock and assert that feature extraction, matching, and mapping were not called. The production control flow appears to suppress them, but the fatal integration rule requires a durable test of that boundary.

## Evidence and validation classification

- The reported 33 focused tests are consistent with the current added WO-001 coverage, but remain documentary evidence in this review because they were not rerun.
- Runtime-boundary JSON persistence is implemented and is no longer the prior raw-evidence blocker. The focused test checks file existence rather than serialized content; this is a lower-priority evidence-strengthening gap because the source serializes the complete probe object.
- No requirements or blueprint update is required. The behavior aligns with [`NFR-RECON-001`](../../../requirements/features/REQ-RECONSTRUCTION.md:48), [`reconstruction-pipeline.md`](../../../blueprints/components/reconstruction-pipeline.md:43), and the [`run_sfm()`](../../../blueprints/features/reconstruction.md:5) flow.
- Human design approval and branch evidence are recorded in [`execution-log.md`](../execution-log.md:132). The prior process-gate blocker is therefore closed for this review. WO-001 metadata/checklist remains `draft`/unchecked at [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:3); that is an orchestrator reconciliation item, not a new technical finding.

## Decision

- **WO-001 technical disposition:** `blocked` by the incomplete top-level `-h` contract, loss of normalized `unavailable` command states, and incomplete fatal-boundary assertions.
- **Recognized remediation:** rejected-version continuation, semantic `help`, documented `version` probing, raw evidence file creation at the runtime boundary, mapper-failure persistence, and scope preservation are accepted as implemented within the reviewed evidence.
- **Required next state:** retain/validate the top-level `-h` result, preserve complete per-command normalized states including `unavailable`, and assert that fatal probe handling prevents all downstream SfM steps. This disposition is limited to WO-001.
