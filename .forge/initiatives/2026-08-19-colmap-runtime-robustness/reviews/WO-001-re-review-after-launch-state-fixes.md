# Independent Re-review — WO-001 After Launch-State and Test Fixes

## Review metadata

- **Review type:** independent WO-001 implementation re-review
- **Status:** `pass-with-warnings`
- **Reviewer:** independent Forge architect/reviewer
- **Date:** 2026-08-19
- **Scope:** WO-001 only; WO-003 is excluded
- **Contract:** [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:1)
- **Implementation:** [`sfm.py`](../../../src/gsforge/sfm.py:180)
- **Tests:** [`test_sfm.py`](../../../tests/test_sfm.py:148)
- **Prior review:** [`WO-001-re-review-after-latest-fixes.md`](WO-001-re-review-after-latest-fixes.md:1)

## Scope and validation basis

I re-read only the active WO-001 contract, current implementation and tests, prior WO-001 reviews, the approval/branch record, and the latest focused validation evidence. No production code or tests were modified. The reported 34 focused tests and scoped quality results were assessed as documentary evidence rather than rerun in this review environment.

## Technical acceptance assessment

### Confirmed

- **Top-level `-h` semantics and assertions:** [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:223) probes `-h`, requires the available-commands marker and `global_mapper`, and retains the result in `commands["-h"]`. The focused capability test also asserts a positive supported `-h` state at [`test_colmap_411_global_mapper_gpu_contract_is_captured()`](../../../tests/test_sfm.py:244). The separate top-level `help` gate requires COLMAP text, available commands, `mapper`, and `global_mapper`.
- **Complete command-state mapping:** The usable-help path now assigns the nested probe's returned state directly at [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:286), preserving `supported`, `unsupported`, `unknown`, and `unavailable` rather than collapsing launch failures to `unknown`. The fatal unusable-help path supplies explicit entries for `help`, `-h`, and all required reconstruction commands at [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:229).
- **Raw evidence JSON retention:** [`run_sfm()`](../../../src/gsforge/sfm.py:859) serializes `raw_evidence` to `logs/colmap-capability-probe.json`, while [`run_probe()`](../../../src/gsforge/sfm.py:204) bounds stdout/stderr and retains arguments, return codes, and launch/timeout error details. This meets the runtime-boundary retention requirement.
- **Fatal persistence and downstream suppression:** The fatal probe path updates the project to failed with the requested method and zero cameras before emitting the fatal diagnostic at [`run_sfm()`](../../../src/gsforge/sfm.py:873). The focused integration test [`test_unusable_probe_persists_failed_state()`](../../../tests/test_sfm.py:351) asserts feature extraction, matching, and mapping are not called.
- **Version/help behavior:** A rejected `--version` remains `version_status="unsupported"` even when help or the documented `version` command supplies a version string. Usable help is authoritative, absent version metadata is non-fatal, and the `version` command plus top-level `-h` are probed.
- **Mapper dispatch:** [`run_mapper()`](../../../src/gsforge/sfm.py:482) still maps `glomap` to `global_mapper` with `GlobalMapper` options and `colmap` to `mapper` with `Mapper` options. No automatic mapper substitution was introduced.

## Validation classification

- The latest focused validation reports 34 tests passing. The recorded scoped Ruff check/format results remain passing for [`sfm.py`](../../../src/gsforge/sfm.py:1) and [`test_sfm.py`](../../../tests/test_sfm.py:1). These results were not independently rerun.
- A lower-priority test-strengthening warning remains: the positive `-h` assertion checks the normalized state through a broader fixture, while the JSON test verifies the `raw_evidence` key rather than separately asserting serialized stdout, stderr, arguments, and return-code fields. The implementation itself retains those fields and the current focused tests cover the launch-failure state and fatal integration behavior.
- No technical blocker remains within WO-001's reviewed scope.

## Process/documentation disposition

- Human design approval and initiative-branch activation are recognized as recorded in [`execution-log.md`](../execution-log.md:132); they are not technical blockers for this review.
- The active WO-001 metadata still says `draft` and its completion checklist remains unchecked in [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:3). This is a separate work-order metadata reconciliation issue, not a technical failure of the implementation.
- No requirements or blueprint change is indicated.

## Decision

- **Technical disposition:** `pass-with-warnings` for WO-001. The launch-state mapping fix resolves the prior technical blocker; the required probe behavior, persistence boundaries, failure handling, version/help semantics, and mapper dispatch are confirmed from the current source and tests.
- **Separate metadata issue:** Reconcile WO-001's stale `draft` status and unchecked completion checklist with the approved/executed state. Do not treat that metadata drift as a new technical blocker.
