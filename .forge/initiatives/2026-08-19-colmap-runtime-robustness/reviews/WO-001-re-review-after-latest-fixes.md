# Independent Re-review — WO-001 After Latest Fixes

## Review metadata

- **Review type:** independent WO-001 implementation re-review after latest fixes
- **Status:** `blocked`
- **Reviewer:** independent Forge architect/reviewer
- **Date:** 2026-08-19
- **Scope:** WO-001 only; WO-003 is excluded
- **Contract:** [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:1)
- **Implementation:** [`sfm.py`](../../../src/gsforge/sfm.py:180)
- **Tests:** [`test_sfm.py`](../../../tests/test_sfm.py:147)
- **Prior review:** [`WO-001-independent-implementation-review-2026-08-19.md`](WO-001-independent-implementation-review-2026-08-19.md:1)

## Scope and validation basis

I re-read only the active WO-001 contract, current implementation and tests, prior WO-001 reviews, the initiative approval/branch record, and the latest execution-log validation. No production code or tests were modified. The recorded focused/full test and Ruff results were assessed as documentary evidence and were not rerun in this review environment.

## Technical acceptance assessment

### Confirmed

- **Top-level help semantics:** [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:180) now evaluates top-level `-h` output for the available-commands marker and `global_mapper`, and records the normalized `-h` result instead of discarding it. The top-level `help` gate also requires COLMAP text, available commands, `mapper`, and `global_mapper`. This provides the required semantic evidence path, although the focused test currently proves `-h` invocation rather than a positive semantic `-h` assertion.
- **Unavailable fatal state:** An unusable top-level help result now returns a complete command map with `help`, `-h`, and all required reconstruction commands represented as `unavailable`; the fatal result is therefore no longer structurally incomplete.
- **Raw evidence retention:** [`run_sfm()`](../../../src/gsforge/sfm.py:801) writes `logs/colmap-capability-probe.json` containing normalized fields plus [`ColmapCapabilityProbe.raw_evidence`](../../../src/gsforge/sfm.py:104), including bounded arguments, return codes, stdout, stderr, and launch/timeout error details. This satisfies runtime-boundary retention; the current test checks artifact existence rather than its serialized contents.
- **Fatal probe persistence and downstream suppression:** The fatal probe branch updates project state to `failed`, preserves the requested method, records zero cameras, and terminates before feature extraction, matching, or mapping. [`test_unusable_probe_persists_failed_state()`](../../../tests/test_sfm.py:337) now asserts all three downstream calls are absent.
- **Version/help behavior:** A rejected `--version` remains `version_status="unsupported"` even when version text is recovered. The documented `version` command and top-level `-h` are probed, while usable help remains authoritative and missing version metadata is non-fatal.
- **Mapper dispatch preservation:** [`run_mapper()`](../../../src/gsforge/sfm.py:484) still dispatches `glomap` to `global_mapper` with `GlobalMapper` options and `colmap` to `mapper` with `Mapper` options. No automatic mapper substitution or calibration behavior was introduced.

### Remaining technical blocker

1. **Launch failures still lose the required `unavailable` per-command state.** [`run_probe()`](../../../src/gsforge/sfm.py:185) correctly records `status="unavailable"`, but the normal subcommand mapping at [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:286) maps every result other than supported or explicitly unsupported to `unknown`. A command that cannot be launched is therefore indistinguishable from an indeterminate probe, violating the contract's four-state vocabulary. The fatal top-level-help map is complete, but the non-fatal usable-help path still has this defect for individual subcommands.

## Validation classification

- The latest log records 86 full-suite tests and passing scoped Ruff checks for [`sfm.py`](../../../src/gsforge/sfm.py:1) and [`test_sfm.py`](../../../tests/test_sfm.py:1). Those results were not independently rerun.
- The test suite contains the newly required fatal-state persistence and downstream-suppression assertions. A lower-priority test-strengthening gap remains: no focused test asserts a semantically valid top-level `-h` result or checks the JSON payload's raw stdout/stderr and return-code fields.
- No production or test change is recommended by this review; the remaining technical issue is an implementation-contract mismatch in normalized launch-failure mapping.

## Process/documentation disposition

- Human design approval and initiative-branch activation are accepted as recorded in [`execution-log.md`](../execution-log.md:132); they are not blockers for this re-review.
- The active WO-001 metadata still says `draft` and its completion checklist remains unchecked in [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:3). This is a process/documentation reconciliation issue, separate from the technical blocker above.
- The execution log is appended with this review result. No requirements or blueprint change is indicated.

## Decision

- **Technical disposition:** `blocked` solely by loss of the normalized `unavailable` state for launch-failed individual command probes.
- **Recognized completed behavior:** top-level `-h` semantic handling, complete fatal command-state structure, raw evidence JSON retention, fatal persistence and downstream suppression, version/help behavior, and mapper dispatch preservation are confirmed from the current source/tests.
- **Process/documentation disposition:** separate reconciliation remains for the WO-001 `draft` status and unchecked completion checklist; human approval and branch activation are recognized and not re-raised as blockers.
