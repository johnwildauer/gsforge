# Fresh Independent Review — WO-001 Resilient COLMAP Capability and Version Probe

## Review metadata

- **Review type:** `fresh independent implementation review`
- **Status:** `blocked`
- **Reviewer:** `independent Forge architect/reviewer`
- **Date:** `2026-08-19`
- **Reviewed work order:** [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md)
- **Reviewed implementation:** [`sfm.py`](../../../src/gsforge/sfm.py) and [`test_sfm.py`](../../../tests/test_sfm.py)
- **Validation policy:** Logged validation and external-probe claims were treated as documentary evidence only; commands and binaries were not rerun.

## Scope and evidence reviewed

This review reread the active work order, the latest execution log, the prior blocked review, current implementation and tests, [`README.md`](../../../README.md), [`REQ-RECONSTRUCTION.md`](../../../requirements/features/REQ-RECONSTRUCTION.md), the reconstruction blueprints, the initiative gate record, and the available COLMAP runtime logs. The remediation claims at [`execution-log.md`](../execution-log.md:110) were checked against the current workspace rather than accepted as proof by assertion.

## Findings

### Technical blockers

1. **Rejected-version semantics are now correct.** When `--version` exits nonzero but semantically usable `help` succeeds, [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:179) returns `binary_available=True` and `version_status="unsupported"`, even if a version string is recovered from help or the `version` command. The focused test at [`test_help_is_authoritative_when_version_is_unsupported()`](../../../tests/test_sfm.py:135) verifies this remediation.

2. **Semantic top-level help validation is present and empty help is rejected.** The implementation requires COLMAP text, an available-commands marker, `mapper`, and `global_mapper`; the empty-help test at [`test_empty_successful_help_is_unusable()`](../../../tests/test_sfm.py:170) verifies the fatal classification. This resolves the prior empty-help blocker for the tested shape.

3. **The normalized per-command state contract is still not fully implemented.** A nonzero subcommand probe is converted to `unknown` when its output is not semantically usable at [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:262), rather than retaining the documented `unsupported` state. The implementation also uses a generic `options` substring without confirming that the response belongs to the requested command. This leaves `unsupported`, `unknown`, and `unavailable` observably conflated and prevents acceptance of the complete normalized contract.

4. **Raw evidence retention is still incomplete.** [`run_sfm()`](../../../src/gsforge/sfm.py:829) logs only [`ColmapCapabilityProbe.evidence_summary()`](../../../src/gsforge/sfm.py:113), which contains statuses and the diagnostic but not the bounded stdout, stderr, return codes, arguments, or exception details held in `raw_evidence`. No linked artifact in the initiative contains the latest raw probe transcript. A summary is useful, but it does not satisfy the work order's evidence-retention requirement.

5. **Integration and failure-state coverage remains inadequate.** Current tests exercise the probe directly and mapper dispatch, but do not prove that [`run_sfm()`](../../../src/gsforge/sfm.py:772) continues after rejected version metadata, records failed state for an unusable probe, or preserves failed state when the selected mapper fails. The remediation claim of 83 passing tests is recorded at [`execution-log.md`](../execution-log.md:114), but the required behavior is not traceable to focused assertions in [`test_sfm.py`](../../../tests/test_sfm.py:134).

### Warnings

- The documented `version` command is now probed at [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:236), and the existing test fixture responds to it, but the test does not explicitly assert that the command was invoked. A call-sequence assertion would make this acceptance condition durable.
- The work-order evidence scope names top-level `-h`, but the current implementation probes `help` and subcommand `-h`, not top-level `-h`. This is a scope/evidence gap unless the work order is narrowed explicitly.
- The current public compatibility helper [`check_colmap_version()`](../../../src/gsforge/sfm.py:283) still collapses all non-known version results to `"unknown"`. This is acceptable only for backward compatibility because the structured probe is the active integration path; callers must not use this helper as the normalized contract.

## Process-gate disposition

The initiative remains `planning-revision` at [`initiative-v1.md`](../initiative-v1.md:9), its design-readiness checklist still lacks independent design approval and human design approval at [`initiative-v1.md`](../initiative-v1.md:90), and the execution log still says branch creation evidence is required at [`execution-log.md`](../execution-log.md:9). No explicit human approval or gate waiver is recorded. These are independent completion blockers; this review does not self-approve them.

## Decision

- **WO-001 outcome:** `blocked` for incomplete per-command state semantics, raw-evidence retention, and missing `run_sfm()` continuation/failure-state tests.
- **Process outcome:** `blocked` because design approval or an explicit waiver and dedicated-branch evidence remain absent.
- **Technical progress recognized:** rejected-version semantics, semantic help rejection, version-command probing, and the existing mapper dispatch boundary are materially improved and not reclassified as failures.
- **Required next action:** Correct the state mapping, retain or link bounded raw evidence, add integration/failure-state assertions, and obtain the missing process authorization/evidence before another review.
