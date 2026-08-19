# Final Fresh Independent Review — WO-001 Resilient COLMAP Capability and Version Probe

## Review metadata

- **Review type:** `final fresh independent implementation review`
- **Status:** `blocked`
- **Reviewer:** `independent Forge architect/reviewer`
- **Date:** 2026-08-19
- **Reviewed work order:** [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:1)
- **Reviewed implementation:** [`sfm.py`](../../../src/gsforge/sfm.py:179) and [`test_sfm.py`](../../../tests/test_sfm.py:137)
- **Prior review superseded for final assessment:** [`WO-001-fresh-implementation-review.md`](WO-001-fresh-implementation-review.md:1)

## Scope and validation basis

I independently reread the active work order, current implementation and tests, README, requirements, reconstruction blueprints, Forge process, execution log, prior reviews, and the linked external evidence. No production code or tests were changed. No command-execution tool is available in this review session, so the logged test/lint results and external probe were assessed as documentary evidence rather than rerun; this limitation is consistent with the repository validation policy in [`00-overview.md`](../../../process/00-overview.md:70).

## Verified implementation behavior

- **Rejected-version state:** [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:179) preserves `version_status="unsupported"` when `--version` returns nonzero, even if version text is recovered from `help` or `version`. [`test_help_is_authoritative_when_version_is_unsupported()`](../../../tests/test_sfm.py:138) asserts this state and confirms both metadata commands and `-h` are invoked.
- **Semantic top-level help:** `help` is rejected unless the combined output contains COLMAP, an available-commands marker, `mapper`, and `global_mapper`; empty successful help is covered by [`test_empty_successful_help_is_unusable()`](../../../tests/test_sfm.py:179). This is materially correct for the required top-level semantic gate.
- **Continuation and mapper failure:** [`test_rejected_version_metadata_does_not_stop_pipeline()`](../../../tests/test_sfm.py:230) proves the nonfatal rejected-version path continues to completion. [`test_mapper_failure_persists_failed_state_without_switching_method()`](../../../tests/test_sfm.py:252) proves a mapper `SystemExit` persists failed state, preserves `glomap`, and records zero cameras.

## Remaining technical findings

### Blocker T1 — top-level `-h` is probed but not validated or used

[`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:219) invokes `-h`, but discards its success, output, and semantic content. The required top-level `-h` probing is therefore only an evidence collection side effect; a binary could provide valid `help` output while `-h` fails or is unusable without affecting the normalized result. The test checks invocation, not semantics. Required action: either make `help` and `-h` an explicitly defined equivalent probe with validation, or narrow the work-order acceptance criterion and preserve the rationale in the evidence contract.

### Blocker T2 — normalized `unavailable` command state is not retained

For subcommands, [`run_probe()`](../../../src/gsforge/sfm.py:184) returns `unavailable` on launch failure, but the mapping at [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:263) converts every non-success/non-unsupported result to `unknown`. Thus the documented per-command states do not distinguish launch unavailability from an indeterminate command response. The implementation also omits command entries when top-level help is fatal. Required action: preserve `unavailable` for launch failures and define complete entries for all required commands in every normalized result shape.

### Blocker T3 — raw evidence is not retained at the `run_sfm()` boundary

[`ColmapCapabilityProbe.raw_evidence`](../../../src/gsforge/sfm.py:109) does contain bounded output, arguments, exit status, and errors, but [`run_sfm()`](../../../src/gsforge/sfm.py:834) logs only [`evidence_summary()`](../../../src/gsforge/sfm.py:113), which contains statuses and a diagnostic, not the bounded streams or return codes. No execution-log-linked artifact retains the raw WO-001 probe transcript. This does not satisfy the work-order requirement to preserve raw command evidence. Required action: retain or link the bounded evidence in a review/log artifact, or explicitly establish an approved persistence destination.

### Finding T4 — fatal probe integration is not tested

The current test set covers timeout behavior directly and covers rejected-version continuation, but does not test [`run_sfm()`](../../../src/gsforge/sfm.py:837) when the probe returns `binary_available=False`, including failed project persistence, diagnostic emission, and no downstream pipeline invocation. This is required by the work-order fatal mapping and the failed-stage requirement in [`REQ-RECONSTRUCTION.md`](../../../requirements/features/REQ-RECONSTRUCTION.md:48).

## Process-gate findings

- The initiative remains `in-progress` at `planning-revision`, and its design-readiness checklist still lacks independent design approval and human design approval in [`initiative-v1.md`](../initiative-v1.md:84).
- Dedicated-branch creation evidence is still absent from [`execution-log.md`](../execution-log.md:9).
- The active work order remains `draft` with an unchecked completion checklist in [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:5).
- The recorded full-suite, Ruff, and external-probe results at [`execution-log.md`](../execution-log.md:118) were not rerun here and therefore remain documentary, not independently executed evidence.

These process findings are separate from the technical findings. This review does not self-approve missing human design approval, a waiver, or branch evidence.

## Decision

- **WO-001 outcome:** `blocked` for T1, T2, and T3; T4 is an additional validation gap.
- **Process outcome:** `blocked` independently of the technical disposition.
- **Recognized remediation:** rejected-version semantics, semantic `help` rejection, `version` probing, rejected-version continuation, mapper-failure state persistence, and preserved dispatch were verified from the current source/tests.
- **Required next state:** retain complete probe evidence, correct/define all per-command state mappings, make the top-level `-h` contract explicit, add fatal-probe integration coverage, and obtain the missing process approvals/evidence before completion.
