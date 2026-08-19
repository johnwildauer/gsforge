# Final Independent Review — WO-001 After Branch Switch

## Review metadata

- **Review type:** final independent implementation review after branch switch and exact fixture update
- **Status:** `blocked`
- **Reviewer:** independent Forge architect/reviewer
- **Date:** 2026-08-19
- **Reviewed work order:** [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:1)
- **Implementation:** [`sfm.py`](../../../src/gsforge/sfm.py:179)
- **Tests:** [`test_sfm.py`](../../../tests/test_sfm.py:147)
- **Prior review:** [`WO-001-final-independent-review.md`](WO-001-final-independent-review.md:1)

## Scope and validation basis

I reread the current initiative/work orders, all prior WO-001/WO-003 review artifacts, execution log, requirements, blueprints, README, Forge process, current [`sfm.py`](../../../src/gsforge/sfm.py:179), [`test_sfm.py`](../../../tests/test_sfm.py:147), the exact COLMAP 4.1.1 fixture, and linked runtime evidence. The execution log records the requested branch switch at [`execution-log.md`](../execution-log.md:126), but this file-inspection environment provides no VCS-status or command-execution tool. Recorded tests and lint results are therefore documentary evidence, not independently rerun validation.

No production code or tests were changed by this review.

## Technical disposition

### Confirmed

- [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:179) preserves `version_status="unsupported"` when `--version` is rejected, probes the documented `version` command, rejects semantically unusable top-level `help`, and retains bounded evidence in the returned [`ColmapCapabilityProbe`](../../../src/gsforge/sfm.py:103).
- The rejected-version continuation test and mapper-failure persistence test cover the non-fatal metadata path and selected mapper failure boundary in [`test_sfm.py`](../../../tests/test_sfm.py:270).
- Mapper dispatch remains separated between `global_mapper`/`GlobalMapper` and `mapper`/`Mapper` in [`run_mapper()`](../../../src/gsforge/sfm.py:461).

### Remaining blockers

1. **Top-level `-h` assertion gap.** [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:219) invokes top-level `-h` but discards its output, status, and semantic result. [`test_help_is_authoritative_when_version_is_unsupported()`](../../../tests/test_sfm.py:148) asserts invocation only. The required top-level `-h` contract is not verified.
2. **Incomplete per-command normalized mapping.** Launch failures return `unavailable` from the nested probe but are converted to `unknown` by the command mapping in [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:255). Fatal top-level results also omit the complete required command map. `supported`, `unsupported`, `unknown`, and `unavailable` are therefore not retained as the documented normalized states for all required commands.
3. **Raw evidence is not retained at the runtime boundary.** [`run_sfm()`](../../../src/gsforge/sfm.py:834) writes only [`evidence_summary()`](../../../src/gsforge/sfm.py:113), which omits bounded stdout/stderr, arguments, return codes, and exception details. The returned raw evidence is not persisted or linked to a WO-001 transcript artifact.
4. **Fatal probe integration is untested.** [`test_unusable_help_marks_binary_unavailable()`](../../../tests/test_sfm.py:179) tests the probe directly, but no [`run_sfm()`](../../../src/gsforge/sfm.py:778) test proves that `binary_available=False` persists failed state, emits the diagnostic, and prevents feature extraction, matching, and mapping.

These are technical acceptance/coverage blockers, not merely documentary preferences, because they map directly to the normalized contract, raw-evidence requirement, fatal integration rule, and NFR failed-stage requirement in [`REQ-RECONSTRUCTION.md`](../../../requirements/features/REQ-RECONSTRUCTION.md:48).

## Process-gate disposition

- The initiative remains `planning-revision`; its design-readiness checklist still lacks independent design approval and human design approval in [`initiative-v1.md`](../initiative-v1.md:84).
- Dedicated-branch creation evidence is not independently established. The log records a switch to the required branch at [`execution-log.md`](../execution-log.md:126), but does not provide branch-creation evidence.
- WO-001 remains `draft` with an unchecked completion checklist in [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:5).
- No human design approval or explicit waiver is recorded. Under [`00-overview.md`](../../../process/00-overview.md:80), this reviewer does not self-approve those gates.

## Decision

- **WO-001 technical outcome:** `blocked` by the four findings above.
- **WO-001 process outcome:** `blocked` independently by missing design authorization/waiver and branch evidence.
- **Required next action:** define and assert top-level `-h`, preserve all normalized command states, retain/link raw probe evidence, add fatal [`run_sfm()`](../../../src/gsforge/sfm.py:778) integration coverage, then obtain the required independent design/human gate decisions. The current 4.1.1 fixture does not resolve these WO-001 boundary findings.
