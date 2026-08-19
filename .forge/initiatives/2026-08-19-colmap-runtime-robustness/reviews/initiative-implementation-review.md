# Initiative Implementation Review — Harden COLMAP Runtime Compatibility

## Review metadata

- **Review type:** `implementation`
- **Status:** `pass-with-warnings`
- **Reviewer:** independent Forge initiative implementation reviewer
- **Date:** 2026-08-19
- **Reviewed artifact:** [`initiative-v1.md`](../initiative-v1.md:1)
- **Reviewed implementation:** [`sfm.py`](../../../src/gsforge/sfm.py:180)
- **Reviewed tests:** [`test_sfm.py`](../../../tests/test_sfm.py:148)
- **Compared evidence:** Current repository contents and all latest initiative/work-order review artifacts; no production code or tests were modified

## Scope of review

This initiative-level review examined the authoritative initiative plan, design review, human approval and branch record, execution log, active work orders, deferred WO-002 record, latest independent WO-001 and WO-003 implementation reviews, prior review history, current SfM implementation and tests, README, external COLMAP evidence, reconstruction requirements and blueprints, and the Forge implementation/closeout process.

The review assesses end-to-end implementation scope, sequencing, validation, documentation, warnings, and readiness for the separate closeout gate. It does not self-close the initiative and does not replace the required closeout review.

## End-to-end assessment

### Scope and sequencing

- WO-001 and WO-003 implement the approved active scope: resilient COLMAP capability discovery and truthful official-binary GLOMAP CPU bundle-adjustment fallback reporting.
- The implementation preserves GLOMAP-to-`global_mapper` and incremental-to-`mapper` dispatch, does not introduce automatic mapper substitution, and does not add focal calibration behavior.
- WO-002 remains explicitly deferred and is not counted toward acceptance.
- The sequence is coherent: capability probing precedes GLOMAP runtime handling, and the required human design approval and branch record are present in the execution log.

### Implementation and failure behavior

- WO-001's latest review confirms rejected `--version` handling, semantic help validation, documented `version` probing, normalized command states, bounded raw evidence persistence at the runtime boundary, non-fatal continuation for usable binaries, fatal failed-state persistence, and downstream-step suppression.
- WO-003's latest review confirms official-binary CPU fallback reporting, explicit GPU-request flags, GLOMAP identity preservation, warning/no-warning coverage, mapper-failure persistence, and no unsafe switch to incremental COLMAP.
- The implementation aligns with the reconstruction requirement for actionable external-binary failure state and the blueprint boundaries owned by the COLMAP runner and `run_sfm()` flow.

### Validation and external evidence

- The execution log records focused and full-suite test results, scoped Ruff checks/formatting, the exact COLMAP 4.1.1 help fixture, and a real 185-frame GLOMAP run.
- The external run reports successful GLOMAP completion, 185 registered cameras, selected `sfm/sparse/0`, persisted completed state, and truthful Ceres CUDA/cuDSS CPU fallback reporting.
- The latest independent WO-001 and WO-003 reviews passed with warnings and found no remaining technical blocker within their approved contracts.
- The recorded evidence was not rerun in this file-inspection environment; this is classified as an evidence limitation, not a failed gate, because the latest independent reviews explicitly assessed the documentary evidence under the repository's unavailable-command-execution policy.

## Blockers

- **None identified for the approved implementation scope.** The latest independent reviews resolved the previously blocking normalized-state, fatal-boundary, mapper-failure, dispatch, warning-boundary, and real-data acceptance findings.

## Warnings and follow-up boundaries

1. **Validation rerun limitation.** Test, Ruff, and external-run results are recorded evidence rather than independently rerun during this review.
2. **WO-001 test assertion depth.** The capability artifact is persisted with bounded arguments, streams, return codes, and errors, but the focused test does not assert every serialized raw-evidence field individually. This is test-strengthening, not an implementation failure.
3. **WO-003 evidence granularity.** [`WO-003-external-evidence-4.1.1.md`](WO-003-external-evidence-4.1.1.md:1) is a concise evidence summary linked to the complete runtime log, not a complete raw stdout/stderr archive for every probe and run command. The latest WO-003 review accepted it for the approved high-level contract.
4. **Bounded solver reporting.** CPU fallback detection is intentionally based on known official warning fragments and reports transiently; it is not general solver telemetry and does not persist a solver-mode field. This is explicitly within the stakeholder-approved WO-003 scope.
5. **Metadata reconciliation.** The initiative design-readiness checklist still contains stale unchecked design-gate entries despite the recorded approval, and final work-order/initiative metadata reconciliation remains a closeout responsibility. These records must be corrected or explicitly reconciled without rewriting the historical log.
6. **Separate closeout and portability boundaries.** The required closeout review, human/manual closeout gate, and merge decision remain outstanding. The Windows Unicode-console workaround remains a separately documented portability follow-up outside this initiative.

## Documentation and traceability

- README documentation covers the consistent-focal-length capture prerequisite, the absence of automatic calibration, official-binary CPU BA behavior, and the unsupported custom COLMAP/Ceres/Caspar boundary.
- No requirements or blueprint change is required by the final implementation evidence; the behavior remains within the existing reconstruction and external-binary failure contracts.
- The deferred calibration boundary is preserved consistently across the initiative, work orders, README, requirements, and blueprints.
- The execution log contains the human design approval, branch record, implementation validation, external evidence, and latest independent review outcomes.

## Decision

- **Outcome:** `pass-with-warnings`
- **Implementation disposition:** The initiative's approved implementation scope is technically ready to advance. No blocker requires returning either active work order to implementation.
- **Closeout readiness:** `ready-for-closeout-review`, but **not closed**. The separate closeout reviewer must reconcile metadata, confirm the complete branch/diff and mandatory manual gates, record any required human closeout approval, and decide whether the initiative may transition to `closed`.
- **Human approval required for this review:** `no`; the existing design approval is recognized, while closeout approval remains a separate gate.
- **Required next state:** Keep the initiative at the implementation-review boundary until this artifact and the execution-log entry are recorded, then advance to the separate closeout review without self-closing.
