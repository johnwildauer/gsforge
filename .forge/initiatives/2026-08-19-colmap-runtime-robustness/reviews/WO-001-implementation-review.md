# Review: Implementation — WO-001 Resilient COLMAP Capability and Version Probe

## Review metadata

- **Review type:** `implementation`
- **Status:** `blocked`
- **Reviewer:** `independent Forge implementation reviewer`
- **Date:** `2026-08-19`
- **Reviewed artifact:** [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md)
- **Compared commit/diff:** Current workspace contents of [`sfm.py`](../../../src/gsforge/sfm.py), [`test_sfm.py`](../../../tests/test_sfm.py), [`README.md`](../../../README.md), and the initiative log; no commit identifier or VCS diff was supplied

## Scope of review

Reviewed the active WO-001 contract and implementation, the probe integration in [`run_sfm()`](../../../src/gsforge/sfm.py:750), focused tests, execution-log validation claims, requirements, reconstruction blueprints, Forge implementation-review policy, and the deferred-calibration boundary. No production code or tests were changed, and no commands or external binaries were run, per the review instruction.

## Findings

### Blockers

- **The implementation contradicts the required unsupported-version state.** The work order requires a successful help probe combined with a rejected `--version` probe to return `version_status="unsupported"` ([`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:40)). [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:169) instead parses a version-looking string from help and returns `known`; the focused test at [`test_help_is_authoritative_when_version_is_unsupported()`](../../../tests/test_sfm.py:135) codifies that contradictory behavior. Required action: reconcile the contract and implementation, then test the exact state transition.

- **Empty or semantically unusable help is accepted as usable.** The probe checks only the help process exit code at [`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:209); an empty successful response is treated as `binary_available=True`, and empty successful subcommand responses are marked `supported`. This violates the required “completely unusable help response” fatal condition and the rule that mapper capability requires command-help evidence. Required action: define and validate usable-help content, including mapper and global-mapper evidence.

- **Probe evidence and successful diagnostics are discarded at the runtime boundary.** [`run_sfm()`](../../../src/gsforge/sfm.py:807) receives the structured result but neither logs its `diagnostic` on a successful probe nor persists/links `raw_evidence`; only the fatal diagnostic reaches the logger. This fails WO-001's raw-evidence retention requirement ([`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:35), [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:80)) and prevents later work orders or operators from seeing the capability basis.

- **Required integration and failure-state tests are absent.** [`test_sfm.py`](../../../tests/test_sfm.py:134) covers only one successful probe shape and one timeout. It does not exercise `run_sfm()` continuation after unsupported metadata, fatal probe persistence, empty help, unavailable mapper/global-mapper capability, bounded evidence, or a selected mapper failure preserving failed state, despite the explicit test plan ([`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:71)). The execution log's “26 focused tests” claim is not independently traceable to those cases.

- **The initiative implementation gate was bypassed without recorded authorization.** The initiative requires design review and human design approval before production implementation ([`initiative-v1.md`](../initiative-v1.md:74)); the latest design review is `blocked` and the initiative remains `planning-revision` ([`initiative-v1.md`](../initiative-v1.md:9), [`initiative-design-review.md`](initiative-design-review.md:48)). The execution log records implementation anyway ([`execution-log.md`](../execution-log.md:65)) but records no explicit waiver or approval. This is a process blocker independent of code correctness.

### Warnings

- **Version discovery does not probe the documented `version` command.** The implementation probes only `--version` and `help` ([`probe_colmap_capabilities()`](../../../src/gsforge/sfm.py:209)); the validation log says the target exposes a `version` command ([`execution-log.md`](../execution-log.md:71)). Consequently, supported version evidence may be missed or inferred from unrelated help text. Recommendation: either add the command to the contract/probe sequence or explicitly document why it is not authoritative.

- **Capability status is derived primarily from exit code.** A zero exit code with non-help output is `supported`, while a nonzero response is `unsupported` even when the process launched but the command is unknown. This makes `unknown`, `unsupported`, and `unavailable` less precise than the normalized contract and weakens diagnostics.

- **Work-order state is not reconciled.** WO-001 remains `draft` and its completion checklist is unchecked ([`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:5)); the execution log claims implementation validation. The orchestrator should set a truthful next state after remediation and review.

### Notes

- Mapper dispatch remains unchanged: [`run_mapper()`](../../../src/gsforge/sfm.py:433) retains `global_mapper` for GLOMAP and `mapper` for incremental COLMAP, with separate option namespaces. No scope violation was found in that area.
- The consistent-focal-length README prerequisite is present and the deferred calibration work order remains outside the active sequence. This review does not treat that boundary as a defect.

## Traceability and validation

- **Requirements checked:** [`REQ-RECONSTRUCTION.md`](../../../requirements/features/REQ-RECONSTRUCTION.md:18), especially AC-RECON-001.1, AC-RECON-001.2, and NFR-RECON-001.
- **Blueprints checked:** [`reconstruction.md`](../../../blueprints/features/reconstruction.md:5), [`reconstruction-pipeline.md`](../../../blueprints/components/reconstruction-pipeline.md:16), and [`project-state.md`](../../../blueprints/components/project-state.md:1).
- **Process checked:** [`00-overview.md`](../../../process/00-overview.md:53) and [`05-implementation-review.md`](../../../process/05-implementation-review.md:5).
- **Commands/evidence checked:** Logged focused-test, full-test, Ruff, and external-probe claims at [`execution-log.md`](../execution-log.md:65); no commands were rerun and no raw probe transcript was present.
- **Manual gates checked:** Dedicated-branch evidence, design approval, target-binary probe evidence, and per-work-order review location. Design approval/branch creation/raw probe transcript are not evidenced sufficiently for approval.

## Decision

- **Outcome:** `blocked`
- **Required next state:** Return WO-001 and the initiative to `in-progress`/planning revision; reconcile the version-status contract, reject unusable help, retain/log evidence, add integration/failure-state tests, and obtain the required design authorization before another implementation review.
- **Human approval required:** `yes`
- **Approval/evidence:** No valid design approval or explicit gate waiver is recorded. This reviewer does not self-approve while blockers remain.
