# Review: Design — Harden COLMAP Runtime Compatibility

## Review metadata

- **Review type:** `design`
- **Status:** `pass-with-warnings`
- **Reviewer:** `independent Forge design reviewer`
- **Date:** `2026-08-19`
- **Reviewed artifact:** [`initiative-v1.md`](../initiative-v1.md:1)
- **Compared commit/diff:** Revised initiative and work-order contracts; production correctness was explicitly excluded

## Scope of review

This independent design review read the governing Forge process, the current initiative plan, revised active [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:1) and [`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md:1), deferred [`WO-002-optional-focal-calibration.md`](../work-orders/WO-002-optional-focal-calibration.md:1), the prior initiative design review, the requirements and reconstruction blueprints, the predecessor initiative records, [`README.md`](../../../README.md:1), and the current [`execution-log.md`](../execution-log.md:1).

The review assesses design coherence only. It does not review or modify production implementation, tests, runtime correctness, or implementation-review artifacts. Existing unapproved production changes are recorded below only as a Forge process concern; their technical correctness is not assessed.

## Findings

### Blockers

- None identified in the revised design contracts. The prior technical blockers are addressed at the planning level: WO-001 now defines a structured normalized result, capability-state vocabulary, bounded raw evidence, diagnostics, and explicit continuation/failure rules; WO-003 now selects a deliberately high-level official-binary CPU bundle-adjustment fallback contract rather than leaving the behavior for later human selection.

### Warnings

- **Human design approval remains outstanding.** The revised design is reviewable and coherent, but the Forge process requires human approval after a passing design review and before implementation is authorized. This reviewer does not self-approve that gate. Required action: obtain explicit human design approval and record it in [`execution-log.md`](../execution-log.md:1), then transition the initiative to `approved` before further implementation work.

- **Dedicated-branch creation evidence is not independently retained.** The log records a later branch switch to [`initiative/2026-08-19-colmap-runtime-robustness`](../initiative-v1.md:11), but the plan still states that branch creation evidence remains a pre-implementation gate. Required action: retain branch-creation evidence before implementation is authorized, or obtain and record an explicit policy-based waiver from the authorized human approver. The branch policy itself is not waived by this review.

- **Implementation occurred before the design gate was cleared.** The current log records production and test changes while the initiative was still in `planning-revision` and before human design approval or waiver was recorded. This is a process-gate violation, not a finding about whether those changes are correct. Required action: do not treat the existing changes as approved implementation evidence; have the orchestrator record the human disposition of the process breach and enforce the design-approval gate before additional implementation or completion transitions.

- **WO-003 naming is broader than its selected contract.** The work-order title says `GLOMAP GPU Solver Compatibility`, while the approved design direction is truthful reporting of the official binary's CPU fallback and an unsupported custom-build boundary. The body, initiative plan, README, and acceptance criteria consistently express the narrower contract, so this is not a design blocker. Required action: reconcile the title during ordinary planning maintenance so implementers and reviewers do not infer a supported GPU-enablement objective.

- **Validation evidence remains an execution responsibility, not design approval.** The work orders identify automated checks, target-binary help inspection, external runtime evidence, raw command streams, exit status, solver messages, project status, camera count, and sparse-model output. Those are appropriate gates, but the recorded evidence must still be retained and independently reviewed during implementation/closeout. This review does not certify the existing implementation or external-evidence artifacts.

### Notes

- **WO-001 is now coherent and testable as a design contract.** Its fields distinguish binary availability, parsed version, version status, per-command capability state, bounded raw evidence, and actionable diagnostics. Its rules correctly make a rejected metadata command non-fatal when usable help and the selected mapper command remain available, while a launch failure, timeout, or unusable help response is fatal and must preserve failed-stage state. The requirement that mapper capability be established by command evidence rather than version text prevents a false compatibility claim.

- **WO-003 is appropriately high-level.** The standard contract accepts completion of GLOMAP with official-binary CPU bundle adjustment when COLMAP reports that fallback, requires truthful reporting without claiming GPU BA, preserves GLOMAP identity, and forbids automatic switching to incremental mapping. The custom COLMAP/Ceres/Caspar path is explicitly user-managed, untested, and unsupported. This scope avoids inventing a structured solver-telemetry subsystem that the stakeholder did not request.

- **Sequencing is coherent.** WO-001 precedes WO-003 and establishes the capability/evidence boundary needed by later runtime work. WO-003 also retains the verified global command contract as an external prerequisite. Deferred WO-002 is not in the active sequence and is not an active dependency. Sequential execution complies with the repository's default parallelism policy.

- **Scope and focal-calibration boundaries are coherent.** The initiative preserves global/incremental dispatch, excludes mapper replacement, broad GPU enablement, calibration controls, and unrelated portability work, and records consistent focal length as an input-capture prerequisite. Deferred WO-002 does not authorize implementation, persistence, or requirement/blueprint changes in this initiative.

- **Requirements and blueprints have plausible paths.** The design maps compatible-binary execution and actionable failed-stage diagnostics to the reconstruction requirements and the [`ColmapRunner`](../../../blueprints/components/reconstruction-pipeline.md:18) / [`run_sfm()`](../../../blueprints/features/reconstruction.md:5) boundaries. No durable requirement or blueprint update is required by this design review; any update remains subject to the work-order authorization and implementation evidence rules.

- **Risk and recovery coverage is adequate for the selected scope.** The plans address command-syntax variation, unavailable binaries, unsupported metadata, mapper failure, CPU fallback truthfulness, unsupported custom builds, no automatic mapper substitution, and the terminal/manual validation constraints. The external binary remains the compatibility authority, and failure must remain visible as failed stage state rather than being hidden by a fallback claim.

## Traceability and validation

- **Requirements checked:** [`REQ-RECONSTRUCTION.md`](../../../requirements/features/REQ-RECONSTRUCTION.md:18), including AC-RECON-001.1, AC-RECON-001.2, NFR-RECON-001, and the automatic-calibration boundary at line 55.
- **Blueprints checked:** [`reconstruction.md`](../../../blueprints/features/reconstruction.md:5), [`reconstruction-pipeline.md`](../../../blueprints/components/reconstruction-pipeline.md:16), [`gsforge-cli.md`](../../../blueprints/containers/gsforge-cli.md:1), and [`project-state.md`](../../../blueprints/components/project-state.md:1).
- **Process checked:** [`00-overview.md`](../../../process/00-overview.md:1), [`02-initiative.md`](../../../process/02-initiative.md:1), [`03-design-review.md`](../../../process/03-design-review.md:1), and [`04-work-order.md`](../../../process/04-work-order.md:1).
- **Initiative artifacts checked:** [`initiative-v1.md`](../initiative-v1.md:1), [`WO-001-colmap-capability-probe.md`](../work-orders/WO-001-colmap-capability-probe.md:1), [`WO-002-optional-focal-calibration.md`](../work-orders/WO-002-optional-focal-calibration.md:1), [`WO-003-glomap-cpu-fallback.md`](../work-orders/WO-003-glomap-cpu-fallback.md:1), [`execution-log.md`](../execution-log.md:1), and the prior blocked-review findings preserved in the log.
- **Predecessor records checked:** predecessor [`initiative-v1.md`](../../2026-08-19-glomap-mapper-selection/initiative-v1.md:1), [`initiative-closeout-review.md`](../../2026-08-19-glomap-mapper-selection/reviews/initiative-closeout-review.md:1), and [`execution-log.md`](../../2026-08-19-glomap-mapper-selection/execution-log.md:1), including the recorded global/incremental command evidence and bounded 185-frame result.
- **Repository documentation checked:** [`README.md`](../../../README.md:93), including consistent-focal-length capture guidance and the official-binary CPU BA/custom-build boundary.
- **Commands/evidence checked:** Documentary evidence and prior review records were inspected only. No production validation was rerun, and no implementation correctness conclusion is made.
- **Manual/process gates checked:** Sequential ordering, dedicated-branch policy, human design approval, external binary validation, per-work-order review cadence, reviewer artifact location, and the explicit no-self-approval rule.

## Decision

- **Outcome:** `pass-with-warnings`
- **Technical design disposition:** The revised initiative design is coherent enough to proceed to the human design-approval gate. WO-001's normalized probe/evidence contract, WO-003's high-level official CPU fallback contract, sequencing, bounded scope, acceptance/validation plan, custom-build boundary, and focal-calibration deferral are mutually consistent.
- **Required next state:** Keep the initiative out of `approved` until the process warnings are resolved or explicitly accepted under Forge policy. Retain dedicated-branch creation evidence, obtain human design approval, record the decision in [`execution-log.md`](../execution-log.md:1), reconcile the WO-003 title, and enforce the gate before treating any implementation as authorized. Then proceed through the separate work-order implementation-review and closeout gates.
- **Human approval required:** `yes`
- **Approval/waiver:** Exact required approval is explicit human approval of the revised initiative design, including the official-binary GLOMAP CPU BA fallback as the supported standard behavior, the unsupported custom-build boundary, and the focal-calibration deferral. A separate policy-based waiver is required only if the dedicated-branch evidence or any other mandatory Forge gate is intentionally bypassed; this reviewer grants neither approval nor waiver.
