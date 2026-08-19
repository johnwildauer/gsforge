# Review: Closeout — Harden COLMAP Runtime Compatibility

## Review metadata

- **Review type:** `closeout`
- **Status:** `complete-with-follow-up`
- **Reviewer:** independent Forge closeout reviewer
- **Date:** 2026-08-19
- **Reviewed artifact:** [`initiative-v1.md`](../initiative-v1.md:1)
- **Compared commit/diff:** No complete branch diff or working-tree status was available to this review; the current branch ref was inspected from [`.git/HEAD`](../../../.git/HEAD:1).

## Scope of review

This separate closeout review read the closeout process, Forge policy, authoritative initiative plan, all three work orders, design and implementation reviews, latest independent WO-001/WO-003 reviews, execution log, predecessor closeout, README, reconstruction requirements, reconstruction blueprints, current branch reference, and recorded runtime/validation evidence. No production source or test files were modified, and no automated checks or external commands were rerun.

## Findings

### Blockers

1. **Required human closeout/merge approval is not evidenced.** Repository policy requires human approval after all work orders complete and before closeout/merge in [`00-overview.md`](../../../process/00-overview.md:80). The log contains human design approval at [`execution-log.md`](../execution-log.md:141), but no separate final human closeout approval or explicit waiver is recorded. The closeout process therefore does not permit a `closed` transition.

2. **Complete branch diff and working-tree reconciliation cannot be verified.** [`.git/HEAD`](../../../.git/HEAD:1) proves that the checked-out ref is `initiative/2026-08-19-colmap-runtime-robustness`, and [`.git/config`](../../../.git/config:17) records branch configuration. However, no `git status`, complete diff, commit identifier, or equivalent clean-tree evidence is retained in the initiative records or available to this review. Consequently, accidental files, untracked changes, unresolved conflicts, and the exact merge scope remain unknown rather than passing by assumption.

### Warnings and follow-ups

- **Validation is documentary in this environment.** The log records 86 full-suite tests, scoped Ruff check/format results, and the 185-frame COLMAP 4.1.1 run. The latest independent reviews explicitly assessed those claims without rerunning them because command execution was unavailable. This is evidence limitation, not a newly observed validation failure.
- **WO-001 evidence depth.** The implementation persists bounded raw probe evidence, but its tests do not assert every serialized raw-evidence field individually. The latest WO-001 review accepted this as test strengthening only.
- **WO-003 evidence granularity.** [`WO-003-external-evidence-4.1.1.md`](WO-003-external-evidence-4.1.1.md:1) is a concise summary linked to [`garden-glomap-sfm.txt`](../../../garden-glomap-sfm.txt:1210), not a complete raw archive for every probe and run command. The latest WO-003 review accepted it for the approved high-level contract.
- **Bounded solver reporting.** CPU-fallback detection is intentionally high-level, transient, and based on known warning fragments; no general solver telemetry or persisted solver-mode field is required by the approved scope.
- **Deferred work remains discoverable.** WO-002 focal calibration is correctly retained as deferred and excluded from acceptance. Consistent focal length is documented in [`README.md`](../../../README.md:100). The Windows Unicode-console workaround remains an explicitly separate portability follow-up.
- **Title/scope wording.** WO-003's title is broader than its approved CPU-fallback reporting contract. This is a documentation follow-up, not a closeout blocker.
- **No requirements or blueprint change is required.** The implementation remains aligned with [`REQ-RECONSTRUCTION.md`](../../../requirements/features/REQ-RECONSTRUCTION.md:18), [`reconstruction.md`](../../../blueprints/features/reconstruction.md:5), [`reconstruction-pipeline.md`](../../../blueprints/components/reconstruction-pipeline.md:16), and [`gsforge-cli.md`](../../../blueprints/containers/gsforge-cli.md:1).

## Reconciliation assessment

- **Work orders:** WO-001 and WO-003 are both marked `complete-with-warnings`; their completion checklists are checked and each has a latest independent implementation review with `pass-with-warnings`. Deferred WO-002 is not counted toward acceptance and remains incomplete by design.
- **Scope:** The approved capability probe, failed-state behavior, truthful official-binary GLOMAP CPU fallback reporting, dispatch preservation, focal-length documentation, and unsupported custom-build boundary are represented in source, tests, README, and the initiative records. No evidence shows calibration, mapper replacement, or unrelated portability work was added.
- **Design and implementation gates:** The design review is `pass-with-warnings`; human design approval is recorded; the initiative implementation review is `pass-with-warnings` and says `ready-for-closeout-review`. Earlier blocked reviews are preserved as history and are superseded for current technical disposition by the later passing re-reviews; they are not deleted or rewritten.
- **External/manual validation:** The recorded real-data run completed GLOMAP with 185 cameras, selected `sfm/sparse/0`, persisted completed state, and reported CPU Ceres/cuDSS fallback while preserving GLOMAP identity. The binary/help evidence and the UTF-8 console workaround are recorded. The closeout reviewer does not treat the separate Unicode-console issue as an initiative defect.
- **Branch/approval evidence:** The current branch ref is observed, and branch activation is recorded in [`execution-log.md`](../execution-log.md:135). Branch creation evidence, clean-tree status, complete diff, and final human closeout approval are not retained. These missing records are classified as manual/process gates, not inferred as passed.

## Decision

- **Outcome:** `blocked`
- **Disposition:** The scoped implementation is technically complete with warnings and suitable for closeout follow-up, but the initiative cannot be closed from the available evidence. The mandatory final human closeout/merge approval is missing, and branch/diff cleanliness cannot be verified.
- **Required next state:** Keep the initiative at the closeout gate with status `blocked`. Obtain and record final human closeout/merge approval (or an authorized policy waiver), retain complete branch status/diff evidence including accidental-file and conflict checks, then perform the closeout decision again. Do not claim that the missing manual gates passed.
- **Human approval required:** `yes`
- **Approval/evidence:** Design approval is recorded at [`execution-log.md`](../execution-log.md:141). No final closeout approval or waiver is recorded.
