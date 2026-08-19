# Work Order: WO-002 — Optional Focal-Length Calibration Policy

## Metadata

- **Status:** `deferred`
- **Assignee:** `gsforge implementer`
- **Phase/order:** `2`
- **Initiative:** [`../initiative-v1.md`](../initiative-v1.md)
- **Owner/orchestrator:** `gsforge maintainers`
- **Review artifact location:** [`../reviews/`](../reviews/)

## Summary

Deferred candidate for a later initiative: turn focal-length refinement from an implicit mapper argument into an explicit, optional policy. It is not part of the current initiative and must not be implemented under the current gate.

## Deferred scope

- Inspect current camera initialization, mapper argument construction, CLI surface, project metadata, and tests.
- Design and implement one explicit opt-in/opt-out calibration contract for both supported mapper methods, subject to design approval.
- Add focused command-construction and state/reporting tests for default and explicit policy paths.
- Run a bounded same-project A/B validation with calibration disabled/enabled where the target binary supports both forms.

## Out of scope

- Full camera-model selection, principal-point or distortion calibration policy changes, external calibration import, or reconstruction-quality tuning.
- Automatic calibration beyond the explicitly approved focal-length option.
- Changing global/incremental mapper selection or solver fallback.

## Future requirements to revisit

- [`AC-RECON-001.2`](../../../requirements/features/REQ-RECONSTRUCTION.md:25): **The command shall expose global and incremental method choices and persist the selected method and outcome; the documented global mapper behavior is a current implementation alignment gap requiring verification or correction in a future initiative.**
- [`REQ-RECON-001` user story](../../../requirements/features/REQ-RECONSTRUCTION.md:20): **As an artist, I want camera poses and sparse points from prepared frames, so that I can train or hand off a scene reconstruction.**
- Current constraint requiring explicit reconciliation: [`REQ-RECONSTRUCTION.md` out-of-scope item](../../../requirements/features/REQ-RECONSTRUCTION.md:55): **Automatic camera calibration beyond the selected COLMAP defaults.**

## Blueprints

- [`reconstruction.md`](../../../blueprints/features/reconstruction.md:21) — `ReconstructionCommand` exposes method selection and completion summary.
- [`reconstruction-pipeline.md`](../../../blueprints/components/reconstruction-pipeline.md:16) — `ColmapRunner` owns mapper options.
- [`project-state.md`](../../../blueprints/components/project-state.md:1) — persisted stage/method outcome is the state boundary.

## Implementation plan

1. Present default and user-control options at design review; do not assume that enabling refinement by default is acceptable.
2. Define the policy's CLI/API, persistence, command mapping, validation, and backward-compatibility behavior.
3. Implement the smallest change across the approved command/state boundaries and add regression tests.
4. Run automated checks and a controlled A/B external run, recording camera counts, exit statuses, selected model, and emitted focal options.

## Files and systems

- **Create:** no production file expected; tests may be expanded in existing test modules.
- **Update:** likely [`src/gsforge/cli.py`](../../../src/gsforge/cli.py:322), [`src/gsforge/sfm.py`](../../../src/gsforge/sfm.py:327), project state code, and focused tests; exact ownership is finalized at design review.
- **Avoid changing:** non-focal camera options, mapper selection, solver behavior, media, and unrelated requirements.

## Validation and E2E acceptance tests

- **Automated checks:** focused CLI/mapper/state tests; `pixi run test`; `pixi run lint`; `pixi run format`, or executable-compatible equivalents.
- **Manual/external checks:** target `mapper -h` and `global_mapper -h`; same prepared project with policy disabled and enabled; inspect command, exit result, camera count, sparse model, and persisted method/policy.
- **Acceptance test:** Given a prepared project and a supported COLMAP binary, when the user selects the approved focal policy, then only the corresponding focal refinement option changes, the selection is persisted/reported, and the default behavior matches the human-approved contract.
- **Recovery:** unsupported focal option must fail clearly or use the approved safe default; it must never silently claim calibration occurred.

## Documentation updates

- This work order explicitly authorizes updating the relevant requirement and blueprint only if the approved behavior changes the current calibration boundary. The update must preserve requirement/blueprint/initiative separation and cite implementation evidence.
- Append the policy decision and A/B evidence to [`../execution-log.md`](../execution-log.md).

## Completion checklist

- [ ] Human-approved default and control semantics are recorded.
- [ ] Implementation and focused tests are complete within scope.
- [ ] Automated and manual evidence has passed.
- [ ] Authorized durable documentation is reconciled or explicitly unchanged.
- [ ] Independent review artifact is under [`../reviews/`](../reviews/).
- [ ] Execution log is appended.
