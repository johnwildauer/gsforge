# Work Order: WO-001 — Verify and Correct GLOMAP Mapper Selection

## Metadata

- **Status:** `complete`
- **Assignee:** `gsforge implementer`
- **Phase/order:** `1`
- **Initiative:** [`../initiative-v1.md`](../initiative-v1.md)
- **Owner/orchestrator:** `gsforge maintainers`

## Summary

Determine the actual global-SfM command and accepted mapper options exposed by the target COLMAP binary, then correct the existing method dispatch so `glomap` selects that command while `colmap` remains incremental. Add regression tests around the generated subprocess commands so the reported method cannot diverge from the invoked mapper again.

## In scope

- Inspect `colmap --version`, `colmap -h`, and the relevant `mapper -h`/`global_mapper -h` output for the binary used by the initiative.
- Update only the mapper command construction in [`src/gsforge/sfm.py`](../../../src/gsforge/sfm.py:327) as needed to match that evidence.
- Add focused tests in [`tests/test_sfm.py`](../../../tests/test_sfm.py:1) for global and incremental command construction.
- Record command-contract evidence and implementation validation in the initiative execution log.

## Out of scope

- Feature extraction/matching, calibration, camera priors, model selection, import/export, project state, CLI surface, training, README, requirements, and blueprints.
- Real-data quality improvements, automatic retry/fallback, or support for every historical COLMAP release.

## Requirements

- [`REQ-RECON-001`](../../../requirements/features/REQ-RECONSTRUCTION.md:18): **User Story:** As an artist, I want camera poses and sparse points from prepared frames, so that I can train or hand off a scene reconstruction.
- [`AC-RECON-001.1`](../../../requirements/features/REQ-RECONSTRUCTION.md:24): When prepared frames exist and a compatible COLMAP binary is available, the system shall run feature extraction, matching, and mapping.
- [`AC-RECON-001.2`](../../../requirements/features/REQ-RECONSTRUCTION.md:25): The command shall expose global and incremental method choices and persist the selected method and outcome; the documented global mapper behavior is a current implementation alignment gap requiring verification or correction in a future initiative.

## Blueprints

- [`reconstruction.md`](../../../blueprints/features/reconstruction.md:5) — the SfM flow delegates mapping through `run_sfm()`.
- [`reconstruction-pipeline.md`](../../../blueprints/components/reconstruction-pipeline.md:16) — `ColmapRunner` owns external extraction, matching, and mapping commands.
- [`gsforge-cli.md`](../../../blueprints/containers/gsforge-cli.md:24) — the CLI is the product boundary and COLMAP is an external system.

## Implementation plan

1. Capture the target binary version and help output for the available global and incremental mapping commands; decide the smallest supported command form and record the evidence before code changes.
2. Refactor the existing `run_mapper()` argument assembly only enough to dispatch `colmap` to the verified incremental command and `glomap` to the verified global command, retaining only options accepted by each command.
3. Extend `tests/test_sfm.py` with mocked subprocess assertions for both methods, including the subcommand, paths, global selection, and absence of accidental global selection on the incremental path.
4. Run focused tests, lint/format checks configured by the repository, and the post-planning A/B user gate from the initiative plan.

## Files and systems

- **Create:** none.
- **Update:** [`src/gsforge/sfm.py`](../../../src/gsforge/sfm.py:327), [`tests/test_sfm.py`](../../../tests/test_sfm.py:1).
- **Execution evidence:** [`../execution-log.md`](../execution-log.md).
- **Avoid changing:** all other source files, CLI options, README, requirements, blueprints, and generated project/media artifacts outside an isolated A/B test project.

## Validation and E2E acceptance tests

- **Automated checks:** `pixi run test tests/test_sfm.py`; `pixi run lint`; `pixi run format` (or the repository-equivalent commands if task availability is verified differently). Also inspect the captured command lists directly in the focused tests.
- **Binary contract checks:** `colmap --version`; `colmap -h`; `colmap mapper -h`; `colmap global_mapper -h` when available. Preserve the exact relevant output in the execution log or linked review evidence.
- **Manual/external checks:** Human A/B run on the same isolated prepared-frame project: A uses `gsforge sfm --method colmap`; B uses `gsforge sfm --method glomap`. Record exit status, emitted mapper command, COLMAP version, registered cameras, selected sparse model, and any unknown-option/subcommand errors. External COLMAP availability and a suitable dataset are manual gates.
- **Acceptance test:** Given a target COLMAP binary whose help output identifies the supported global mapper command, when `run_mapper(..., "glomap")` is executed, then the captured subprocess command invokes that global command with only accepted options; when `run_mapper(..., "colmap")` is executed, then it invokes the incremental mapper without global-only selection.

## Documentation updates

- No requirements or blueprint updates are authorized for this work order. The existing alignment-gap wording remains accurate until implementation and A/B evidence support a later baseline reconciliation.
- Append command-contract findings, validation results, A/B outcome, and any blocked incompatibility to [`../execution-log.md`](../execution-log.md).

## Completion checklist

- [x] Implementation is complete within scope.
- [x] Tests/checks and required gates have evidence.
- [x] Authorized durable documentation is updated.
- [x] Execution log is appended.
