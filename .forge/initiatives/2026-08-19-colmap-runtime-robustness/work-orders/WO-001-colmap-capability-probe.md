# Work Order: WO-001 — Resilient COLMAP Capability and Version Probe

## Metadata

- **Status:** `complete-with-warnings`
- **Assignee:** `gsforge implementer`
- **Phase/order:** `1`
- **Initiative:** [`../initiative-v1.md`](../initiative-v1.md)
- **Owner/orchestrator:** `gsforge maintainers`
- **Review artifact location:** [`../reviews/`](../reviews/)

## Summary

Replace the unconditional `colmap --version` probe with a capability-aware probe that treats the target binary's supported help command as the authority. The probe must provide useful version/capability diagnostics when available and remain non-fatal when metadata discovery is unsupported but the subsequent reconstruction commands are usable.

## Evidence gathered before implementation

- The current [`check_colmap_version()`](../../../src/gsforge/sfm.py:154) invokes only `colmap --version`, assumes the first combined output line is a version, catches every exception, and returns the string `unknown` on failure.
- [`run_sfm()`](../../../src/gsforge/sfm.py:688) calls the probe before its failure-state `try` block, so the revised probe must not raise for an unsupported metadata command and must distinguish a metadata warning from an unavailable binary.
- The predecessor real-data log records a target Windows binary rejecting `--version` and explicitly directing the user to `colmap help`; the same binary completed both GLOMAP and incremental COLMAP runs on 185 frames.
- Existing tests cover mapper dispatch and sparse-model selection but contain no probe contract coverage. The first tests must therefore establish the normalized result and `run_sfm()` continuation/failure behavior rather than only test string parsing.
- The target binary is identified as COLMAP `4.0.2`, commit `d927f7e` dated `2026-03-18`, released with CUDA, installed at `bin/colmap-x64-windows-cuda/bin/colmap.exe`.
- The supplied top-level help lists the relevant commands, including `help`, `feature_extractor`, `exhaustive_matcher`, `sequential_matcher`, `mapper`, and `global_mapper`; `global_mapper -h` confirms the `GlobalMapper.*` namespace and GPU-related options including `gp_use_gpu=1` and `ba_ceres_use_gpu=1`.

## Normalized contract to implement

The probe returns a structured result with these fields:

| Field              | Meaning                                                                                                                                                                                            |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `binary_available` | The executable could be launched successfully.                                                                                                                                                     |
| `version`          | Parsed version text when a supported command provides it; otherwise `None`.                                                                                                                        |
| `version_status`   | `known`, `unsupported`, `unknown`, or `unavailable`.                                                                                                                                               |
| `commands`         | Per-command capability states for `help`, `feature_extractor`, `exhaustive_matcher`, `sequential_matcher`, `mapper`, and `global_mapper`: `supported`, `unsupported`, `unknown`, or `unavailable`. |
| `raw_evidence`     | Bounded stdout/stderr and exit status for each attempted probe, suitable for logging without becoming a reconstruction prerequisite.                                                               |
| `diagnostic`       | Actionable human-readable summary of what was found and what is missing.                                                                                                                           |

Integration rules:

1. A successful `help` probe with an unsupported `--version` is `binary_available=true`, `version_status=unsupported`, and is non-fatal; `run_sfm()` proceeds to normal command execution.
2. A launch failure, timeout, or completely unusable help response is `binary_available=false`, `version_status=unavailable`, and is fatal; the SfM stage must be recorded as failed with the diagnostic before returning/raising.
3. An executable whose top-level help is usable but whose selected mapper capability is unknown must not be rejected solely because version text is absent. The selected mapper command remains the authority, and its normal command failure must preserve failed-stage state.
4. The probe must not claim `mapper` or `global_mapper` support merely from a version string; command help or equivalent command evidence is required.

## In scope

- Inspect and record the target binary's behavior for `--version`, `help`, `-h`, and relevant subcommand help.
- Define a normalized probe result sufficient for later work orders to distinguish binary unavailable, capability known, and version unknown.
- Update the SfM probe path and focused tests without changing mapper dispatch.
- Preserve raw command/exit/stdout/stderr evidence in the execution log or linked review artifact.

## Out of scope

- Changing feature, matching, mapper, calibration, solver, or sparse-model commands.
- Requiring a version string before running classic COLMAP or GLOMAP.
- Supporting every historical COLMAP output format.
- Any Windows console redesign.

## Requirements

- [`AC-RECON-001.1`](../../../requirements/features/REQ-RECONSTRUCTION.md:24): **When prepared frames exist and a compatible COLMAP binary is available, the system shall run feature extraction, matching, and mapping.**
- [`NFR-RECON-001`](../../../requirements/features/REQ-RECONSTRUCTION.md:48): **External binary failures shall preserve a failed project status and expose actionable diagnostics.**

## Blueprints

- [`reconstruction-pipeline.md`](../../../blueprints/components/reconstruction-pipeline.md:16) — `ColmapRunner` owns binary discovery and external commands.
- [`reconstruction.md`](../../../blueprints/features/reconstruction.md:5) — `run_sfm()` establishes the external-binary pipeline.

## Implementation plan

1. Capture target help/version behavior, including the observed unsupported `--version` response, and implement the normalized probe state contract above.
2. Integrate the probe at the existing COLMAP boundary while keeping metadata discovery separate from mapper command execution.
3. Add mocked tests for supported version, unsupported version with supported help, empty/failing help, timeout, and actionable warning/error behavior.
4. Run focused tests, configured lint/format checks, and a real target-binary smoke probe using executable-compatible commands.

## Files and systems

- **Create:** likely no production file; tests may be added only if existing coverage cannot host the cases.
- **Update:** [`src/gsforge/sfm.py`](../../../src/gsforge/sfm.py:154), [`tests/test_sfm.py`](../../../tests/test_sfm.py:1), and this initiative's execution log.
- **Avoid changing:** mapper construction, CLI options, requirements, blueprints, media, and project artifacts.

## Validation and E2E acceptance tests

- **Automated checks:** `pixi run test tests/test_sfm.py`; `pixi run lint`; `pixi run format`; use executable-compatible equivalents if the terminal cannot run the declared shell command.
- **External checks:** target binary probes for `--version`, `help`, `-h`, and relevant command help; capture exit status and both output streams.
- **Acceptance test:** Given a binary that rejects `--version` and advertises `help`, when the probe runs, then it returns the structured non-fatal `unsupported` version result, records bounded raw evidence, logs the actionable diagnostic, and allows the normal SfM path to continue to command execution.
- **Recovery:** if all probes fail, retain current binary-not-found/failure semantics and record the exact evidence; do not infer a version or command capability.

## Documentation updates

- No requirements or blueprint change is authorized unless review finds that the normalized probe is a durable external contract. Record that decision explicitly.
- Append probe evidence and validation results to [`../execution-log.md`](../execution-log.md).

## Completion checklist

- [x] Implementation is complete within scope.
- [x] Tests/checks and external probe evidence have passed.
- [x] Authorized durable documentation is updated or explicitly unchanged.
- [x] Independent review artifact is under [`../reviews/`](../reviews/).
- [x] Execution log is appended.
