# Initiative: Restore GLOMAP Mapper Selection

## Initiative metadata

- **Status:** `closed-with-follow-up`
- **Owner/orchestrator:** `gsforge maintainers`
- **Created:** `2026-08-19`
- **Updated:** `2026-08-19`
- **Current gate:** `closed`
- **Authoritative version:** `v1`
- **Development branch:** `initiative/2026-08-19-glomap-mapper-selection`

## Outcome

Make the default `gsforge sfm --method glomap` path invoke the installed COLMAP global-SfM entry point, using only mapper options demonstrated to be accepted by the target COLMAP binary. Preserve the existing incremental `--method colmap` path and provide focused regression coverage for the command construction.

## Scope

### In scope

- Correct the GLOMAP/global mapper selection in [`run_mapper()`](../../../src/gsforge/sfm.py:327), based on the installed COLMAP command contract.
- Preserve the shared database, image, output, and currently intended bundle-adjustment options where the selected command accepts them.
- Add focused tests that distinguish the global and incremental subprocess command forms without invoking a real COLMAP binary.
- Add a post-planning human A/B gate using the repository's prepared-frame workflow and an installed COLMAP binary.

### Out of scope

- Changes to feature extraction, feature matching, sparse-model enumeration/selection, import/export, project persistence, training, or unrelated COLMAP/COLMAP behavior.
- Recalibration, `view_graph_calibrator`, camera-model changes, new CLI options, or automatic fallback between mapper commands.
- Updating requirements, blueprints, README, or external documentation; the existing baseline already records the alignment gap.
- Implementing the cleanup in this planning phase.

## Context and motivation

The repository presents GLOMAP as the default global SfM method, but the executable path currently does not select a global mapper. In [`run_mapper()`](../../../src/gsforge/sfm.py:327), `method == "glomap"` is mapped to the diagnostic value `GLOBAL` at [`sfm.py:350`](../../../src/gsforge/sfm.py:350), while the only proposed `--Mapper.mapper_type GLOBAL` argument is commented out at [`sfm.py:368`](../../../src/gsforge/sfm.py:368). The resulting command always invokes the `mapper` subcommand at [`sfm.py:374`](../../../src/gsforge/sfm.py:374) with no effective global-selection argument. The CLI defaults to `glomap` at [`cli.py:334`](../../../src/gsforge/cli.py:334) and exposes the choice at [`cli.py:360`](../../../src/gsforge/cli.py:360), so the mismatch affects the default user-visible behavior.

The approved reconstruction requirement explicitly identifies this as an implementation alignment gap in [`REQ-RECONSTRUCTION.md`](../../requirements/features/REQ-RECONSTRUCTION.md:24) and records that the mapper currently does not pass the documented global flag at [`REQ-RECONSTRUCTION.md`](../../requirements/features/REQ-RECONSTRUCTION.md:61). The reconstruction blueprint locates the flow in [`run_sfm()`](../../blueprints/features/reconstruction.md:5) and the pipeline blueprint assigns mapping to `ColmapRunner` at [`reconstruction-pipeline.md`](../../blueprints/components/reconstruction-pipeline.md:16).

The current official COLMAP CLI documentation reports that global SfM is selected with the `global_mapper` subcommand, not by passing a `GLOBAL` value to the ordinary `mapper` command; see the live command-line guidance at [COLMAP CLI documentation](https://colmap.github.io/cli.html). Because the local installed binary/version is the execution authority, the implementation work order must verify the available command and option set with `colmap -h` and the relevant command's `-h` output before changing code. This resolves the remembered version/flag discrepancy without guessing from documentation alone.

## Requirements and blueprint references

- **Requirement:** [`REQ-RECON-001`](../../requirements/features/REQ-RECONSTRUCTION.md:18), especially AC-RECON-001.1 and AC-RECON-001.2.
- **Blueprint:** [`reconstruction.md`](../../blueprints/features/reconstruction.md:1), especially the `run_sfm()` flow and requirement alignment.
- **Blueprint:** [`reconstruction-pipeline.md`](../../blueprints/components/reconstruction-pipeline.md:1), especially `ColmapRunner` and its external-binary contract.
- **CLI boundary:** [`gsforge-cli.md`](../../blueprints/containers/gsforge-cli.md:1).

## Proposed design

Use a method-to-command mapping inside the existing mapper runner. The incremental method continues to invoke `colmap mapper`; the global method invokes the global command supported by the target binary. Shared arguments remain centralized, while method-specific arguments are added only after help-output verification establishes that they are valid for that command/version. The implementation must not silently run incremental mapping while reporting `glomap`.

The focused tests will capture `subprocess.run` for `run_mapper()` and assert the complete command boundary relevant to this initiative: executable, subcommand, method-specific global selection, shared paths, and accepted options. They will also assert that the incremental path remains incremental. Tests will not claim that a fake binary produced a valid reconstruction.

### Post-planning A/B user gate

After implementation review, a human operator will run the same prepared-frame project twice against the installed COLMAP binary:

- **A — control:** `gsforge sfm --method colmap`, recording the emitted mapper command, exit result, registered-camera count, and selected sparse model.
- **B — candidate global path:** `gsforge sfm --method glomap`, recording the same evidence and confirming that the binary accepts the selected global command/options.

The gate passes only when B is visibly the global invocation rather than the incremental control, both invocations handle command-line options without an unknown-option/subcommand failure, and the resulting artifacts/status are inspectable. The human approver records the binary version, command help evidence, dataset/project used, and A/B outcome in the execution log. This is a validation gate, not permission to broaden scope into calibration or reconstruction-quality tuning.

## Work-order plan

| Order | Work order                                                                         | Purpose                                                                                                                                               | Dependencies | Review cadence |
| ----- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | -------------- |
| 1     | [`work-orders/WO-001-mapper-selection.md`](work-orders/WO-001-mapper-selection.md) | Verify the installed COLMAP mapper contract, implement the minimal global/incremental command correction, and add focused command-construction tests. | none         | per work order |

## Acceptance criteria

- The `glomap` method does not invoke the ordinary incremental `mapper` command without an effective global selection.
- The selected global command and its method-specific options are supported by the target COLMAP binary, with `colmap -h`/command-help evidence recorded before implementation approval.
- The `colmap` method continues to invoke the incremental mapper with its existing behavior.
- Focused automated tests fail for the current incorrect command form and pass for the corrected global and incremental forms.
- The post-planning A/B user gate records observable command, exit, status, camera-count, and sparse-model evidence for both methods, or explicitly blocks the initiative with the exact incompatibility.
- No source-code implementation, test update, or durable requirement/blueprint change is performed before design review and human design approval.

## Risks and decisions

- **Risk:** COLMAP builds expose different global-mapper command names or option namespaces. **Mitigation:** Treat the installed binary's `-h` output as the implementation authority; record version and help output, and block rather than guess when incompatible.
- **Risk:** Shared `--Mapper.*` options may not be accepted by `global_mapper`. **Mitigation:** Verify each retained option against the global command help and keep method-specific argument lists explicit.
- **Risk:** The A/B runs may fail for dataset or external-tool reasons unrelated to command selection. **Mitigation:** Record exact stderr/exit status and distinguish command-contract failure from reconstruction-quality failure; do not expand the initiative to fix unrelated causes.
- **Decision needed:** Confirm the target COLMAP binary/version and approve the A/B dataset at design review. **Owner:** human approver.

## Design readiness checklist

- [x] Scope and exclusions are explicit.
- [x] Requirements and blueprints are linked.
- [x] Work orders are independently executable and ordered.
- [x] Validation commands and manual gates are identified.
- [x] Required human approval has been obtained or is pending at the documented gate.
