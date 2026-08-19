# Initiative: Harden COLMAP Runtime Compatibility

## Initiative metadata

- **Status:** `closed-with-follow-up`
- **Owner/orchestrator:** `gsforge maintainers`
- **Created:** `2026-08-19`
- **Updated:** `2026-08-19`
- **Current gate:** `closed`
- **Authoritative version:** `v1`
- **Development branch:** `initiative/2026-08-19-colmap-runtime-robustness` (required; creation evidence to be recorded before implementation)
- **Predecessor:** [`2026-08-19-glomap-mapper-selection`](../2026-08-19-glomap-mapper-selection/initiative-v1.md)

## Outcome

Make the COLMAP-backed reconstruction path tolerant of observed binary capability differences and ensure the GLOMAP path truthfully reports the solver mode used by the supported official binary. Focal-length calibration is explicitly deferred to a later initiative; this initiative documents the capture requirement that footage maintain consistent focal length. The initiative must preserve the working global/incremental dispatch from the predecessor initiative and leave each active concern independently reviewable.

## Scope

### In scope

- Replace the unconditional `colmap --version` probe with capability/version discovery that recognizes the observed binary contract (`help` is supported while `--version` is not), without making version metadata a prerequisite for reconstruction.
- Document and diagnose the existing GLOMAP binary's missing CUDA/cuDSS solver support; CPU fallback is an acceptable official-binary GLOMAP outcome when clearly reported.
- Document consistent focal length across source footage as a prerequisite for reliable reconstruction; do not add calibration controls in this initiative.
- Add focused automated coverage and bounded external/manual validation plans for each work order.
- Record any authorized durable requirement or blueprint changes only after implementation evidence establishes the final contract.

### Out of scope

- Windows Unicode console behavior; the predecessor's symptom was caused by two separate development terminals, not a product portability requirement.
- Replacing COLMAP, changing feature extraction/matching, changing mapper selection, or changing sparse-model selection.
- General GPU performance tuning, CUDA/PyTorch/gsplat training changes, dependency upgrades, or support for every historical COLMAP build.
- Qualitative reconstruction optimization beyond verifying that the selected command completes and produces inspectable output.
- Production code, tests, or runtime configuration changes during this planning/design-review phase.

## Context and motivation

The closed predecessor recorded four follow-ups. Its execution evidence shows the target Windows COLMAP binary emitted a diagnostic that `--version` was unsupported and directed users to `colmap help`. The current [`check_colmap_version()`](../../../src/gsforge/sfm.py:154) treats the failed probe as non-fatal but does not identify capabilities. The current [`run_mapper()`](../../../src/gsforge/sfm.py:327) always enables focal-length refinement, while the approved baseline explicitly excludes automatic camera calibration. The same GLOMAP run also left a non-fatal CPU Ceres/cuDSS fallback warning.

These are related by their external-binary compatibility boundary, but they have different ownership and acceptance evidence. They are therefore three sequential work orders rather than one broad implementation order.

## Requirements and blueprint references

- **Baseline requirements:** [`REQ-RECONSTRUCTION.md`](../../requirements/features/REQ-RECONSTRUCTION.md:18), especially AC-RECON-001.1, AC-RECON-001.2, and NFR-RECON-001.
- **Calibration constraint:** [`REQ-RECONSTRUCTION.md`](../../requirements/features/REQ-RECONSTRUCTION.md:51), which currently lists automatic camera calibration beyond selected defaults as out of scope. This initiative does not change that boundary; focal calibration is deferred.
- **Feature blueprint:** [`reconstruction.md`](../../blueprints/features/reconstruction.md:5), especially the `run_sfm()` flow and external command boundary.
- **Component blueprint:** [`reconstruction-pipeline.md`](../../blueprints/components/reconstruction-pipeline.md:16), especially `ColmapRunner` and external-binary failure behavior.
- **CLI boundary:** [`gsforge-cli.md`](../../blueprints/containers/gsforge-cli.md:1).
- **Process policy:** [`00-overview.md`](../../process/00-overview.md:63), including the dedicated-branch, sequential-order, manual-gate, executable-terminal, and reviewer-location policies.

## Proposed design

Keep `ColmapRunner` as the single owner of external COLMAP invocation. First establish a capability probe that prefers supported help output, captures version text when available, and distinguishes unknown version from unavailable binary without aborting otherwise compatible workflows. Then verify and report whether GLOMAP's bundle-adjustment solver is GPU-backed or CPU-fallback in the supported target environment. If the binary uses CPU fallback, preserve the truthful diagnostic while allowing the valid GLOMAP result to complete. User-built GPU-capable variants remain unsupported and outside the standard runtime contract.

All implementation work orders are sequential. WO-001 provides capability evidence and probe helpers used by WO-003. WO-003 consumes the verified GLOMAP command and runtime capability model. The deferred calibration work order remains recorded as a later-initiative candidate and is not executable under this initiative. Each active work-order reviewer must place an independent artifact in [`reviews/`](reviews/).

## Work-order plan

| Order | Work order                                                                                       | Purpose                                                                                    | Dependencies                             | Review cadence                                                                                                              |
| ----- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 1     | [`work-orders/WO-001-colmap-capability-probe.md`](work-orders/WO-001-colmap-capability-probe.md) | Replace the unsupported version probe with resilient capability discovery and diagnostics. | none                                     | per work order; review in [`reviews/WO-001-design-review.md`](reviews/WO-001-design-review.md) or implementation equivalent |
| 2     | [`work-orders/WO-003-glomap-cpu-fallback.md`](work-orders/WO-003-glomap-cpu-fallback.md)         | Make official-binary CPU BA fallback explicit and document unsupported GPU custom builds.  | WO-001; verified global command contract | per work order; review in [`reviews/WO-003-design-review.md`](reviews/WO-003-design-review.md) or implementation equivalent |

WO-002 is retained as a deferred planning record only. It must not be implemented or counted toward this initiative's acceptance.

## Acceptance criteria

- The initiative has a dedicated branch recorded and all design, work-order, implementation, and closeout reviews are stored under its [`reviews/`](reviews/) directory.
- A COLMAP binary that rejects `--version` but supports `help` can complete the probe without a traceback or false hard failure; supported version/capability evidence is logged when available.
- README documentation states that source footage must maintain consistent focal length; no focal-calibration behavior is changed by this initiative.
- GLOMAP behavior on the target official binary is verified against command help and a controlled external run; CPU fallback is explicitly reported when present, while actual mapper failures preserve failed-stage state with actionable diagnostics.
- Existing `glomap` global and `colmap` incremental command forms remain intact unless a work order explicitly proves and authorizes a compatible adjustment.
- Every work order has automated checks, external/manual gates, risks, recovery behavior, and an execution-log entry before it can be marked complete.
- No production code or tests are implemented before this initiative passes design review and receives human design approval.

## Risks and decisions

- **Risk:** COLMAP help/version syntax varies by release. **Mitigation:** use the target binary's observed help output as authority, preserve raw evidence, and block unsupported assumptions.
- **Decision:** Focal-length calibration is deferred to a later initiative. Consistent focal length across source footage is a documented prerequisite for this tool's reconstruction workflow.
- **Risk:** The project-local COLMAP binary may bundle Ceres without CUDA/cuDSS support. **Mitigation:** report CPU fallback truthfully, document the Ceres 2.3/custom-build status, and do not claim GPU execution.
- **Risk:** The executable terminal cannot run PowerShell. **Mitigation:** record the constraint in policy, use executable-compatible validation, and log any required terminal switch.
- **Decision:** Standard official binaries use option 1: GLOMAP with explicitly reported CPU BA fallback is valid. GPU BA through custom COLMAP/Ceres or Caspar builds is untested, unsupported, and user-managed.

## Design readiness checklist

- [x] Scope and exclusions are explicit, including the rejected Unicode-console follow-up.
- [x] Requirements and blueprints are linked, with the calibration conflict called out.
- [x] Active work orders are independently scoped and sequentially ordered; focal calibration is explicitly deferred.
- [x] Automated, external, manual, branch, reviewer-location, and terminal constraints are identified.
- [x] Independent design review has passed.
- [x] Human design approval has been recorded.
