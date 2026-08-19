# Work Order: WO-003 — GLOMAP CPU Fallback Reporting

## Metadata

- **Status:** `complete-with-warnings`
- **Assignee:** `gsforge implementer`
- **Phase/order:** `2`
- **Initiative:** [`../initiative-v1.md`](../initiative-v1.md)
- **Owner/orchestrator:** `gsforge maintainers`
- **Review artifact location:** [`../reviews/`](../reviews/)

## Summary

Make the supported GLOMAP runtime's CPU Ceres/cuDSS fallback explicit and truthful. The work order must preserve GLOMAP execution with official binaries, report when bundle adjustment is CPU-bound, and document the unsupported custom-build path for users who require GPU bundle adjustment. CPU fallback is valid for the standard binary workflow; gsforge must not misrepresent it as GPU execution.

## Target evidence received

- Target: COLMAP 4.0.2 Windows binary, commit `d927f7e` dated `2026-03-18`, installed at `bin/colmap-x64-windows-cuda/bin/colmap.exe`.
- Host: NVIDIA RTX 4090, NVIDIA Studio Driver 610.88, CUDA Toolkit/runtime 12.4.
- `global_mapper -h` exposes `GlobalMapper.gp_use_gpu=1` and `GlobalMapper.ba_ceres_use_gpu=1`, both defaulting to enabled.
- The existing run in [`garden-glomap-sfm.txt`](../../../garden-glomap-sfm.txt) shows GPU SIFT extraction/matching, but global positioning and bundle adjustment report Ceres compiled without CUDA and without cuDSS, then use CPU dense/sparse solvers.

This establishes a binary/build capability mismatch rather than a missing gsforge command-line option. The next investigation must determine whether the distributed COLMAP 4.0.2 binary is expected to provide GPU-enabled Ceres/cuDSS on Windows, or whether the release package cannot satisfy the requested GLOMAP GPU contract.

## Changelog evidence and revised investigation path

The official [COLMAP changelog](https://colmap.github.io/changelog.html) materially changes the likely remediation path:

- COLMAP 4.0.0 introduced GLOMAP as a first-class `global_mapper` pipeline and documented automatic CPU BA fallback when the GPU solver fails. This confirms that the observed CPU fallback is expected behavior in the current product, not evidence that gsforge silently changed mapper methods.
- COLMAP 4.0.3 fixed piping of bundle-adjustment options into the global mapper. The current 4.0.2 target predates that fix, so upgrading is justified independently of the Ceres build warning.
- COLMAP 4.1.0 added Caspar, a GPU-accelerated bundle-adjustment backend selectable as an alternative to the default Ceres solver. It also added GPU device selection and backend controls.
- COLMAP 4.1.1 is the latest stable release listed by the changelog and includes additional Windows/CUDA-related fixes, although the changelog does not by itself prove that a Windows release bundles CUDA-enabled Ceres or Caspar.

The preferred next experiment is therefore an official COLMAP 4.1.1 Windows CUDA release, followed by `global_mapper -h` inspection and the same 185-frame A/B run. The experiment must determine whether the release exposes a GPU BA backend usable by GLOMAP and must capture explicit solver/backend messages. Do not modify gsforge's mapper dispatch until this binary-level experiment is complete.

## Issue and FAQ evidence — supported interpretation

The official [issue #3474](https://github.com/colmap/colmap/issues/3474) reproduces the same warning on the Windows CUDA prebuilt binary. The official FAQ states that CUDA-enabled Ceres is not included in official CUDA binaries until Ceres 2.3 is officially released; GPU Ceres/cuDSS therefore requires compiling Ceres with CUDA/cuDSS and linking that build to COLMAP.

The FAQ also distinguishes mapper-specific options from the current GLOMAP path:

- `Mapper.ba_use_gpu` applies to the incremental mapper, not `global_mapper`.
- `BundleAdjustmentCeres.use_gpu` applies to the standalone `bundle_adjuster` command.
- Caspar is selectable for incremental mapper backends, but the FAQ explicitly states that `global_mapper` does not expose a Caspar backend selector.
- Therefore, no additional gsforge command-line flag can turn the official Windows `global_mapper` binary's CPU fallback into GPU Ceres/cuDSS. The current `GlobalMapper.ba_ceres_use_gpu=1` request is already the correct request; the linked Ceres capability is absent at build time.

WO-003 focuses on high-level truthful reporting of this known binary limitation. Standard gsforge supports GLOMAP with CPU BA when official binaries fall back. Users who require GPU BA must independently compile COLMAP/Ceres or experiment with Caspar; that path is unsupported and does not need to be made robust by gsforge.

## Timing evidence from the 185-frame run

The GLOMAP mapper reported `204.584 s` total. Within that:

- Global positioning: `40.704 s`.
- Iterative bundle adjustment: `86.512 s`.
- Iterative retriangulation/refinement: `74.701 s`.

The BA phase is approximately 42% of mapper time and approximately 29% of the full SfM run when combined with the recorded feature extraction (`0.851 min`) and matching (`0.642 min`) phases. It is a meaningful optimization opportunity, but not the whole pipeline. The standard workflow accepts COLMAP's own fallback behavior and reports it at a high level.

## In scope

- Inspect target GLOMAP/global-mapper help, runtime output, and accepted solver/CUDA/Ceres/cuDSS options.
- Identify the exact COLMAP binary build and whether its bundled Ceres was compiled with CUDA and cuDSS support.
- Determine and document the supported official-binary behavior and the unsupported custom-build path without changing gsforge's mapper dispatch.
- Implement the smallest approved GLOMAP-only diagnostic or compatibility change so CPU fallback is visible and accurately reported.
- Add mocked command/diagnostic tests and retain the existing real-data validation path.

## Out of scope

- Installing or rebuilding COLMAP, Ceres, CUDA, cuDSS, PyTorch, or gsplat.
- General GPU enablement, feature extraction GPU options, or incremental COLMAP solver changes.
- Silently claiming GPU solver execution, changing mapper selection, or claiming qualitative superiority.

## Requirements

- [`AC-RECON-001.1`](../../../requirements/features/REQ-RECONSTRUCTION.md:24): **When prepared frames exist and a compatible COLMAP binary is available, the system shall run feature extraction, matching, and mapping.**
- [`NFR-RECON-001`](../../../requirements/features/REQ-RECONSTRUCTION.md:48): **External binary failures shall preserve a failed project status and expose actionable diagnostics.**

## Blueprints

- [`reconstruction-pipeline.md`](../../../blueprints/components/reconstruction-pipeline.md:16) — `ColmapRunner` owns global mapping and external binary failure behavior.
- [`reconstruction.md`](../../../blueprints/features/reconstruction.md:5) — `run_sfm()` persists stage outcome and selected sparse model.

## Implementation plan

1. Capture target binary/version/help evidence and reproduce the Ceres/cuDSS warning in an isolated project without modifying user media.
2. Identify why the existing binary reports missing CUDA/cuDSS support and determine whether a supported GPU-capable build is available.
3. Define the approved contract: official-binary GLOMAP completion with CPU BA is valid but must be explicitly reported; GPU BA remains an optional unsupported custom-build path.
4. Implement command/configuration and diagnostic changes only for the approved GLOMAP path; retain global command selection from the predecessor.
5. Add tests for solver-mode diagnostics, warning/error classification, and failed-stage state; run normal regression checks and retain real-data evidence for the official binary.

## Files and systems

- **Create:** no production file expected; test fixtures may be added only if required for deterministic diagnostics.
- **Update:** likely [`src/gsforge/sfm.py`](../../../src/gsforge/sfm.py:327), focused tests, and execution evidence; exact files depend on verified binary contract.
- **Avoid changing:** incremental mapper behavior, calibration policy, training, and system-installed binaries.

## Validation and E2E acceptance tests

- **Automated checks:** focused `run_mapper()`/failure-state tests; `pixi run test`; `pixi run lint`; `pixi run format`, or executable-compatible equivalents.
- **Binary contract checks:** target `global_mapper -h` plus recorded runtime output for GPU-capable and CPU/no-cuDSS conditions.
- **Manual/external checks:** isolated prepared-frame project, explicitly exercising the approved CPU fallback or diagnostic block; record binary build, environment, command, exit status, solver messages, camera count, and sparse model.
- **Acceptance test:** Given the supported official binary, when GLOMAP runs and COLMAP reports a Ceres solver result or CPU fallback, then GLOMAP may complete, the high-level solver limitation is reported when detectable, and the result remains identified as GLOMAP.
- **Recovery:** if the mapper itself fails, preserve the failed state and provide actionable diagnostics; do not auto-switch to incremental COLMAP. Custom GPU builds remain user-managed, untested, and unsupported.

## Documentation updates

- Update blueprints or requirements only if the verified fallback becomes a durable supported contract; otherwise record the warning/decision in the initiative log only.
- Append binary evidence, fallback decision, validation, and any blocked incompatibility to [`../execution-log.md`](../execution-log.md).

## Completion checklist

- [x] Target solver/fallback contract is evidenced.
- [x] Implementation and focused tests are complete within scope.
- [x] CPU-oriented external gate and automated checks have evidence.
- [x] Authorized durable documentation is reconciled or explicitly unchanged.
- [x] Independent review artifact is under [`../reviews/`](../reviews/).
- [x] Execution log is appended.
