# Product Overview

> Durable product definition. Statements are marked observed, inferred, or desired where that distinction matters.

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Evidence basis:** repository observation and README product intent

## Business problem

Virtual-production artists need a reproducible way to turn raw video or rendered image sequences into a usable 3D Gaussian Splatting scene. The surrounding tools have substantial setup and interoperability costs: frame preparation, COLMAP reconstruction, training, progress inspection, and handoff must otherwise be managed separately. gsforge packages that workflow behind a CLI and a portable project folder.

## Current state

**Observed:** The repository is a Python 3.10 CLI package with a Typer entry point, Pixi environments, unit tests for ingest and sparse-model selection, and a README documenting initialization, ingest, SfM, import/export, training, status, and run-all commands. The implementation stores state in `project.json` and writes source, prepared frames, COLMAP output, checkpoints, previews, final PLY, and logs under a `.gsproject` directory.

**Inferred:** The primary user is an artist or technical artist working on a workstation, with portability and interoperability more important than a GUI in the current baseline.

**Current gap:** Real external-tool execution, GPU training, and end-to-end command validation are not covered by the observed unit tests and require a manual workstation gate.

## Personas and stakeholders

| Persona/stakeholder            | Goals                                                                | Pain points                                                       | Success looks like                                                              |
| ------------------------------ | -------------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Virtual-production artist      | Convert footage or image sequences into a scene and inspect progress | Toolchain setup, long runtimes, opaque failures                   | A portable project and standard PLY output from scriptable commands             |
| Technical artist / pipeline TD | Automate repeatable reconstruction and handoff                       | Inconsistent folder formats and external-tool integration         | Stable CLI commands, persisted metadata, and COLMAP interchange                 |
| Maintainer                     | Extend backends and preserve behavior                                | GPU/external binary dependencies and limited integration coverage | Focused modules, tests for deterministic logic, and auditable process artifacts |

## Product description

gsforge is a CLI-first pipeline that creates a portable project, ingests video or numbered image sequences into prepared PNG frames, runs or imports a COLMAP-compatible sparse reconstruction, trains a 3DGS model with the gsplat backend, saves checkpoints and preview renders, reports status, and exports a standard COLMAP folder. The user-facing boundary is the command line; project folders are the durable artifact boundary; external FFmpeg, COLMAP, CUDA/PyTorch, and gsplat are runtime dependencies.

## MVP and boundaries

### In scope

- Creating and inspecting portable projects.
- Video and image-sequence ingest with bounded uniform sampling and optional downscale.
- Global/incremental SfM invocation, external reconstruction import, and COLMAP export.
- gsplat training with checkpoints, previews, resume/restart controls, and standard PLY output.

### Out of scope

- GUI or web interfaces.
- Additional training backends listed in the README roadmap.
- Quality filtering, multi-camera rigs, dense reconstruction, mesh generation, and hosted execution.

## Success metrics

| Metric               | Baseline                         | Target                                                                          | Measurement method                        |
| -------------------- | -------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------- |
| Runnable baseline    | No baseline measurement recorded | Unit tests pass; external-tool smoke tests pass when dependencies are installed | `pixi run test`, plus manual gates        |
| Portable handoff     | No baseline measurement recorded | Export contains standard COLMAP layout and final output is standard PLY         | CLI smoke test and artifact inspection    |
| Iteration visibility | No baseline measurement recorded | Checkpoints and previews are produced at configured intervals                   | Training smoke test on a CUDA workstation |

## Product constraints

- Python is constrained to `>=3.10,<3.11`; Pixi declares Windows `win-64` and CUDA 12.4.
- The documented CUDA/PyTorch/gsplat combination is workstation-oriented; CPU fallback exists but is not a production-performance target.
- Large project media and generated outputs are ignored by [`../../.gitignore`](../../.gitignore); source repository commits should remain lightweight.
- MIT licensing is stated in the README, but the referenced license file was not present in the inspected repository inventory; confirm before release.

## Feature map

Link each user-facing capability to one file under [`features/`](features/).

- [`REQ-PROJECT-LIFECYCLE.md`](features/REQ-PROJECT-LIFECYCLE.md) — portable project creation, state, and recovery controls
- [`REQ-MEDIA-INGEST.md`](features/REQ-MEDIA-INGEST.md) — video/image-sequence preparation
- [`REQ-RECONSTRUCTION.md`](features/REQ-RECONSTRUCTION.md) — SfM, import, and COLMAP interchange
- [`REQ-TRAINING.md`](features/REQ-TRAINING.md) — gsplat training, progress artifacts, and standard PLY output

## Open questions

- Is `master` the permanent release branch, or will a named release branch policy be adopted later? Current policy treats `master` as the default from Git metadata and allows a future specified release branch.
- Which real-footage and CUDA workstation fixture should be the acceptance fixture for manual end-to-end validation?
- Should the documented `LICENSE` claim be retained after the missing file is supplied?

## Initialization handoff recommendation

The repository is documented but not yet demonstrably runnable because the available validation commands were not executed in this session. The first initiative should be a small validation-and-alignment slice: run the Pixi test/lint baseline, perform external-tool smoke checks in an isolated project, and reconcile the documented GLOMAP behavior with [`run_mapper()`](../../src/gsforge/sfm.py:327). Do not treat this recommendation as an initiative; initialization creates no initiative artifact.
