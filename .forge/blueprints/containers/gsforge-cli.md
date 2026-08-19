# Container Blueprint: gsforge CLI

## Container Summary

The CLI is the single runnable product boundary. It parses user commands, resolves a project path, delegates to pipeline modules, and reports outcomes. It is consumed by artists and automation from a Pixi-managed Python environment.

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Technical context:** [`../technical-context.md`](../technical-context.md)

## Infrastructure

- **Technology/runtime:** Python 3.10, Typer, Rich; entry point [`gsforge.cli:app()`](../../../src/gsforge/cli.py:60)
- **Build/package:** setuptools via [`pyproject.toml`](../../../pyproject.toml:1), editable Pixi dependency via [`pixi.toml`](../../../pixi.toml:15)
- **Deployment/distribution:** local Pixi environment; CLI script `gsforge`
- **Configuration:** command options and project-relative `project.json`; no service configuration observed
- **Dependencies:** FFmpeg, COLMAP, PyTorch/CUDA, gsplat, Pillow, OpenCV, Rich, tqdm

## Entry points and boundaries

- **Inputs:** `init-project`, `ingest`, `sfm`, `import-colmap`, `export-colmap`, `train`, `info`, and `run-all` commands in [`cli.py`](../../../src/gsforge/cli.py:78)
- **Outputs:** terminal summaries/errors and project artifacts under a `.gsproject` directory
- **External systems:** FFmpeg/ffprobe, COLMAP, CUDA/PyTorch, gsplat, filesystem
- **Internal areas:** project state, ingest, SfM, training, shared utilities

## System Contracts

### Key Contracts

- Commands operating on a project resolve an explicit project or search parents for one.
- Fatal pipeline errors terminate the command; stage failures update persisted status where implemented.
- Project-relative metadata paths preserve portability.

### Integration Contracts

- The package script maps `gsforge` to [`app()`](../../../src/gsforge/cli.py:60).
- The project folder contract is documented in [`README.md`](../../../README.md:373) and managed by [`GSProject.create()`](../../../src/gsforge/project.py:153).

## Architecture Decision Records

### ADR-001: Keep the product CLI-first

**Context:** The repository and README define command workflows and no GUI runtime.

**Decision:** Treat the CLI and portable project folder as the current product boundary.

**Consequences:** Automation and manual commands are first-class; GUI work remains out of the initialization baseline.
