# Technical Context

> Navigable technical map based on repository evidence inspected during initialization.

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Evidence basis:** observed repository files; inferred runtime flow is labeled below

## Repository map

| Area                  | Path/signpost                                                                                            | Responsibility                                       |
| --------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| CLI entry point       | [`src/gsforge/cli.py`](../../src/gsforge/cli.py:1)                                                       | Typer commands and orchestration                     |
| Project state         | [`src/gsforge/project.py`](../../src/gsforge/project.py:1)                                               | Portable folder and `project.json` metadata          |
| Media preparation     | [`src/gsforge/ingest.py`](../../src/gsforge/ingest.py:1)                                                 | FFmpeg video and Pillow sequence ingest              |
| Reconstruction        | [`src/gsforge/sfm.py`](../../src/gsforge/sfm.py:1)                                                       | COLMAP discovery, pipeline, import/export            |
| Training              | [`src/gsforge/train.py`](../../src/gsforge/train.py:1)                                                   | COLMAP loading, gsplat optimization, PLY/checkpoints |
| Shared utilities      | [`src/gsforge/utils.py`](../../src/gsforge/utils.py:1)                                                   | Defaults, logging, progress, path resolution         |
| Tests                 | [`tests/test_ingest.py`](../../tests/test_ingest.py:1), [`tests/test_sfm.py`](../../tests/test_sfm.py:1) | Deterministic ingest and sparse-model unit coverage  |
| Packaging/environment | [`pyproject.toml`](../../pyproject.toml:1), [`pixi.toml`](../../pixi.toml:1)                             | Setuptools package and Windows Pixi environment      |

## Runtime and deployment topology

**Observed:** The package is installed as an editable PyPI dependency in a Pixi environment. The declared platform is Windows `win-64`, Python is 3.10, and CUDA system requirement is 12.4. The package script exposes `gsforge`.

**Inferred runtime flow:** `init-project` creates a project; `ingest` prepares frames; `sfm` or `import-colmap` supplies a sparse model; `train` consumes it and writes outputs; `info` reads metadata; `run-all` composes ingest, SfM, and training. There is no server, database service, or deployment manifest observed.

## Technology baseline

| Concern     | Choice            | Version/constraint                             | Confidence          |
| ----------- | ----------------- | ---------------------------------------------- | ------------------- |
| Language    | Python            | >=3.10,<3.11                                   | observed            |
| Environment | Pixi              | Windows win-64; lockfile present               | observed            |
| CLI         | Typer + Rich      | Typer >=0.12, Rich >=13                        | observed            |
| Media       | FFmpeg and Pillow | FFmpeg external; Pillow >=10                   | observed            |
| SfM         | COLMAP/GLOMAP     | COLMAP 4.x required for documented global mode | observed/documented |
| Training    | PyTorch + gsplat  | torch 2.4.1 cu124, gsplat >=1.3                | observed            |
| Packaging   | setuptools        | >=68, wheel, setuptools-scm                    | observed            |

## Cross-cutting concerns

- **Configuration and secrets:** CLI options and project metadata; no secret store or environment contract observed.
- **Authentication/authorization:** Not applicable to the local CLI baseline; no service boundary observed.
- **Errors and retries:** Fatal user-facing errors exit; training records failed state; no automatic retry policy observed.
- **Logging and observability:** Rich terminal logging, tqdm progress, and project `logs/` directory; structured logging contract is not observed.
- **Persistence and migrations:** JSON metadata with unknown-key filtering; no migration framework observed.
- **Testing strategy:** pytest unit tests use isolated temporary directories and mocks for external binaries; integration/manual coverage remains required for FFmpeg, COLMAP, CUDA, and training.

## Boundary inventory

Link each runnable boundary to a container blueprint under [`containers/`](containers/).

- [`gsforge-cli.md`](containers/gsforge-cli.md) — local CLI and pipeline boundary

## Architectural risks and unknowns

- **Verified discrepancy:** README describes GLOMAP/global mapping, but [`run_mapper()`](../../src/gsforge/sfm.py:327) currently leaves the global mapper argument commented out. This is a product/implementation alignment risk, not silently resolved during initialization.
- **Validation gap:** no `tests/test_train.py` was observed; full GPU training and PLY compatibility require manual validation.
- **External-tool risk:** FFmpeg and COLMAP availability/version are workstation-dependent and are not represented in Pixi package dependencies.
- **Documentation risk:** README claims a `LICENSE` file, but it was not present in the inspected file inventory.
