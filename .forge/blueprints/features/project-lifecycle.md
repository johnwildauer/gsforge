# Feature Blueprint: Portable Project Lifecycle

## Feature Summary

The CLI creates and resolves a `.gsproject` directory through `GSProject`, persists `ProjectMeta` as `project.json`, and derives stage-specific artifact paths and resume state.

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Requirements:** [`../../requirements/features/REQ-PROJECT-LIFECYCLE.md`](../../requirements/features/REQ-PROJECT-LIFECYCLE.md)

## Component Blueprint Composition

Uses [`project-state.md`](../components/project-state.md), with CLI entry points [`init_project()`](../../../src/gsforge/cli.py:83), [`info()`](../../../src/gsforge/cli.py:590), and training resume logic in [`run_training()`](../../../src/gsforge/train.py:1636).

## Feature-Specific Components

```component
name: ProjectCommandAdapter
container: gsforge CLI
responsibilities:
  - Resolve command options into GSProject operations
  - Print project creation and status summaries
```

## Models and contracts

```model
name: ProjectMeta
store: project.json
description: Portable pipeline metadata
fields:
  - name: optional project name
  - input_type: video, images, or imported
  - sfm_status: pending, completed, or failed
  - training_status: pending, completed, or failed
constraints:
  - Paths are relative where portability requires it
```

### Key Contracts

- Stage methods persist metadata after state changes.

### Integration Contracts

- `project.json` is consumed by `info`, training resume logic, and subsequent commands.

## Requirement alignment

- REQ-PROJECT-001 → [`GSProject.create()`](../../../src/gsforge/project.py:153) creates canonical directories and metadata.
- REQ-PROJECT-002 → [`update_after_ingest()`](../../../src/gsforge/project.py:352), [`update_after_sfm()`](../../../src/gsforge/project.py:373), and [`update_after_training()`](../../../src/gsforge/project.py:399).
- REQ-PROJECT-003 → [`should_resume()`](../../../src/gsforge/project.py:521) and checkpoint resolution in [`run_training()`](../../../src/gsforge/train.py:1735).

## Architecture Decision Records

### ADR-001: Metadata is the pipeline status source

**Context:** Commands need a cheap, portable status view without rescanning all artifacts.

**Decision:** Persist authoritative stage metadata in `project.json` while retaining files as output evidence.

**Consequences:** Metadata can become stale after manual edits; commands must validate required files before consuming it.
