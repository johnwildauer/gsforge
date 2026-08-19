# Component Blueprint: Project State and Artifact Management

## Capability Summary

Owns project creation, metadata persistence, stage status, path derivation, and resume eligibility. It is shared by all pipeline commands.

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Containers:** [`gsforge-cli.md`](../containers/gsforge-cli.md)

## Core Components

```component
name: GSProject
container: gsforge CLI
responsibilities:
  - Create and load portable project directories
  - Derive canonical artifact paths
  - Persist stage metadata and status
```

```component
name: ProjectMeta
container: gsforge CLI
responsibilities:
  - Represent JSON-compatible pipeline metadata
  - Filter unknown fields for backward-compatible loading
  - Record ingest, SfM, and training outputs
```

`GSProject` owns the filesystem boundary and delegates computation to pipeline components. `ProjectMeta` is serialized by [`GSProject.save()`](../../../src/gsforge/project.py:254), including a temporary-file replacement step.

## System Contracts

### Key Contracts

- Stage transitions are persisted after ingest, SfM, and training.
- `best_sparse_dir` uses recorded selection and falls back to `sfm/sparse/0`.
- Smart resume requires completed training status, a recorded iteration, and an existing checkpoint.

### Integration Contracts

- Metadata is JSON in `project.json`.
- Canonical paths are exposed by properties in [`GSProject`](../../../src/gsforge/project.py:129).

## Architecture Decision Records

### ADR-001: Store portable relative paths

**Context:** Projects are intended to be moved and shared.

**Decision:** Store source and output paths relative to the project root where possible.

**Consequences:** A project can move between machines; external absolute inputs are copied into the project.
