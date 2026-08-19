# Feature Blueprint: Structure-from-Motion Reconstruction and COLMAP Interchange

## Feature Summary

The reconstruction flow in [`run_sfm()`](../../../src/gsforge/sfm.py:637) finds COLMAP, extracts and matches features, maps frames, selects the strongest sparse model, and records state. Import/export provide alternate and downstream paths.

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Requirements:** [`../../requirements/features/REQ-RECONSTRUCTION.md`](../../requirements/features/REQ-RECONSTRUCTION.md)

## Component Blueprint Composition

Uses [`reconstruction-pipeline.md`](../components/reconstruction-pipeline.md) and [`project-state.md`](../components/project-state.md). `preprocess/` feeds COLMAP; `sfm/` stores its database and sparse models.

## Feature-Specific Components

```component
name: ReconstructionCommand
container: gsforge CLI
responsibilities:
  - Enforce ingest/SfM preconditions
  - Select method and expose completion summary
```

## Models and contracts

```model
name: SfmResult
store: in-memory command result plus project.json
description: Status, registered camera count, and selected sparse directory
fields:
  - status: completed or failed
  - camera_count: registered images
  - sparse_dir: selected COLMAP model
constraints:
  - Sparse files use COLMAP binary or text conventions
```

## Requirement alignment

- REQ-RECON-001 → [`run_sfm()`](../../../src/gsforge/sfm.py:637) and [`select_best_sparse_model()`](../../../src/gsforge/sfm.py:560).
- REQ-RECON-002 → [`import_colmap_reconstruction()`](../../../src/gsforge/sfm.py:751).
- REQ-RECON-003 → [`export_colmap()`](../../../src/gsforge/sfm.py:845).

## Architecture Decision Records

### ADR-001: Keep multiple sparse models until selection

**Context:** COLMAP can produce multiple disconnected models.

**Decision:** Enumerate valid numbered models and select by registered-image count.

**Consequences:** The best model is deterministic; analyzer compatibility is an external version gate.
