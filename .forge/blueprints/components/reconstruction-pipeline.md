# Component Blueprint: Reconstruction Pipeline

## Capability Summary

Runs COLMAP feature extraction, matching, mapping, sparse-model discovery/selection, external import, and standard export.

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Containers:** [`gsforge-cli.md`](../containers/gsforge-cli.md)

## Core Components

```component
name: ColmapRunner
container: gsforge CLI
responsibilities:
  - Discover a project-local or PATH COLMAP binary
  - Run extraction, matching, and mapping steps
```

```component
name: SparseModelSelector
container: gsforge CLI
responsibilities:
  - Enumerate valid numbered sparse models
  - Analyze registered-image counts
  - Select the best model or deterministic fallback
```

```component
name: ColmapInterchange
container: gsforge CLI
responsibilities:
  - Copy external sparse models into a project
  - Export images and sparse files in standard layout
```

Implemented in [`sfm.py`](../../../src/gsforge/sfm.py:104), [`select_best_sparse_model()`](../../../src/gsforge/sfm.py:560), and [`export_colmap()`](../../../src/gsforge/sfm.py:845).

## System Contracts

### Key Contracts

- Valid sparse models contain recognized binary or text COLMAP files.
- Best-model selection maximizes registered images and keeps the lowest model index on ties.
- Missing COLMAP or failed steps are fatal to the command and should leave failed stage state.

### Integration Contracts

- COLMAP input/output uses `cameras`, `images`, and `points3D` binary/text files.
- External export uses `images/` and `sparse/0/`.

## Architecture Decision Records

### ADR-001: Preserve COLMAP as the interchange boundary

**Context:** The README lists COLMAP GUI, nerfstudio, LichtFeld, and other consumers.

**Decision:** Keep sparse reconstructions in standard COLMAP form and provide import/export rather than a proprietary format.

**Consequences:** Interoperability is strong; COLMAP version and external binary availability are manual gates.
