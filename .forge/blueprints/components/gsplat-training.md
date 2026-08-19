# Component Blueprint: gsplat Training

## Capability Summary

Loads COLMAP data, converts camera poses, initializes or restores Gaussian parameters, optimizes with gsplat, saves progress artifacts, and writes a standard PLY model.

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Containers:** [`gsforge-cli.md`](../containers/gsforge-cli.md)

## Core Components

```component
name: ColmapDataLoader
container: gsforge CLI
responsibilities:
  - Load binary-first or text-fallback cameras, images, and points
  - Normalize camera intrinsics and poses
```

```component
name: GsplatTrainer
container: gsforge CLI
responsibilities:
  - Initialize or restore Gaussian parameters
  - Optimize with adaptive density control
  - Save checkpoints, previews, and final PLY
```

Implemented in [`load_colmap_data()`](../../../src/gsforge/train.py:615), [`GsplatTrainer`](../../../src/gsforge/train.py:792), and [`run_training()`](../../../src/gsforge/train.py:1636).

## System Contracts

### Key Contracts

- Training requires a valid sparse model and at least one prepared image.
- A checkpoint carries iteration, Gaussian tensors, and loss state.
- CUDA is preferred; CPU is an explicit slow fallback.
- OOM errors include recovery suggestions.

### Integration Contracts

- Input is a COLMAP sparse model plus matching `preprocess/frame_*.png` images.
- Outputs are `models/checkpoints/ckpt_NNNNNN.pth`, `renders/preview_NNNNNN.png`, and `models/final_scene.ply`.

## Architecture Decision Records

### ADR-001: Abstract training backends

**Context:** The README roadmap names Brush and Inria backends, while the current registry contains only gsplat.

**Decision:** Keep [`BaseTrainer`](../../../src/gsforge/train.py:181) as the extension point and expose only the registered gsplat backend in the baseline.

**Consequences:** Current scope is clear; future backends can share project and CLI contracts.
