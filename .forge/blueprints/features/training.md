# Feature Blueprint: 3D Gaussian Splatting Training and Outputs

## Feature Summary

Training loads COLMAP data and prepared images, initializes or restores Gaussian tensors, rasterizes with gsplat, applies adaptive density control, and saves checkpoints, previews, and a standard PLY.

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Requirements:** [`../../requirements/features/REQ-TRAINING.md`](../../requirements/features/REQ-TRAINING.md)

## Component Blueprint Composition

Uses [`gsplat-training.md`](../components/gsplat-training.md) and [`project-state.md`](../components/project-state.md). The CLI adapter is [`train()`](../../../src/gsforge/cli.py:456), while backend selection uses [`get_trainer()`](../../../src/gsforge/train.py:1603).

## Feature-Specific Components

```component
name: TrainingCommand
container: gsforge CLI
responsibilities:
  - Resolve resume/restart options
  - Select the registered backend
  - Report output and failure state
```

## Models and contracts

```model
name: TrainingResult
store: in-memory command result plus project.json and model artifacts
description: Final PLY, iteration count, device, and duration
fields:
  - final_ply: portable output path
  - iterations: target count
  - device: cuda or cpu
constraints:
  - PLY uses standard binary little-endian layout
```

## Requirement alignment

- REQ-TRAIN-001 → [`GsplatTrainer.train()`](../../../src/gsforge/train.py:830) and [`load_colmap_data()`](../../../src/gsforge/train.py:615).
- REQ-TRAIN-002 → checkpoint/preview writes in [`_train_with_gsplat()`](../../../src/gsforge/train.py:1030).
- REQ-TRAIN-003 → resume precedence and backend registry in [`run_training()`](../../../src/gsforge/train.py:1636).

## Architecture Decision Records

### ADR-001: Save standard PLY plus resumable checkpoints

**Context:** Users need both interoperability and recovery during long GPU runs.

**Decision:** Emit standard PLY for handoff and `.pth` checkpoints for gsforge resume.

**Consequences:** PLY is broadly consumable; checkpoints remain implementation/backend-specific.
