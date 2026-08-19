# Feature Requirements: 3D Gaussian Splatting Training and Outputs

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Evidence basis:** observed repository behavior and README product intent
- **Product overview:** [`../product-overview.md`](../product-overview.md)
- **Blueprint counterpart:** [`../../blueprints/features/training.md`](../../blueprints/features/training.md)

## Overview

After reconstruction, users need a repeatable training command that produces a portable 3D Gaussian Splatting model, intermediate checkpoints, and visual previews. The current product provides one gsplat backend and exposes future backends as roadmap items.

## Requirements

### REQ-TRAIN-001: Train from a sparse reconstruction

**User Story:** As an artist, I want to train a 3DGS scene from reconstructed cameras and points, so that I can obtain a viewable model.

**Acceptance Criteria:**

- **AC-TRAIN-001.1:** When a valid sparse model, prepared images, PyTorch, and gsplat are available, the system shall initialize and optimize Gaussian parameters.
- **AC-TRAIN-001.2:** The system shall select a CUDA device when available and otherwise report that CPU training is supported but slow.
- **AC-TRAIN-001.3:** On success, the system shall write a standard binary PLY final model and persist its relative path and completed iteration count.

### REQ-TRAIN-002: Provide progress artifacts

**User Story:** As an artist, I want previews and checkpoints during training, so that I can evaluate quality and recover from interruption.

**Acceptance Criteria:**

- **AC-TRAIN-002.1:** At the configured interval, the system shall save a checkpoint containing the trainable Gaussian state and a preview render.
- **AC-TRAIN-002.2:** A failed run shall be marked failed without being reported as complete.

### REQ-TRAIN-003: Support backend and run controls

**User Story:** As a user, I want explicit iteration, backend, resume, and restart controls, so that training is reproducible and bounded.

**Acceptance Criteria:**

- **AC-TRAIN-003.1:** The command shall expose iteration and preview cadence controls and reject unknown backends.
- **AC-TRAIN-003.2:** Resume precedence shall distinguish forced restart, explicit checkpoint, latest checkpoint, smart resume, and fresh initialization.

## Non-functional requirements

- **NFR-TRAIN-001:** CUDA out-of-memory failures shall provide mitigation guidance.
- **NFR-TRAIN-002:** The output PLY shall use the documented standard layout expected by common 3DGS viewers.

## Out of scope

- Brush, Inria, or nerfstudio training backends; these are roadmap items.
- GUI visualization or hosted training.

## Traceability notes

- **Related product goal:** one-command, inspectable, interoperable 3DGS training.
- **Dependencies:** PyTorch 2.4.1 CUDA 12.4, gsplat, Pillow, NumPy, and a completed COLMAP reconstruction.
- **Known risks:** full GPU training was not run during initialization; the source contains a training implementation but no dedicated training test module was observed.
