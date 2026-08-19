# Feature Requirements: Portable Project Lifecycle

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Evidence basis:** observed repository behavior and README product intent
- **Product overview:** [`../product-overview.md`](../product-overview.md)
- **Blueprint counterpart:** [`../../blueprints/features/project-lifecycle.md`](../../blueprints/features/project-lifecycle.md)

## Overview

Users need a portable project boundary that keeps source media, derived reconstruction data, model outputs, and pipeline status together. The project must be inspectable and movable without requiring the user to reconstruct hidden state.

## Terminology

| Term            | Definition                                                                      |
| --------------- | ------------------------------------------------------------------------------- |
| Project         | A self-contained directory ending in `.gsproject` or containing `project.json`. |
| Pipeline status | The persisted state of ingest, SfM, and training stages.                        |
| Checkpoint      | A saved intermediate training state that can be resumed.                        |

## Requirements

### REQ-PROJECT-001: Create a self-contained project

**User Story:** As a virtual-production artist, I want to create a named project, so that all pipeline artifacts have a predictable portable home.

**Acceptance Criteria:**

- **AC-PROJECT-001.1:** When project creation succeeds, the system shall create the project metadata file and canonical source, preprocessing, reconstruction, model, render, and log areas.
- **AC-PROJECT-001.2:** When a project is loaded, the system shall accept either its `.gsproject` directory name or a directory containing the metadata file.

### REQ-PROJECT-002: Persist pipeline state

**User Story:** As a user, I want pipeline results and status persisted, so that I can inspect progress and continue work later.

**Acceptance Criteria:**

- **AC-PROJECT-002.1:** After each completed or failed pipeline stage, the system shall persist the stage status and relevant counts or output paths.
- **AC-PROJECT-002.2:** The status command shall present project identity, ingest, reconstruction, and training state without requiring a new pipeline run.

### REQ-PROJECT-003: Resume or restart training safely

**User Story:** As a user, I want to resume from a known checkpoint or explicitly restart, so that interrupted or iterative training does not silently lose work.

**Acceptance Criteria:**

- **AC-PROJECT-003.1:** When a valid checkpoint is selected, training shall resume from its recorded iteration.
- **AC-PROJECT-003.2:** When restart is explicitly requested, training shall ignore existing checkpoints and start from the reconstruction.
- **AC-PROJECT-003.3:** When checkpoint metadata is missing or invalid, the system shall report the failure and avoid treating the run as complete.

## Non-functional requirements

- **NFR-PROJECT-001:** Metadata writes should be crash-tolerant enough to avoid replacing a valid metadata file with a partial JSON document.
- **NFR-PROJECT-002:** Project paths stored in metadata should remain portable across machines.

## Out of scope

- GUI or web project management.
- Remote artifact storage or synchronization.
- Automatic deletion of user artifacts.

## Traceability notes

- **Related product goal:** portable, reproducible, scriptable 3DGS workflows.
- **Dependencies:** filesystem access and JSON serialization.
- **Known risks:** no repository-level end-to-end test was observed for full project creation, resume, or status flows.
