# Feature Requirements: Structure-from-Motion Reconstruction and COLMAP Interchange

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Evidence basis:** observed repository behavior and README product intent
- **Product overview:** [`../product-overview.md`](../product-overview.md)
- **Blueprint counterpart:** [`../../blueprints/features/reconstruction.md`](../../blueprints/features/reconstruction.md)

## Overview

The product turns prepared frames into a COLMAP-compatible sparse reconstruction, preferring the global method for typical VP footage while retaining classic incremental SfM and import/export paths for interoperability.

## Requirements

### REQ-RECON-001: Produce a sparse reconstruction

**User Story:** As an artist, I want camera poses and sparse points from prepared frames, so that I can train or hand off a scene reconstruction.

**Acceptance Criteria:**

- **AC-RECON-001.1:** When prepared frames exist and a compatible COLMAP binary is available, the system shall run feature extraction, matching, and mapping.
- **AC-RECON-001.2:** The command shall expose global and incremental method choices and persist the selected method and outcome; the documented global mapper behavior is a current implementation alignment gap requiring verification or correction in a future initiative.
- **AC-RECON-001.3:** The system shall select the valid sparse model with the greatest registered-image count when multiple models are produced.

### REQ-RECON-002: Import an existing reconstruction

**User Story:** As a user with an external COLMAP result, I want to import it, so that I can continue with gsforge training without rerunning SfM.

**Acceptance Criteria:**

- **AC-RECON-002.1:** When a supplied directory contains recognized COLMAP binary or text model files, the system shall copy them into the project reconstruction area.
- **AC-RECON-002.2:** The system shall persist imported reconstruction status and registered camera count.

### REQ-RECON-003: Export a standard interchange folder

**User Story:** As a user, I want a standard COLMAP folder, so that other viewers and training tools can consume the reconstruction.

**Acceptance Criteria:**

- **AC-RECON-003.1:** After a completed reconstruction, export shall create `images/` and `sparse/0/` with the prepared images and sparse model files.
- **AC-RECON-003.2:** Export shall fail clearly when the required reconstruction is absent.

## Non-functional requirements

- **NFR-RECON-001:** External binary failures shall preserve a failed project status and expose actionable diagnostics.
- **NFR-RECON-002:** Standard COLMAP file formats shall be retained for interoperability.

## Out of scope

- Replacing COLMAP with a new SfM engine.
- Dense reconstruction or mesh generation.
- Automatic camera calibration beyond the selected COLMAP defaults.

## Traceability notes

- **Related product goal:** fast GLOMAP-first reconstruction and handoff to external tools.
- **Dependencies:** COLMAP 4.x for global mode; the repository also documents classic COLMAP compatibility.
- **Known risks:** no real COLMAP binary is exercised by the tests; the mapper implementation does not currently pass the documented global mapper flag despite the product documentation describing global mode.
