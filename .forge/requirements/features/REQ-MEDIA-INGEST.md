# Feature Requirements: Media Ingest and Frame Preparation

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Evidence basis:** observed repository behavior and README product intent
- **Product overview:** [`../product-overview.md`](../product-overview.md)
- **Blueprint counterpart:** [`../../blueprints/features/media-ingest.md`](../../blueprints/features/media-ingest.md)

## Overview

Users provide video or the first frame of a numbered image sequence. The product prepares a bounded, uniformly sampled PNG frame set while retaining source media inside the project.

## Terminology

| Term           | Definition                                                          |
| -------------- | ------------------------------------------------------------------- |
| Video input    | A `.mp4` or `.mov` file probed and sampled through FFmpeg.          |
| Image sequence | Same-directory frames sharing a prefix and trailing numeric suffix. |
| Downscale      | An integer spatial reduction applied while writing prepared frames. |

## Requirements

### REQ-MEDIA-001: Accept supported media inputs

**User Story:** As an artist, I want to ingest footage or an image sequence, so that common VP source formats enter the same reconstruction workflow.

**Acceptance Criteria:**

- **AC-MEDIA-001.1:** When the input is `.mp4` or `.mov`, the system shall treat it as video.
- **AC-MEDIA-001.2:** When the input is `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, or `.exr`, the system shall discover matching numbered siblings and require at least two frames.
- **AC-MEDIA-001.3:** When the input extension is unsupported or a sequence cannot be identified, the system shall stop with an actionable error.

### REQ-MEDIA-002: Sample and prepare frames

**User Story:** As an artist, I want to control the prepared frame count and resolution, so that reconstruction cost fits my footage and hardware.

**Acceptance Criteria:**

- **AC-MEDIA-002.1:** The system shall distribute selected frames across the full source sequence and avoid duplicate output indices.
- **AC-MEDIA-002.2:** When fewer source frames exist than requested, the system shall use all available frames and warn the user.
- **AC-MEDIA-002.3:** The system shall write consistently numbered PNG frames and record the requested count, scale, and actual count.

### REQ-MEDIA-003: Preserve source provenance

**User Story:** As a user, I want source media copied into the project, so that the project remains self-contained.

**Acceptance Criteria:**

- **AC-MEDIA-003.1:** After successful ingest, the original video or selected sequence frames shall be present under the project source area.
- **AC-MEDIA-003.2:** The system shall record a project-relative source path.

## Non-functional requirements

- **NFR-MEDIA-001:** The ingest path shall provide actionable errors when FFmpeg or readable media is unavailable.
- **NFR-MEDIA-002:** The preparation operation shall not require a network service.

## Out of scope

- Blur, exposure, or quality filtering.
- Audio processing.
- Camera metadata calibration beyond what downstream COLMAP receives.

## Traceability notes

- **Related product goal:** fast, reproducible preparation from raw VP footage.
- **Dependencies:** FFmpeg/ffprobe for video and Pillow for image sequences.
- **Known risks:** video tests mock FFmpeg; real FFmpeg behavior remains a manual gate.
