# Feature Blueprint: Media Ingest and Frame Preparation

## Feature Summary

The CLI delegates to [`extract_frames()`](../../../src/gsforge/ingest.py:667), which classifies the input, discovers or probes source frames, samples them, and writes project-local PNGs.

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Requirements:** [`../../requirements/features/REQ-MEDIA-INGEST.md`](../../requirements/features/REQ-MEDIA-INGEST.md)

## Component Blueprint Composition

Uses [`media-preparation.md`](../components/media-preparation.md) and [`project-state.md`](../components/project-state.md). Video processing calls FFmpeg; sequence processing calls Pillow.

## Feature-Specific Components

```component
name: IngestCommand
container: gsforge CLI
responsibilities:
  - Validate input path and expose count/scale options
  - Report extraction summary
```

## Models and contracts

```model
name: IngestResult
store: in-memory command result plus project.json
description: Number of frames, effective FPS, resolution, and output directory
fields:
  - num_frames: actual written count
  - effective_fps: video-derived or sequence-assumed rate
  - resolution: output dimensions
constraints:
  - Output frames are frame_NNNNNN.png
```

## Requirement alignment

- REQ-MEDIA-001 → [`classify_input()`](../../../src/gsforge/ingest.py:116) and [`resolve_image_sequence()`](../../../src/gsforge/ingest.py:153).
- REQ-MEDIA-002 → [`select_frames_evenly()`](../../../src/gsforge/ingest.py:234) and exporters at [`ingest_image_sequence()`](../../../src/gsforge/ingest.py:520).
- REQ-MEDIA-003 → source copying and metadata update in [`extract_frames()`](../../../src/gsforge/ingest.py:797).

## Architecture Decision Records

### ADR-001: Normalize sequence outputs to PNG

**Context:** Downstream COLMAP and training expect a predictable image set while inputs include TIFF and EXR.

**Decision:** Copy original inputs for provenance and re-export selected frames as RGB PNG.

**Consequences:** Downstream behavior is consistent; some high-bit-depth source information is not retained in prepared frames.
