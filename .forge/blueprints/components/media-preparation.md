# Component Blueprint: Media Preparation

## Capability Summary

Classifies inputs, discovers numbered sequences, samples frames evenly, copies source media, and writes normalized PNG frames.

## Document status

- **Status:** `approved-baseline`
- **Owner:** gsforge maintainers
- **Last updated:** 2026-08-19
- **Containers:** [`gsforge-cli.md`](../containers/gsforge-cli.md)

## Core Components

```component
name: InputClassifier
container: gsforge CLI
responsibilities:
  - Classify supported video and image extensions
  - Reject unsupported inputs
```

```component
name: FrameSelector
container: gsforge CLI
responsibilities:
  - Discover numeric siblings
  - Select evenly distributed source indices
```

```component
name: FrameExporter
container: gsforge CLI
responsibilities:
  - Use FFmpeg for video extraction
  - Use Pillow for sequence conversion and scaling
  - Update project metadata
```

The components are implemented in [`ingest.py`](../../../src/gsforge/ingest.py:116), [`select_frames_evenly()`](../../../src/gsforge/ingest.py:234), and [`extract_frames()`](../../../src/gsforge/ingest.py:667). Outputs flow to `source/` and `preprocess/`.

## System Contracts

### Key Contracts

- Output names are `frame_NNNNNN.png`.
- Sequence input requires at least two matching frames.
- A request above availability emits a warning and uses all available frames.

### Integration Contracts

- FFmpeg receives a generated select filter for videos.
- Pillow reads supported sequence formats and writes RGB PNG.

## Architecture Decision Records

### ADR-001: Uniform sampling across the full source

**Context:** VP footage can be long and dense; downstream reconstruction needs broad temporal coverage.

**Decision:** Select indices using the shared even-sampling function rather than prefix-only slicing.

**Consequences:** The frame budget is bounded and representative; exact VFR frame counts remain approximate because video totals derive from duration and FPS.
