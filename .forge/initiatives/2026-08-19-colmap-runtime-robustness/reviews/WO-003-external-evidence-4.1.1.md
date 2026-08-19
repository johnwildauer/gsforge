# WO-003 External Evidence — COLMAP 4.1.1 Windows CUDA Binary

## Environment

- **Binary:** `bin/colmap-x64-windows-cuda/bin/colmap.exe`
- **Reported build:** COLMAP 4.1.1, commit `a0d785f`, 2026-07-17, with CUDA
- **GPU:** NVIDIA RTX 4090
- **Driver:** NVIDIA Studio Driver 610.88
- **CUDA:** Toolkit/runtime 12.4

## Capability probe

Command: `colmap.exe help`

- **Exit status:** `0`
- **Observed:** top-level help exposes `version`, `global_mapper`, `mapper`, feature extraction, and matching commands.

Command: `colmap.exe global_mapper -h`

- **Exit status:** `0`
- **Observed options:** `GlobalMapper.gp_use_gpu=1`, `GlobalMapper.gp_gpu_index=-1`, `GlobalMapper.ba_ceres_use_gpu=1`, and `GlobalMapper.ba_ceres_gpu_index=-1`.
- **Observed absence:** no Caspar backend selector is exposed by `global_mapper`.

The implementation's live probe independently reproduced supported states for `help`, `version`, `-h`, `feature_extractor`, `exhaustive_matcher`, `sequential_matcher`, `mapper`, and `global_mapper`.

## Runtime evidence

The same 185-frame GLOMAP workflow is captured in [`garden-glomap-sfm.txt`](../../../garden-glomap-sfm.txt). The run completed with 185 registered cameras and selected `sfm/sparse/0`. The mapper reported the known official-binary Ceres limitation and continued with CPU bundle adjustment. This is accepted behavior under the stakeholder-approved standard-binary contract; custom GPU Ceres/Caspar builds remain unsupported.

## Scope

This artifact is evidence for the official-binary support boundary. It makes no claim that a custom COLMAP/Ceres/Caspar build is compatible with gsforge or that GLOMAP bundle adjustment is GPU-accelerated in the distributed binary.
