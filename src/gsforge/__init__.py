"""
gsforge — CLI-first tool for virtual production 3D Gaussian Splatting workflows.

Package layout
--------------
gsforge/project.py  — GSProject: smart folder, metadata, status tracking
gsforge/ingest.py   — Frame extraction: MP4/MOV via FFmpeg, image sequences via Pillow
gsforge/sfm.py      — GLOMAP/COLMAP runner + import/export helpers
gsforge/train.py    — gsplat training wrapper (added in next iteration)
gsforge/utils.py    — logging, progress, path helpers, shared constants
"""

__version__ = "0.1.0"
