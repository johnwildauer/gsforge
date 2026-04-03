"""
gsforge — CLI-first tool for virtual production 3D Gaussian Splatting workflows.

Package layout
--------------
src/project.py  — GSProject: smart folder, metadata, status tracking
src/ingest.py   — FFmpeg video → frames with VP-tuned downsampling
src/sfm.py      — GLOMAP/COLMAP runner + import/export helpers
src/train.py    — gsplat training wrapper (added in next iteration)
src/utils.py    — logging, progress, path helpers, shared constants
"""

__version__ = "0.1.0"
