"""gsforge/train.py - 3D Gaussian Splatting training layer for gsforge.

Takes a completed SfM reconstruction (COLMAP sparse/0/) and extracted frames
(preprocess/) and trains a 3DGS model using the gsplat library.

Architecture: BaseTrainer ABC + GsplatTrainer implementation.
Future backends (Brush, Inria, nerfstudio) can be added by subclassing BaseTrainer.

Data flow:
  COLMAP sparse/0/ -> load_colmap_data() -> ColmapData
  ColmapData -> GsplatTrainer.train() -> models/final_scene.ply

VP design philosophy: clear errors, fast iteration, safe defaults, interoperability.
"""

from __future__ import annotations

import math
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from gsforge.utils import (
    console,
    ensure_dir,
    log_error,
    log_info,
    log_step,
    log_success,
    log_warning,
    make_progress,
    print_summary_table,
)

if TYPE_CHECKING:
    from gsforge.project import GSProject

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINAL_PLY_NAME: str = "final_scene.ply"
CHECKPOINT_TEMPLATE: str = "ckpt_{:06d}.pth"
PREVIEW_TEMPLATE: str = "preview_{:06d}.png"
CHECKPOINTS_SUBDIR: str = "checkpoints"

# Learning rates tuned for VP footage (well-constrained scenes, clean studio sets).
# Lower LR for means than Inria defaults prevents floaters on smooth backgrounds.
DEFAULT_LR_MEANS: float = 1.6e-4
DEFAULT_LR_OPACITIES: float = 5e-2
DEFAULT_LR_SCALES: float = 5e-3
DEFAULT_LR_QUATS: float = 1e-3
DEFAULT_LR_SH: float = 2.5e-3

# Densification schedule.
# Wait 500 iters for Gaussians to settle before splitting/cloning.
DENSIFY_FROM_ITER: int = 500
DENSIFY_EVERY: int = 100
DENSIFY_UNTIL_ITER: int = 15_000
DENSIFY_GRAD_THRESHOLD: float = 2e-4
OPACITY_PRUNE_THRESHOLD: float = 0.005
OPACITY_RESET_EVERY: int = 3_000


# ---------------------------------------------------------------------------
# COLMAP data structures
# ---------------------------------------------------------------------------


@dataclass
class ColmapCamera:
    """Single camera model from a COLMAP reconstruction.

    All camera models are normalised to (fx, fy, cx, cy) pixel-space intrinsics
    so that gsplat receives a consistent format regardless of the original model.
    """

    camera_id: int
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class ColmapImage:
    """Single registered image from a COLMAP reconstruction.

    Pose stored as world-to-camera quaternion + translation (COLMAP convention).
    Converted to camera-to-world 4x4 matrix before passing to gsplat.
    """

    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str


@dataclass
class ColmapPoint3D:
    """Single 3D point from a COLMAP sparse reconstruction."""

    point3d_id: int
    x: float
    y: float
    z: float
    r: int
    g: int
    b: int
    error: float


@dataclass
class ColmapData:
    """All data loaded from a COLMAP sparse/0/ directory."""

    cameras: dict[int, ColmapCamera]
    images: list[ColmapImage]
    points3d: list[ColmapPoint3D]

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_points(self) -> int:
        return len(self.points3d)


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Summary of a completed training run."""

    final_ply: Path
    iterations: int
    device: str
    duration_seconds: float


# ---------------------------------------------------------------------------
# Base trainer abstraction
# ---------------------------------------------------------------------------


class BaseTrainer(ABC):
    """Abstract base class for all gsforge training backends.

    Subclass this to add new backends (Brush, Inria, nerfstudio).
    The CLI and project management code never need to know which backend is active.

    Parameters
    ----------
    project : GSProject
        The loaded project instance. All paths are derived from this.
    iterations : int
        Total training iterations.
    preview_every : int
        Save a preview render every N iterations. VP users need visual
        feedback early to catch bad reconstructions before wasting GPU time.
    """

    def __init__(
        self,
        project: "GSProject",
        iterations: int,
        preview_every: int,
    ) -> None:
        self.project = project
        self.iterations = iterations
        self.preview_every = preview_every

        self.sparse_dir: Path = project.sparse_dir
        self.image_dir: Path = project.preprocess_dir
        self.models_dir: Path = project.models_dir
        self.renders_dir: Path = project.renders_dir
        self.checkpoints_dir: Path = project.models_dir / CHECKPOINTS_SUBDIR

    @abstractmethod
    def train(self) -> TrainingResult:
        """Run training and return a summary.

        Implementations must validate inputs, load data, run the training loop
        with progress reporting, save checkpoints/previews, save final .ply,
        and return a TrainingResult. Raise exceptions on failure (do not call
        log_error directly) so run_training() can update project.json first.
        """
        ...

    def _validate_inputs(self) -> None:
        """Validate SfM output and images exist before training starts.

        Raises
        ------
        FileNotFoundError
            If sparse/0/ is missing or contains no COLMAP model files.
        ValueError
            If preprocess/ has no images.
        """
        if not self.sparse_dir.exists():
            raise FileNotFoundError(
                f"No SfM reconstruction found at: {self.sparse_dir}\n"
                "  Run 'gsforge sfm' or 'gsforge import-colmap' before training."
            )

        bin_files = {"cameras.bin", "images.bin", "points3D.bin"}
        txt_files = {"cameras.txt", "images.txt", "points3D.txt"}
        existing = {f.name for f in self.sparse_dir.iterdir() if f.is_file()}

        if not (bin_files & existing) and not (txt_files & existing):
            raise FileNotFoundError(
                f"SfM directory exists but contains no COLMAP model files: {self.sparse_dir}\n"
                "  Expected cameras.bin/txt, images.bin/txt, points3D.bin/txt.\n"
                "  Re-run 'gsforge sfm' to regenerate the reconstruction."
            )

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}\n"
                "  Run 'gsforge ingest' first."
            )

        images = list(self.image_dir.glob("frame_*.png"))
        if not images:
            raise ValueError(
                f"No images found in: {self.image_dir}\n"
                "  Run 'gsforge ingest' to extract frames first."
            )

        log_info(
            f"Validated inputs: {len(images)} images, sparse model at {self.sparse_dir}"
        )

    def _ensure_output_dirs(self) -> None:
        """Create all output directories if they do not exist."""
        ensure_dir(self.models_dir)
        ensure_dir(self.renders_dir)
        ensure_dir(self.checkpoints_dir)


# ---------------------------------------------------------------------------
# COLMAP binary format loaders
# ---------------------------------------------------------------------------


def _read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian: str = "<"):
    """Read and unpack binary data from a COLMAP binary file (little-endian)."""
    data = fid.read(num_bytes)
    return struct.unpack(endian + format_char_sequence, data)


def load_cameras_bin(cameras_bin_path: Path) -> dict[int, ColmapCamera]:
    """Load cameras from COLMAP binary cameras.bin.

    Binary format: uint64 count, then per-camera: uint32 id, int32 model_id,
    uint64 width, uint64 height, float64[N] params.
    All models normalised to (fx, fy, cx, cy) for gsplat compatibility.
    """
    CAMERA_MODELS = {
        0: ("SIMPLE_PINHOLE", 3),
        1: ("PINHOLE", 4),
        2: ("SIMPLE_RADIAL", 4),
        3: ("RADIAL", 5),
        4: ("OPENCV", 8),
        5: ("OPENCV_FISHEYE", 8),
        6: ("FULL_OPENCV", 12),
        7: ("FOV", 5),
        8: ("SIMPLE_RADIAL_FISHEYE", 4),
        9: ("RADIAL_FISHEYE", 5),
        10: ("THIN_PRISM_FISHEYE", 12),
    }

    cameras: dict[int, ColmapCamera] = {}

    with open(cameras_bin_path, "rb") as fid:
        (num_cameras,) = _read_next_bytes(fid, 8, "Q")

        for _ in range(num_cameras):
            (camera_id, model_id, width, height) = _read_next_bytes(fid, 24, "iiQQ")

            if model_id not in CAMERA_MODELS:
                raise ValueError(
                    f"Unsupported COLMAP camera model ID: {model_id}. "
                    "gsforge supports PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV."
                )

            model_name, num_params = CAMERA_MODELS[model_id]
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)

            if model_id == 0:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model_id == 1:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model_id in (2, 3):
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model_id in (4, 5, 6):
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                fx = fy = params[0]
                cx, cy = width / 2.0, height / 2.0
                log_warning(
                    f"Camera model {model_name} not fully supported - "
                    "using approximate intrinsics."
                )

            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model=model_name,
                width=int(width),
                height=int(height),
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
            )

    return cameras


def load_cameras_txt(cameras_txt_path: Path) -> dict[int, ColmapCamera]:
    """Load cameras from COLMAP text cameras.txt (fallback when .bin absent)."""
    cameras: dict[int, ColmapCamera] = {}

    with open(cameras_txt_path) as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id, model = int(parts[0]), parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(p) for p in parts[4:]]

            if model == "SIMPLE_PINHOLE":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == "PINHOLE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model in ("SIMPLE_RADIAL", "RADIAL"):
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                fx = fy = params[0] if params else float(width) / 2.0
                cx, cy = float(width) / 2.0, float(height) / 2.0
                log_warning(
                    f"Camera model {model} not fully supported - "
                    "using approximate intrinsics."
                )

            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model=model,
                width=width,
                height=height,
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
            )

    return cameras


def load_images_bin(images_bin_path: Path) -> list[ColmapImage]:
    """Load registered images from COLMAP binary images.bin.

    Binary format per image: uint32 id, float64[4] quat, float64[3] trans,
    uint32 camera_id, null-terminated name, uint64 num_points2D, point2D data.
    """
    images: list[ColmapImage] = []

    with open(images_bin_path, "rb") as fid:
        (num_images,) = _read_next_bytes(fid, 8, "Q")

        for _ in range(num_images):
            (image_id,) = _read_next_bytes(fid, 4, "I")
            (qw, qx, qy, qz) = _read_next_bytes(fid, 32, "dddd")
            (tx, ty, tz) = _read_next_bytes(fid, 24, "ddd")
            (camera_id,) = _read_next_bytes(fid, 4, "I")

            name_chars = []
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name_chars.append(char.decode("utf-8"))
            name = "".join(name_chars)

            (num_points2d,) = _read_next_bytes(fid, 8, "Q")
            fid.read(num_points2d * 24)  # skip point2D data

            images.append(
                ColmapImage(
                    image_id=image_id,
                    qw=qw,
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    tx=tx,
                    ty=ty,
                    tz=tz,
                    camera_id=camera_id,
                    name=name,
                )
            )

    return images


def load_images_txt(images_txt_path: Path) -> list[ColmapImage]:
    """Load registered images from COLMAP text images.txt (fallback).

    Text format: alternating lines of image header and point2D list.
    """
    images: list[ColmapImage] = []

    with open(images_txt_path) as fid:
        lines = [ln.strip() for ln in fid if ln.strip() and not ln.startswith("#")]

    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        images.append(
            ColmapImage(
                image_id=int(parts[0]),
                qw=float(parts[1]),
                qx=float(parts[2]),
                qy=float(parts[3]),
                qz=float(parts[4]),
                tx=float(parts[5]),
                ty=float(parts[6]),
                tz=float(parts[7]),
                camera_id=int(parts[8]),
                name=parts[9],
            )
        )

    return images


def load_points3d_bin(points3d_bin_path: Path) -> list[ColmapPoint3D]:
    """Load 3D points from COLMAP binary points3D.bin.

    Binary format per point: uint64 id, float64[3] xyz, uint8[3] rgb,
    float64 error, uint64 track_length, track data (skipped).
    """
    points: list[ColmapPoint3D] = []

    with open(points3d_bin_path, "rb") as fid:
        (num_points,) = _read_next_bytes(fid, 8, "Q")

        for _ in range(num_points):
            (point3d_id,) = _read_next_bytes(fid, 8, "Q")
            (x, y, z) = _read_next_bytes(fid, 24, "ddd")
            (r, g, b) = _read_next_bytes(fid, 3, "BBB")
            (error,) = _read_next_bytes(fid, 8, "d")
            (track_length,) = _read_next_bytes(fid, 8, "Q")
            fid.read(track_length * 8)

            points.append(
                ColmapPoint3D(
                    point3d_id=int(point3d_id),
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    r=int(r),
                    g=int(g),
                    b=int(b),
                    error=float(error),
                )
            )

    return points


def load_points3d_txt(points3d_txt_path: Path) -> list[ColmapPoint3D]:
    """Load 3D points from COLMAP text points3D.txt (fallback)."""
    points: list[ColmapPoint3D] = []

    with open(points3d_txt_path) as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            points.append(
                ColmapPoint3D(
                    point3d_id=int(parts[0]),
                    x=float(parts[1]),
                    y=float(parts[2]),
                    z=float(parts[3]),
                    r=int(parts[4]),
                    g=int(parts[5]),
                    b=int(parts[6]),
                    error=float(parts[7]),
                )
            )

    return points


def load_colmap_data(sparse_dir: Path) -> ColmapData:
    """Load a complete COLMAP sparse reconstruction from sparse/0/.

    Tries binary format first (faster), falls back to text format.

    We load this ourselves rather than using gsplat's ColmapDataset because
    gsplat assumes images live in a sibling 'images/' folder, but gsforge
    stores them in 'preprocess/'. Loading manually gives us full path control.

    Parameters
    ----------
    sparse_dir : Path
        Path to the COLMAP sparse/0/ directory.

    Returns
    -------
    ColmapData
        All cameras, images, and 3D points.
    """
    log_step("Loading COLMAP data", str(sparse_dir))

    cameras_bin = sparse_dir / "cameras.bin"
    cameras_txt = sparse_dir / "cameras.txt"
    if cameras_bin.exists():
        log_info("Loading cameras from cameras.bin")
        cameras = load_cameras_bin(cameras_bin)
    elif cameras_txt.exists():
        log_info("Loading cameras from cameras.txt")
        cameras = load_cameras_txt(cameras_txt)
    else:
        raise FileNotFoundError(f"No cameras.bin or cameras.txt found in: {sparse_dir}")

    images_bin = sparse_dir / "images.bin"
    images_txt = sparse_dir / "images.txt"
    if images_bin.exists():
        log_info("Loading images from images.bin")
        images = load_images_bin(images_bin)
    elif images_txt.exists():
        log_info("Loading images from images.txt")
        images = load_images_txt(images_txt)
    else:
        raise FileNotFoundError(f"No images.bin or images.txt found in: {sparse_dir}")

    points3d_bin = sparse_dir / "points3D.bin"
    points3d_txt = sparse_dir / "points3D.txt"
    if points3d_bin.exists():
        log_info("Loading 3D points from points3D.bin")
        points3d = load_points3d_bin(points3d_bin)
    elif points3d_txt.exists():
        log_info("Loading 3D points from points3D.txt")
        points3d = load_points3d_txt(points3d_txt)
    else:
        raise FileNotFoundError(
            f"No points3D.bin or points3D.txt found in: {sparse_dir}"
        )

    data = ColmapData(cameras=cameras, images=images, points3d=points3d)
    log_success(
        f"Loaded COLMAP data: {data.num_cameras} cameras, "
        f"{data.num_images} images, {data.num_points} 3D points"
    )
    return data


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def select_device() -> str:
    """Select the best available compute device for training.

    Returns "cuda" if a CUDA GPU is available, otherwise "cpu".
    Always prints which device is selected so VP users know immediately
    if they are accidentally training on CPU (hours vs. minutes on GPU).
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            log_success(f"Using GPU: {device_name} ({vram_gb:.1f} GB VRAM)")
            return "cuda"
        else:
            log_warning(
                "No CUDA GPU detected - training on CPU.\n"
                "  Training will be very slow (hours instead of minutes).\n"
                "  For VP workflows, a CUDA GPU with >=8 GB VRAM is strongly recommended."
            )
            return "cpu"
    except ImportError:
        log_warning(
            "PyTorch not found - cannot select device.\n"
            "  Install: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        return "cpu"


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> list[list[float]]:
    """Convert a unit quaternion to a 3x3 rotation matrix (row-major list).

    COLMAP quaternions are world-to-camera (w2c).
    gsplat expects camera-to-world (c2w), so callers must invert the result.
    """
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)
    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qx * qw)
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx * qx + qy * qy)
    return [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]


def _colmap_image_to_c2w(img: ColmapImage) -> list[list[float]]:
    """Convert a ColmapImage pose to a 4x4 camera-to-world matrix.

    COLMAP stores world-to-camera (w2c): p_cam = R * p_world + t
    gsplat expects camera-to-world (c2w): p_world = R_c2w * p_cam + t_c2w

    Inversion: R_c2w = R^T,  t_c2w = -R^T @ t
    """
    R = _quat_to_rotmat(img.qw, img.qx, img.qy, img.qz)
    tx, ty, tz = img.tx, img.ty, img.tz

    Rt = [[R[j][i] for j in range(3)] for i in range(3)]  # transpose

    tcx = -(Rt[0][0] * tx + Rt[0][1] * ty + Rt[0][2] * tz)
    tcy = -(Rt[1][0] * tx + Rt[1][1] * ty + Rt[1][2] * tz)
    tcz = -(Rt[2][0] * tx + Rt[2][1] * ty + Rt[2][2] * tz)

    return [
        [Rt[0][0], Rt[0][1], Rt[0][2], tcx],
        [Rt[1][0], Rt[1][1], Rt[1][2], tcy],
        [Rt[2][0], Rt[2][1], Rt[2][2], tcz],
        [0.0, 0.0, 0.0, 1.0],
    ]


# ---------------------------------------------------------------------------
# Preview render helper
# ---------------------------------------------------------------------------


def _save_preview_render(render_tensor, output_path: Path) -> None:
    """Save a rendered image tensor to disk as PNG.

    Accepts a (H, W, 3) float32 tensor in [0, 1] range.
    Uses PIL - no additional dependencies beyond what gsforge already requires.

    VP rationale: preview renders every N iterations let artists catch bad
    reconstructions early and abort before wasting the full training budget.
    """
    try:
        from PIL import Image
        import numpy as np

        img_np = (render_tensor.detach().cpu().clamp(0, 1) * 255).byte().numpy()
        Image.fromarray(img_np, mode="RGB").save(str(output_path))
        log_info(f"Preview saved: {output_path.name}")
    except Exception as exc:
        log_warning(f"Could not save preview render: {exc}")


# ---------------------------------------------------------------------------
# GsplatTrainer - STUB PLACEHOLDER (methods added below)
# ---------------------------------------------------------------------------


class GsplatTrainer(BaseTrainer):
    """3DGS trainer using the gsplat library.

    Implements the full training pipeline:
      1. Input validation
      2. COLMAP data loading
      3. Gaussian initialisation from sparse point cloud
      4. Optimisation loop with densification/pruning
      5. Periodic checkpointing and preview renders
      6. Final .ply export

    Densification strategy follows the Inria paper's adaptive density control:
      - Clone Gaussians with large positional gradients (under-reconstructed regions)
      - Split large Gaussians (over-reconstructed regions)
      - Prune transparent Gaussians (opacity < threshold)
      - Periodically reset opacities to prevent saturation
    """

    def train(self) -> TrainingResult:
        """Run the full gsplat training pipeline."""
        start_time = time.time()

        log_step("Starting Training", f"backend=gsplat  iterations={self.iterations}")
        self._validate_inputs()
        self._ensure_output_dirs()

        log_step("Importing dependencies")
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is not installed.\n"
                "  Install: pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )

        try:
            import gsplat
        except ImportError:
            raise ImportError(
                "gsplat is not installed.\n"
                "  Install PyTorch first, then: pip install gsplat\n"
                "  See: https://docs.gsplat.studio/main/installation/installation.html"
            )

        device = select_device()
        torch_device = torch.device(device)

        log_step("Loading Data")
        colmap_data = load_colmap_data(self.sparse_dir)

        if colmap_data.num_points == 0:
            raise ValueError(
                "COLMAP reconstruction has no 3D points.\n"
                "  This usually means SfM failed to triangulate points.\n"
                "  Re-run 'gsforge sfm' or check that images have sufficient overlap."
            )

        if colmap_data.num_images == 0:
            raise ValueError(
                "COLMAP reconstruction has no registered images.\n"
                "  Re-run 'gsforge sfm' to regenerate the reconstruction."
            )

        log_step("Preparing tensors")

        # Initial Gaussian means from sparse point cloud - shape (N, 3) float32.
        # Initialising at COLMAP points gives a much better starting point than
        # random init: the sparse cloud captures rough scene geometry so Gaussians
        # only need to refine position/scale/colour rather than discover structure.
        points_xyz = torch.tensor(
            [[p.x, p.y, p.z] for p in colmap_data.points3d],
            dtype=torch.float32,
            device=torch_device,
        )

        # Per-point colours from COLMAP - shape (N, 3) float32 in [0, 1].
        # Seeds the spherical harmonic DC component (degree-0 colour).
        point_colours = torch.tensor(
            [[p.r / 255.0, p.g / 255.0, p.b / 255.0] for p in colmap_data.points3d],
            dtype=torch.float32,
            device=torch_device,
        )

        log_info(
            f"Initialising {points_xyz.shape[0]} Gaussians from sparse point cloud"
        )
        log_step("Training", f"{self.iterations} iterations on {device.upper()}")

        try:
            result = self._train_with_gsplat(
                torch_device=torch_device,
                points_xyz=points_xyz,
                point_colours=point_colours,
                colmap_data=colmap_data,
                gsplat_module=gsplat,
                torch_module=torch,
            )
        except RuntimeError as exc:
            exc_str = str(exc)
            if "out of memory" in exc_str.lower():
                raise RuntimeError(
                    "CUDA ran out of memory during training.\n"
                    "  Suggestions:\n"
                    "    - Reduce --iterations (e.g. 7500 instead of 15000)\n"
                    "    - Use a GPU with more VRAM (>=8 GB recommended)\n"
                    "    - Close other GPU-intensive applications\n"
                    f"  Original error: {exc_str}"
                ) from exc
            raise

        duration = time.time() - start_time
        return TrainingResult(
            final_ply=result,
            iterations=self.iterations,
            device=device,
            duration_seconds=duration,
        )

    def _train_with_gsplat(
        self,
        torch_device,
        points_xyz,
        point_colours,
        colmap_data: ColmapData,
        gsplat_module,
        torch_module,
    ) -> Path:
        """Core gsplat training loop. Returns path to saved final .ply.

        Training loop:
          1. Build camera tensors from COLMAP data.
          2. Initialise Gaussian parameters (means, scales, quats, opacities, SH).
          3. For each iteration: sample camera, rasterise, compute loss, backprop.
          4. Densify/prune at scheduled intervals.
          5. Save checkpoint + preview render at preview_every intervals.
          6. Save final .ply and return its path.
        """
        import random

        torch = torch_module
        gsplat = gsplat_module

        sorted_images = sorted(colmap_data.images, key=lambda im: im.name)
        cam_Ks: list = []
        cam_c2ws: list = []
        cam_widths: list = []
        cam_heights: list = []
        cam_image_paths: list = []

        for img in sorted_images:
            cam = colmap_data.cameras.get(img.camera_id)
            if cam is None:
                log_warning(
                    f"Image {img.name} references unknown camera {img.camera_id} - skipping."
                )
                continue

            K = torch.tensor(
                [[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=torch_device,
            )
            cam_Ks.append(K)
            c2w = torch.tensor(
                _colmap_image_to_c2w(img),
                dtype=torch.float32,
                device=torch_device,
            )
            cam_c2ws.append(c2w)
            cam_widths.append(cam.width)
            cam_heights.append(cam.height)
            cam_image_paths.append(self.image_dir / Path(img.name).name)

        if not cam_Ks:
            raise ValueError(
                "No valid cameras found after filtering. "
                "Check that camera IDs in images.bin match cameras.bin."
            )

        num_cams = len(cam_Ks)
        train_W = cam_widths[0]
        train_H = cam_heights[0]
        log_info(f"Built {num_cams} camera tensors ({train_W}x{train_H})")

        num_points = points_xyz.shape[0]

        # Scales in log-space: optimiser can freely explore without hitting
        # non-negativity constraint. Initial ~0.01 world units = small isotropic blobs.
        log_scales = torch.full(
            (num_points, 3),
            fill_value=math.log(0.01),
            dtype=torch.float32,
            device=torch_device,
        )

        # Identity quaternions (w=1, xyz=0)
        quats = torch.zeros(num_points, 4, dtype=torch.float32, device=torch_device)
        quats[:, 0] = 1.0

        # Opacities in logit-space. logit(0.1) ~ -2.2: start semi-transparent
        # so the scene builds up gradually rather than starting with opaque blobs.
        opacities_logit = torch.full(
            (num_points,),
            fill_value=-2.2,
            dtype=torch.float32,
            device=torch_device,
        )

        # Spherical harmonics: degree 0 (DC) seeded from COLMAP point colours.
        # Higher-degree SH added progressively to model view-dependent effects
        # (specular highlights on LED walls, etc.)
        sh_degree = 3
        num_sh_coeffs = (sh_degree + 1) ** 2  # 16 for degree 3
        C0 = 0.28209479177387814  # SH DC normalisation constant
        sh_coeffs = torch.zeros(
            num_points,
            num_sh_coeffs,
            3,
            dtype=torch.float32,
            device=torch_device,
        )
        sh_coeffs[:, 0, :] = (point_colours - 0.5) / C0

        means = points_xyz.clone().requires_grad_(True)
        log_scales = log_scales.requires_grad_(True)
        quats = quats.requires_grad_(True)
        opacities_logit = opacities_logit.requires_grad_(True)
        sh_coeffs = sh_coeffs.requires_grad_(True)

        optimizer = torch.optim.Adam(
            [
                {"params": [means], "lr": DEFAULT_LR_MEANS, "name": "means"},
                {"params": [log_scales], "lr": DEFAULT_LR_SCALES, "name": "scales"},
                {"params": [quats], "lr": DEFAULT_LR_QUATS, "name": "quats"},
                {
                    "params": [opacities_logit],
                    "lr": DEFAULT_LR_OPACITIES,
                    "name": "opacities",
                },
                {"params": [sh_coeffs], "lr": DEFAULT_LR_SH, "name": "sh"},
            ],
            eps=1e-15,
        )

        grad_accum = torch.zeros(num_points, device=torch_device)
        grad_count = torch.zeros(num_points, device=torch_device, dtype=torch.int32)

        log_info("Loading training images...")
        train_images: list = []
        missing_images: list = []

        for img_path in cam_image_paths:
            if not img_path.exists():
                missing_images.append(img_path.name)
                train_images.append(None)
                continue
            try:
                from PIL import Image as PILImage
                import numpy as np

                pil_img = PILImage.open(img_path).convert("RGB")
                if pil_img.width != train_W or pil_img.height != train_H:
                    pil_img = pil_img.resize((train_W, train_H), PILImage.LANCZOS)
                img_np = np.array(pil_img, dtype=np.float32) / 255.0
                train_images.append(torch.from_numpy(img_np))
            except Exception as exc:
                log_warning(f"Could not load image {img_path.name}: {exc}")
                train_images.append(None)

        valid_indices = [i for i, img in enumerate(train_images) if img is not None]
        if not valid_indices:
            raise FileNotFoundError(
                "No training images could be loaded.\n"
                "  Check that image filenames in images.bin match files in preprocess/.\n"
                f"  Missing examples: {missing_images[:5]}"
            )

        if missing_images:
            log_warning(
                f"{len(missing_images)} images referenced in COLMAP could not be found "
                f"in {self.image_dir}. Training will use {len(valid_indices)} images."
            )

        valid_Ks = [cam_Ks[i] for i in valid_indices]
        valid_c2ws = [cam_c2ws[i] for i in valid_indices]
        valid_images = [train_images[i] for i in valid_indices]
        num_train = len(valid_indices)
        log_success(f"Loaded {num_train} training images ({train_W}x{train_H})")

        final_ply_path = self.models_dir / FINAL_PLY_NAME
        active_sh_degree = 0

        pbar = make_progress(
            range(1, self.iterations + 1),
            desc="Training",
            total=self.iterations,
            unit="iter",
        )

        for iteration in pbar:
            active_sh_degree = min(sh_degree, iteration // 1000)

            cam_idx = random.randint(0, num_train - 1)
            K = valid_Ks[cam_idx]
            c2w = valid_c2ws[cam_idx]
            gt_image = valid_images[cam_idx].to(torch_device)

            w2c = torch.inverse(c2w)
            scales = torch.exp(log_scales)
            quats_norm = torch.nn.functional.normalize(quats, dim=-1)
            opacities = torch.sigmoid(opacities_logit)
            num_active = (active_sh_degree + 1) ** 2
            active_sh = sh_coeffs[:, :num_active, :]

            try:
                render_colors, render_alphas, info = gsplat.rasterization(
                    means=means,
                    quats=quats_norm,
                    scales=scales,
                    opacities=opacities,
                    colors=active_sh,
                    viewmats=w2c.unsqueeze(0),
                    Ks=K.unsqueeze(0),
                    width=train_W,
                    height=train_H,
                    sh_degree=active_sh_degree,
                    near_plane=0.01,
                    far_plane=1000.0,
                    render_mode="RGB",
                )
            except TypeError:
                render_colors, render_alphas, info = gsplat.rasterization(
                    means=means,
                    quats=quats_norm,
                    scales=scales,
                    opacities=opacities,
                    colors=active_sh,
                    viewmats=w2c.unsqueeze(0),
                    Ks=K.unsqueeze(0),
                    width=train_W,
                    height=train_H,
                    sh_degree=active_sh_degree,
                )

            rendered = render_colors[0]  # (H, W, 3)
            loss = torch.abs(rendered - gt_image).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if DENSIFY_FROM_ITER <= iteration <= DENSIFY_UNTIL_ITER:
                if info.get("means2d") is not None and info["means2d"].grad is not None:
                    grad_mag = info["means2d"].grad[0].norm(dim=-1)
                    grad_accum += grad_mag.detach()
                    grad_count += 1

            optimizer.step()

            if (
                DENSIFY_FROM_ITER <= iteration <= DENSIFY_UNTIL_ITER
                and iteration % DENSIFY_EVERY == 0
            ):
                (
                    means,
                    log_scales,
                    quats,
                    opacities_logit,
                    sh_coeffs,
                    grad_accum,
                    grad_count,
                    optimizer,
                ) = self._densify_and_prune(
                    means=means,
                    log_scales=log_scales,
                    quats=quats,
                    opacities_logit=opacities_logit,
                    sh_coeffs=sh_coeffs,
                    grad_accum=grad_accum,
                    grad_count=grad_count,
                    optimizer=optimizer,
                    torch_device=torch_device,
                    torch=torch,
                )

            if iteration % OPACITY_RESET_EVERY == 0:
                with torch.no_grad():
                    opacities_logit.fill_(-4.6)
                log_info(f"[iter {iteration}] Opacity reset")

            if iteration % self.preview_every == 0 or iteration == self.iterations:
                ckpt_path = self.checkpoints_dir / CHECKPOINT_TEMPLATE.format(iteration)
                torch.save(
                    {
                        "iteration": iteration,
                        "means": means.detach().cpu(),
                        "log_scales": log_scales.detach().cpu(),
                        "quats": quats.detach().cpu(),
                        "opacities_logit": opacities_logit.detach().cpu(),
                        "sh_coeffs": sh_coeffs.detach().cpu(),
                        "loss": loss.item(),
                    },
                    str(ckpt_path),
                )

                preview_path = self.renders_dir / PREVIEW_TEMPLATE.format(iteration)
                with torch.no_grad():
                    _save_preview_render(rendered, preview_path)

                log_info(
                    f"[iter {iteration}/{self.iterations}] "
                    f"loss={loss.item():.4f}  "
                    f"gaussians={means.shape[0]:,}  "
                    f"ckpt={ckpt_path.name}"
                )

        log_step("Saving Outputs")
        self._save_ply(
            path=final_ply_path,
            means=means,
            log_scales=log_scales,
            quats=quats,
            opacities_logit=opacities_logit,
            sh_coeffs=sh_coeffs,
            torch=torch,
        )
        log_success(f"Final model saved: {final_ply_path}")
        return final_ply_path

    def _densify_and_prune(
        self,
        means,
        log_scales,
        quats,
        opacities_logit,
        sh_coeffs,
        grad_accum,
        grad_count,
        optimizer,
        torch_device,
        torch,
    ):
        """Adaptive density control: clone, split, and prune Gaussians.

        Follows the Inria 3DGS paper's adaptive density control algorithm:
          - Clone: duplicate Gaussians with high positional gradients and small scale
            (under-reconstructed fine details — e.g. actor's face on LED stage)
          - Split: replace large Gaussians with two smaller ones at high-gradient regions
            (over-reconstructed coarse areas — e.g. large background panels)
          - Prune: remove nearly-transparent Gaussians (opacity < threshold)

        Returns updated parameter tensors and a fresh optimizer with the new N.
        """
        with torch.no_grad():
            avg_grad = torch.where(
                grad_count > 0,
                grad_accum / grad_count.float(),
                torch.zeros_like(grad_accum),
            )

            densify_mask = avg_grad >= DENSIFY_GRAD_THRESHOLD
            max_scale = torch.exp(log_scales).max(dim=-1).values
            scene_extent = (
                (means.max(dim=0).values - means.min(dim=0).values).max().item()
            )
            split_threshold = 0.01 * scene_extent
            split_mask = densify_mask & (max_scale > split_threshold)
            clone_mask = densify_mask & (max_scale <= split_threshold)

            # Clone: duplicate under-reconstructed small Gaussians
            if clone_mask.any():
                means = torch.cat([means, means[clone_mask].detach()], dim=0)
                log_scales = torch.cat(
                    [log_scales, log_scales[clone_mask].detach()], dim=0
                )
                quats = torch.cat([quats, quats[clone_mask].detach()], dim=0)
                opacities_logit = torch.cat(
                    [opacities_logit, opacities_logit[clone_mask].detach()], dim=0
                )
                sh_coeffs = torch.cat(
                    [sh_coeffs, sh_coeffs[clone_mask].detach()], dim=0
                )

            # Split: replace large Gaussians with two smaller ones
            if split_mask.any():
                n_split = split_mask.sum().item()
                stds = torch.exp(log_scales[split_mask])
                samples = torch.randn(n_split, 3, device=torch_device) * stds
                new_means_a = means[split_mask] + samples
                new_means_b = means[split_mask] - samples
                new_log_scales = log_scales[split_mask] - math.log(1.6)
                keep = ~split_mask
                means = torch.cat([means[keep], new_means_a, new_means_b], dim=0)
                log_scales = torch.cat(
                    [log_scales[keep], new_log_scales, new_log_scales], dim=0
                )
                quats = torch.cat(
                    [quats[keep], quats[split_mask], quats[split_mask]], dim=0
                )
                opacities_logit = torch.cat(
                    [
                        opacities_logit[keep],
                        opacities_logit[split_mask],
                        opacities_logit[split_mask],
                    ],
                    dim=0,
                )
                sh_coeffs = torch.cat(
                    [sh_coeffs[keep], sh_coeffs[split_mask], sh_coeffs[split_mask]],
                    dim=0,
                )

            # Prune: remove nearly-transparent Gaussians
            opacities = torch.sigmoid(opacities_logit)
            prune_mask = opacities < OPACITY_PRUNE_THRESHOLD
            if prune_mask.any():
                keep = ~prune_mask
                means = means[keep]
                log_scales = log_scales[keep]
                quats = quats[keep]
                opacities_logit = opacities_logit[keep]
                sh_coeffs = sh_coeffs[keep]

            new_n = means.shape[0]
            grad_accum = torch.zeros(new_n, device=torch_device)
            grad_count = torch.zeros(new_n, device=torch_device, dtype=torch.int32)

        # Re-attach requires_grad and rebuild optimizer with new tensors
        means = means.detach().requires_grad_(True)
        log_scales = log_scales.detach().requires_grad_(True)
        quats = quats.detach().requires_grad_(True)
        opacities_logit = opacities_logit.detach().requires_grad_(True)
        sh_coeffs = sh_coeffs.detach().requires_grad_(True)

        optimizer = torch.optim.Adam(
            [
                {"params": [means], "lr": DEFAULT_LR_MEANS, "name": "means"},
                {"params": [log_scales], "lr": DEFAULT_LR_SCALES, "name": "scales"},
                {"params": [quats], "lr": DEFAULT_LR_QUATS, "name": "quats"},
                {
                    "params": [opacities_logit],
                    "lr": DEFAULT_LR_OPACITIES,
                    "name": "opacities",
                },
                {"params": [sh_coeffs], "lr": DEFAULT_LR_SH, "name": "sh"},
            ],
            eps=1e-15,
        )

        return (
            means,
            log_scales,
            quats,
            opacities_logit,
            sh_coeffs,
            grad_accum,
            grad_count,
            optimizer,
        )

    def _save_ply(
        self,
        path: Path,
        means,
        log_scales,
        quats,
        opacities_logit,
        sh_coeffs,
        torch,
    ) -> None:
        """Save trained Gaussians as a standard .ply file.

        The .ply format is the interchange format for 3DGS models. It is
        readable by SuperSplat, KIRI Engine viewer, Luma AI viewer, and any
        custom gsplat/3DGS renderer.

        The property layout matches the Inria 3DGS reference implementation
        so that existing viewers and converters work without modification.

        Parameters
        ----------
        path : Path
            Output .ply file path.
        means, log_scales, quats, opacities_logit, sh_coeffs : torch.Tensor
            Trained Gaussian parameters.
        torch : module
            The torch module (passed in to avoid re-importing at module level).
        """
        import numpy as np

        with torch.no_grad():
            xyz = means.detach().cpu().numpy().astype(np.float32)
            scales = log_scales.detach().cpu().numpy().astype(np.float32)
            rots = quats.detach().cpu().numpy().astype(np.float32)
            opacities = opacities_logit.detach().cpu().numpy().astype(np.float32)
            sh = sh_coeffs.detach().cpu().numpy().astype(np.float32)

        n = xyz.shape[0]
        num_sh = sh.shape[1]

        f_dc = sh[:, 0, :]  # (N, 3) DC SH coefficients
        f_rest = (
            sh[:, 1:, :].reshape(n, -1)
            if num_sh > 1
            else np.zeros((n, 0), dtype=np.float32)
        )
        num_rest = f_rest.shape[1]

        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
        ]
        for i in range(num_rest):
            header_lines.append(f"property float f_rest_{i}")
        header_lines += [
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
        ]
        header = "\n".join(header_lines) + "\n"

        normals = np.zeros((n, 3), dtype=np.float32)

        columns = [xyz, normals, f_dc]
        if num_rest > 0:
            columns.append(f_rest)
        columns += [
            opacities.reshape(n, 1),
            scales,
            rots,
        ]

        vertex_data = np.concatenate(columns, axis=1)

        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(vertex_data.tobytes())

        log_info(
            f"Saved {n:,} Gaussians to {path.name} ({path.stat().st_size / 1e6:.1f} MB)"
        )


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[BaseTrainer]] = {
    "gsplat": GsplatTrainer,
}


def get_trainer(backend: str) -> type[BaseTrainer]:
    """Return the trainer class for the given backend name.

    Parameters
    ----------
    backend : str
        Backend identifier (e.g. "gsplat").

    Returns
    -------
    type[BaseTrainer]
        The trainer class (not an instance).

    Raises
    ------
    ValueError
        If the backend is not registered.
    """
    if backend not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise ValueError(
            f"Unknown training backend: {backend!r}.\n"
            f"  Available backends: {available}\n"
            "  Brush and Inria backends are planned for a future release."
        )
    return _BACKENDS[backend]


# ---------------------------------------------------------------------------
# Public API: run_training
# ---------------------------------------------------------------------------


def run_training(
    project_path: Path,
    backend: str = "gsplat",
    iterations: int = 15_000,
    preview_every: int = 500,
) -> None:
    """Entry point for the gsforge train CLI command.

    Loads the project, runs training with the specified backend, updates
    project.json, and prints a summary table.

    Parameters
    ----------
    project_path : Path
        Path to the .gsproject directory.
    backend : str
        Training backend identifier. Currently only "gsplat" is supported.
    iterations : int
        Total number of training iterations.
    preview_every : int
        Save a preview render every N iterations.
    """
    from gsforge.project import GSProject

    proj = GSProject.from_path(project_path)

    TrainerClass = get_trainer(backend)
    trainer = TrainerClass(
        project=proj,
        iterations=iterations,
        preview_every=preview_every,
    )

    try:
        result = trainer.train()
    except (FileNotFoundError, ValueError, ImportError) as exc:
        # User-facing errors: print clearly and mark training as failed
        log_error(str(exc))
        proj.update_after_training(training_status="failed")
        return
    except Exception as exc:
        # Unexpected errors: still update project.json before re-raising
        proj.update_after_training(training_status="failed")
        log_error(
            f"Training failed with an unexpected error: {exc}\n"
            "  Check the output above for details."
        )
        return

    # Compute relative path for project.json (portable across machines)
    try:
        final_ply_rel = str(result.final_ply.relative_to(proj.root))
    except ValueError:
        final_ply_rel = str(result.final_ply)

    proj.update_after_training(
        training_status="completed",
        final_ply=final_ply_rel,
        last_iteration=result.iterations,
    )

    hours, rem = divmod(int(result.duration_seconds), 3600)
    mins, secs = divmod(rem, 60)
    duration_str = (
        f"{hours}h {mins}m {secs}s"
        if hours
        else f"{mins}m {secs}s" if mins else f"{secs}s"
    )

    print_summary_table(
        title="[bold cyan]Training complete[/bold cyan]",
        rows=[
            ("Iterations", str(result.iterations)),
            ("Final output", final_ply_rel),
            ("Device", result.device.upper()),
            ("Duration", duration_str),
            ("Checkpoints", str(proj.models_dir / CHECKPOINTS_SUBDIR)),
            ("Previews", str(proj.renders_dir)),
        ],
    )
    log_success(
        "Done. Open the .ply in a 3DGS viewer:\n"
        "  SuperSplat: https://supersplat.dev\n"
        "  Or run: gsforge info"
    )
