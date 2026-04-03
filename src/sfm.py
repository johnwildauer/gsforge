"""
src/sfm.py — Structure-from-Motion runner for gsforge.

Why GLOMAP by default?
----------------------
GLOMAP is a global SfM method shipped as part of COLMAP 4.x.  Compared to
classic incremental COLMAP it is:

  - 5–20x faster on typical VP footage (smooth camera paths, good overlap).
  - More robust: global methods don't suffer from drift accumulation that
    plagues incremental SfM on long sequences.
  - Equally accurate for well-constrained scenes (which VP footage always is).

GLOMAP is invoked via the same ``colmap`` binary using the
``mapper --MapperType GLOBAL`` flag introduced in COLMAP 4.0.
Users only need to install COLMAP 4.x — no separate GLOMAP binary.

Classic incremental COLMAP is still available via ``--method colmap`` for
difficult footage (e.g. very wide baselines, textureless surfaces).

Binary discovery
----------------
We look for the COLMAP binary in this order:
  1. ``./bin/colmap`` or ``./bin/colmap.exe``  (project-local install)
  2. ``colmap`` on the system PATH

This lets users drop a portable COLMAP build into their project without
modifying system PATH — useful in locked-down studio environments.

COLMAP pipeline (both methods)
-------------------------------
  1. feature_extractor  — SIFT features on every frame
  2. exhaustive_matcher (or sequential_matcher for large sets) — match pairs
  3. mapper             — reconstruct cameras + sparse point cloud
     - GLOMAP: --MapperType GLOBAL
     - COLMAP: --MapperType INCREMENTAL (default)

Output
------
All SfM output goes into ``project/sfm/``.  The sparse model is written to
``project/sfm/sparse/0/`` in standard COLMAP binary format:
  - cameras.bin
  - images.bin
  - points3D.bin

This is the format expected by gsplat, nerfstudio, LichtFeld Studio, and
the COLMAP GUI.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from src.utils import (
    console,
    ensure_dir,
    log_error,
    log_info,
    log_step,
    log_success,
    log_warning,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SfmMethodLiteral = Literal["glomap", "colmap"]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SfmResult:
    """Summary of a completed SfM run."""

    status: str  # "completed" or "failed"
    camera_count: int  # number of cameras successfully registered
    sparse_dir: Path  # absolute path to sparse/0/


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


def find_colmap_binary() -> Path:
    """Locate the COLMAP 4.x binary.

    Search order:
      1. ``./bin/colmap.exe`` (Windows, project-local)
      2. ``./bin/colmap``     (Linux/macOS, project-local)
      3. ``colmap`` on system PATH

    Returns
    -------
    Path
        Absolute path to the colmap binary.

    Raises
    ------
    SystemExit
        If no COLMAP binary is found anywhere.
    """
    # Project-local bin/ directory (portable studio installs)
    local_bin = Path("bin")
    for candidate in [local_bin / "colmap.exe", local_bin / "colmap"]:
        if candidate.exists():
            log_info(f"Using project-local COLMAP: {candidate.resolve()}")
            return candidate.resolve()

    # System PATH
    system_colmap = shutil.which("colmap")
    if system_colmap:
        log_info(f"Using system COLMAP: {system_colmap}")
        return Path(system_colmap)

    log_error(
        "COLMAP binary not found.\n"
        "  gsforge requires COLMAP 4.x for both GLOMAP and classic SfM.\n\n"
        "  Install options:\n"
        "    • Download from https://github.com/colmap/colmap/releases\n"
        "      and add to PATH, OR\n"
        "    • Place the binary at ./bin/colmap (or ./bin/colmap.exe on Windows)\n"
        "      for a project-local install.\n\n"
        "  COLMAP 4.x is required for GLOMAP support (--MapperType GLOBAL).\n"
        "  COLMAP 3.x will work for classic incremental SfM only."
    )
    # log_error calls sys.exit — this line is unreachable but satisfies mypy
    raise RuntimeError("unreachable")


def check_colmap_version(colmap_bin: Path) -> str:
    """Return the COLMAP version string (e.g. '4.0.0').

    We warn (but don't abort) if the version is < 4.0, because GLOMAP
    requires 4.x but classic COLMAP still works on 3.x.
    """
    try:
        result = subprocess.run(
            [str(colmap_bin), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version_line = (result.stdout + result.stderr).strip().splitlines()[0]
        log_info(f"COLMAP version: {version_line}")
        return version_line
    except Exception as exc:
        log_warning(f"Could not determine COLMAP version: {exc}")
        return "unknown"


# ---------------------------------------------------------------------------
# COLMAP pipeline steps
# ---------------------------------------------------------------------------


def _run_colmap_step(
    colmap_bin: Path,
    subcommand: str,
    args: list[str],
    *,
    step_name: str,
) -> None:
    """Run a single COLMAP subcommand and raise on failure.

    Parameters
    ----------
    colmap_bin:
        Path to the colmap binary.
    subcommand:
        COLMAP subcommand (e.g. ``"feature_extractor"``).
    args:
        Additional arguments as a flat list of strings.
    step_name:
        Human-readable name for log messages.
    """
    cmd = [str(colmap_bin), subcommand] + args
    log_info(f"Running: {' '.join(cmd[:5])} …")

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # let COLMAP print to terminal for live feedback
            text=True,
            check=False,  # we check returncode manually for better messages
        )
    except FileNotFoundError:
        log_error(f"COLMAP binary not found at: {colmap_bin}")

    if result.returncode != 0:
        log_error(
            f"COLMAP {step_name} failed (exit code {result.returncode}).\n"
            "  Check the output above for details.\n"
            "  Common causes:\n"
            "    • Not enough images with sufficient overlap\n"
            "    • Images are too dark / blurry / textureless\n"
            "    • Insufficient GPU memory for feature extraction"
        )


def run_feature_extraction(
    colmap_bin: Path,
    database_path: Path,
    image_path: Path,
) -> None:
    """Run COLMAP feature extraction (SIFT) on all images.

    We use SIFT with default parameters which work well for VP footage.
    GPU acceleration is used automatically if available.

    Parameters
    ----------
    colmap_bin:
        Path to the colmap binary.
    database_path:
        Path to the COLMAP database file (will be created).
    image_path:
        Directory containing the extracted frames.
    """
    log_step("Feature extraction", f"{len(list(image_path.glob('*.png')))} images")

    _run_colmap_step(
        colmap_bin,
        "feature_extractor",
        [
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_path),
            # Use GPU for feature extraction if available — dramatically faster
            "--ImageReader.single_camera",
            "1",  # assume single camera (typical for VP)
            "--SiftExtraction.use_gpu",
            "1",
            "--SiftExtraction.gpu_index",
            "-1",  # auto-select GPU
        ],
        step_name="feature_extractor",
    )


def run_feature_matching(
    colmap_bin: Path,
    database_path: Path,
    num_images: int,
) -> None:
    """Run COLMAP feature matching.

    Matching strategy:
      - sequential_matcher for > 200 images: assumes temporal ordering
        (which is always true for video-derived frames) and is O(N) instead
        of O(N²).  This is the right choice for VP footage.
      - exhaustive_matcher for <= 200 images: checks all pairs, more robust
        for small image sets where sequential assumptions may not hold.

    Parameters
    ----------
    colmap_bin:
        Path to the colmap binary.
    database_path:
        Path to the COLMAP database file.
    num_images:
        Total number of images — used to choose the matching strategy.
    """
    # Sequential matching is O(N) and ideal for video-derived frames because
    # consecutive frames always overlap.  The overlap window of 10 means each
    # frame is matched against its 10 nearest temporal neighbours.
    if num_images > 200:
        log_step("Feature matching", f"sequential (N={num_images} > 200)")
        _run_colmap_step(
            colmap_bin,
            "sequential_matcher",
            [
                "--database_path",
                str(database_path),
                "--SequentialMatching.overlap",
                "10",  # match each frame to ±10 neighbours
                "--SequentialMatching.loop_detection",
                "0",  # no loop closure needed for VP
                "--SiftMatching.use_gpu",
                "1",
                "--SiftMatching.gpu_index",
                "-1",
            ],
            step_name="sequential_matcher",
        )
    else:
        log_step("Feature matching", f"exhaustive (N={num_images} <= 200)")
        _run_colmap_step(
            colmap_bin,
            "exhaustive_matcher",
            [
                "--database_path",
                str(database_path),
                "--SiftMatching.use_gpu",
                "1",
                "--SiftMatching.gpu_index",
                "-1",
            ],
            step_name="exhaustive_matcher",
        )


def run_mapper(
    colmap_bin: Path,
    database_path: Path,
    image_path: Path,
    output_path: Path,
    method: SfmMethodLiteral,
) -> None:
    """Run the COLMAP mapper (GLOMAP global or classic incremental).

    Parameters
    ----------
    colmap_bin:
        Path to the colmap binary.
    database_path:
        Path to the COLMAP database file.
    image_path:
        Directory containing the extracted frames.
    output_path:
        Directory where sparse/0/ will be written.
    method:
        ``"glomap"`` for global SfM (COLMAP 4.x required) or
        ``"colmap"`` for classic incremental SfM.
    """
    mapper_type = "GLOBAL" if method == "glomap" else "INCREMENTAL"
    log_step("Mapper", f"method={method} ({mapper_type})")

    args = [
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_path),
        "--Mapper.ba_refine_focal_length",
        "1",
        "--Mapper.ba_refine_principal_point",
        "0",  # VP cameras have known principal point
        "--Mapper.ba_refine_extra_params",
        "1",
    ]

    # GLOMAP flag — only available in COLMAP 4.x
    # If the user has COLMAP 3.x and chose glomap, COLMAP will error with a
    # clear message about the unknown flag.
    if method == "glomap":
        args += ["--Mapper.mapper_type", "GLOBAL"]

    _run_colmap_step(
        colmap_bin,
        "mapper",
        args,
        step_name=f"mapper ({method})",
    )


# ---------------------------------------------------------------------------
# Camera count helper
# ---------------------------------------------------------------------------


def count_registered_cameras(sparse_dir: Path) -> int:
    """Count the number of registered cameras in a COLMAP sparse model.

    We read ``images.bin`` (binary format) or fall back to ``images.txt``
    (text format).  Returns 0 if neither file exists.

    The binary format stores a 4-byte uint64 count at the start of the file,
    so we can get the count without parsing the full file.

    Parameters
    ----------
    sparse_dir:
        Path to the sparse/0/ directory.
    """
    images_bin = sparse_dir / "images.bin"
    images_txt = sparse_dir / "images.txt"

    if images_bin.exists():
        try:
            import struct

            data = images_bin.read_bytes()
            # COLMAP binary format: first 8 bytes = uint64 num_images
            (num_images,) = struct.unpack_from("<Q", data, 0)
            return int(num_images)
        except Exception as exc:
            log_warning(f"Could not parse images.bin: {exc}")

    if images_txt.exists():
        try:
            # Text format: non-comment lines alternate between image header
            # and point2D list.  Image headers are odd-numbered data lines.
            lines = [
                ln
                for ln in images_txt.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
            # Every 2 lines = 1 image (header + point2D line)
            return len(lines) // 2
        except Exception as exc:
            log_warning(f"Could not parse images.txt: {exc}")

    return 0


# ---------------------------------------------------------------------------
# Public API: run_sfm
# ---------------------------------------------------------------------------


def run_sfm(
    project: "GSProject",  # type: ignore[name-defined]
    method: SfmMethodLiteral = "glomap",
) -> SfmResult:
    """Run the full SfM pipeline on the project's extracted frames.

    Steps
    -----
    1. Find the COLMAP binary.
    2. Run feature extraction.
    3. Run feature matching (sequential or exhaustive based on image count).
    4. Run the mapper (GLOMAP global or COLMAP incremental).
    5. Count registered cameras.
    6. Update project.json.

    Parameters
    ----------
    project:
        The loaded GSProject instance.
    method:
        ``"glomap"`` (default) or ``"colmap"``.

    Returns
    -------
    SfmResult
        Summary for the CLI summary table.
    """
    from src.project import GSProject

    # Validate preconditions
    project.require_ingest_done()

    preprocess_dir = project.preprocess_dir
    if not preprocess_dir.exists():
        log_error(f"Preprocess directory not found: {preprocess_dir}")

    images = sorted(preprocess_dir.glob("frame_*.png"))
    if not images:
        log_error(
            f"No frames found in {preprocess_dir}.\n"
            "  Run [bold]gsforge ingest[/bold] first."
        )

    num_images = len(images)
    log_info(f"Found {num_images} frames in {preprocess_dir}")

    # Ensure SfM output directories exist
    sfm_dir = ensure_dir(project.sfm_dir)
    sparse_parent = ensure_dir(project.sfm_dir / "sparse")
    sparse_dir = ensure_dir(project.sparse_dir)  # sparse/0/

    database_path = sfm_dir / "database.db"

    # Find COLMAP binary
    colmap_bin = find_colmap_binary()
    check_colmap_version(colmap_bin)

    # Run pipeline
    try:
        run_feature_extraction(colmap_bin, database_path, preprocess_dir)
        run_feature_matching(colmap_bin, database_path, num_images)
        run_mapper(colmap_bin, database_path, preprocess_dir, sparse_parent, method)
    except SystemExit:
        # log_error calls sys.exit — catch it to update project.json before re-raising
        project.update_after_sfm(
            sfm_method=method,
            sfm_status="failed",
            camera_count=0,
        )
        raise

    # Count registered cameras
    camera_count = count_registered_cameras(sparse_dir)

    if camera_count == 0:
        log_warning(
            "SfM completed but no cameras were registered.\n"
            "  This usually means:\n"
            "    • Not enough feature matches between frames\n"
            "    • Frames are too blurry or textureless\n"
            "    • Try --method colmap (incremental) for difficult footage"
        )
        status = "failed"
    else:
        log_success(f"SfM complete — {camera_count} cameras registered.")
        status = "completed"

    # Update project.json
    project.update_after_sfm(
        sfm_method=method,
        sfm_status=status,  # type: ignore[arg-type]
        camera_count=camera_count,
    )

    return SfmResult(
        status=status,
        camera_count=camera_count,
        sparse_dir=sparse_dir,
    )


# ---------------------------------------------------------------------------
# Public API: import_colmap_reconstruction
# ---------------------------------------------------------------------------


def import_colmap_reconstruction(
    project: "GSProject",  # type: ignore[name-defined]
    source_path: Path,
) -> int:
    """Import an existing COLMAP sparse reconstruction into the project.

    Accepts either:
      - A ``sparse/0/`` directory (contains cameras.bin, images.bin, points3D.bin)
      - A ``sparse/`` directory (will look for ``0/`` inside it)
      - Any directory containing the three .bin files directly

    The reconstruction is copied (not symlinked) into ``project/sfm/sparse/0/``
    so the project remains self-contained.

    Parameters
    ----------
    project:
        The loaded GSProject instance.
    source_path:
        Path to the existing COLMAP reconstruction.

    Returns
    -------
    int
        Number of cameras in the imported reconstruction.
    """
    from src.project import GSProject

    source_path = source_path.resolve()

    # Normalise: find the directory that actually contains the .bin files
    bin_files = {"cameras.bin", "images.bin", "points3D.bin"}
    txt_files = {"cameras.txt", "images.txt", "points3D.txt"}

    def _has_colmap_files(d: Path) -> bool:
        files = {f.name for f in d.iterdir() if f.is_file()}
        return bool(bin_files & files) or bool(txt_files & files)

    # Try source_path directly, then source_path/0/, then source_path/sparse/0/
    candidates = [
        source_path,
        source_path / "0",
        source_path / "sparse" / "0",
    ]
    reconstruction_dir: Optional[Path] = None
    for candidate in candidates:
        if candidate.is_dir() and _has_colmap_files(candidate):
            reconstruction_dir = candidate
            break

    if reconstruction_dir is None:
        log_error(
            f"No COLMAP reconstruction found at: {source_path}\n"
            "  Expected a directory containing cameras.bin, images.bin, points3D.bin\n"
            "  (or their .txt equivalents).\n"
            "  Tried:\n" + "\n".join(f"    • {c}" for c in candidates)
        )

    log_info(f"Found reconstruction at: {reconstruction_dir}")

    # Ensure destination exists
    dest_dir = ensure_dir(project.sparse_dir)

    # Copy all files from reconstruction_dir into sparse/0/
    copied = 0
    for src_file in reconstruction_dir.iterdir():
        if src_file.is_file():
            dest_file = dest_dir / src_file.name
            shutil.copy2(src_file, dest_file)
            log_info(f"  Copied: {src_file.name}")
            copied += 1

    if copied == 0:
        log_error(f"No files found in reconstruction directory: {reconstruction_dir}")

    # Count cameras
    camera_count = count_registered_cameras(dest_dir)
    log_success(f"Imported {camera_count} cameras from {reconstruction_dir}")

    # Update project.json
    project.update_after_sfm(
        sfm_method="imported",
        sfm_status="completed",
        camera_count=camera_count,
    )

    return camera_count


# ---------------------------------------------------------------------------
# Public API: export_colmap
# ---------------------------------------------------------------------------


def export_colmap(
    project: "GSProject",  # type: ignore[name-defined]
    output_path: Path,
) -> None:
    """Export a clean, standard COLMAP folder that any tool can open.

    Creates the canonical COLMAP layout:
        output_path/
        ├── images/       ← copies of the extracted frames
        └── sparse/
            └── 0/        ← cameras.bin, images.bin, points3D.bin

    This layout is accepted by:
      - LichtFeld Studio (drag-and-drop import)
      - nerfstudio (``ns-train gaussian-splatting --data output_path``)
      - gsplat training scripts
      - COLMAP GUI (File → Import Model)

    Parameters
    ----------
    project:
        The loaded GSProject instance.
    output_path:
        Destination directory for the export.
    """
    from src.project import GSProject

    project.require_sfm_done()

    output_path = output_path.resolve()
    images_dest = ensure_dir(output_path / "images")
    sparse_dest = ensure_dir(output_path / "sparse" / "0")

    # Copy sparse model files
    log_step("Exporting sparse model", f"→ {sparse_dest}")
    sparse_src = project.sparse_dir
    if not sparse_src.exists():
        log_error(f"Sparse model not found: {sparse_src}")

    for src_file in sparse_src.iterdir():
        if src_file.is_file():
            shutil.copy2(src_file, sparse_dest / src_file.name)
            log_info(f"  Copied: {src_file.name}")

    # Copy (or symlink) extracted frames into images/
    log_step("Exporting images", f"→ {images_dest}")
    preprocess_dir = project.preprocess_dir
    frames = sorted(preprocess_dir.glob("frame_*.png"))

    if not frames:
        log_warning(
            f"No frames found in {preprocess_dir}.\n"
            "  The export will have an empty images/ directory."
        )
    else:
        for frame in frames:
            dest = images_dest / frame.name
            if not dest.exists():
                shutil.copy2(frame, dest)
        log_success(f"Exported {len(frames)} images to {images_dest}")

    log_success(f"COLMAP export complete: {output_path}")
