"""
gsforge/project.py — GSProject: the "smart folder" at the heart of gsforge.

A gsforge project is simply a directory named ``Something.gsproject/``
containing a ``project.json`` metadata file and a set of canonical
subdirectories.  This design means:

  - The project is self-contained and portable (zip it, move it, share it).
  - Any tool that understands COLMAP can open the ``sfm/`` subfolder directly.
  - The JSON file gives gsforge (and future GUIs) full pipeline state without
    scanning the filesystem.

Folder layout created by GSProject.create()
-------------------------------------------
MyScene.gsproject/
├── project.json          ← pipeline metadata (this module manages it)
├── source/               ← original video / image files (copied on ingest)
├── preprocess/           ← extracted frames: frame_000001.png …
├── sfm/                  ← COLMAP/GLOMAP sparse reconstruction
│   └── sparse/
│       └── 0/            ← cameras.bin, images.bin, points3D.bin
├── models/               ← 3DGS checkpoints + final_scene.ply
├── renders/              ← preview renders during training
└── logs/                 ← per-step log files
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from gsforge.utils import (
    PROJECT_FILENAME,
    SUBFOLDERS,
    console,
    ensure_dir,
    is_gsproject,
    log_error,
    log_info,
    log_success,
    log_warning,
    print_summary_table,
)

# ---------------------------------------------------------------------------
# Type aliases — keep Literal types in one place so they're easy to extend
# ---------------------------------------------------------------------------

InputType = Literal["video", "images", "imported"]
SfmMethod = Literal["glomap", "colmap", "imported", "none"]
PipelineStatus = Literal["pending", "completed", "failed"]


# ---------------------------------------------------------------------------
# Project metadata dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProjectMeta:
    """All persistent metadata for a gsforge project.

    Every field maps 1-to-1 to a key in ``project.json``.  Optional fields
    are None until the corresponding pipeline step completes.

    We use a dataclass (not a plain dict) so that:
      - Fields are typed and IDE-discoverable.
      - ``asdict()`` gives us a clean JSON-serialisable dict for free.
      - Adding new fields in future versions is trivial.
    """

    # --- Always present ---
    version: str = "1.0"
    created: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    # --- Set at init-project time ---
    name: Optional[str] = None

    # --- Set after ingest ---
    input_type: Optional[InputType] = None
    input_path: Optional[str] = None  # relative path inside project dir
    # num_images_requested: the number of images the user asked for.
    # Replaces the old target_fps + max_frames pair.  The actual extracted
    # count may differ (e.g. source had fewer frames than requested).
    num_images_requested: Optional[int] = None
    downscale: Optional[int] = None
    num_extracted_frames: Optional[int] = None

    # --- Set after SfM ---
    sfm_method: Optional[SfmMethod] = None
    sfm_status: Optional[PipelineStatus] = None
    camera_count: Optional[int] = None
    # Path to the best sparse sub-model selected after SfM, relative to project root.
    # e.g. "sfm/sparse/0" or "sfm/sparse/2" (the model with the most registered images).
    sparse_model_dir: Optional[str] = None

    # --- Set after training ---
    training_status: Optional[PipelineStatus] = None
    final_ply: Optional[str] = None  # relative path inside project dir
    last_iteration: Optional[int] = None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict, omitting None values for cleanliness."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectMeta":
        """Construct from a raw dict (e.g. parsed JSON).

        Unknown keys are silently ignored so that older project.json files
        remain loadable after gsforge adds new fields.
        """
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# GSProject — the main public API for this module
# ---------------------------------------------------------------------------


class GSProject:
    """Represents a single gsforge project directory.

    Typical usage
    -------------
    # Create a brand-new project:
    project = GSProject.create(Path("."), name="MyVPScene")

    # Load an existing project:
    project = GSProject.from_path(Path("MyVPScene.gsproject"))

    # Update metadata after a pipeline step:
    project.update_after_ingest(num_frames=312, num_images_requested=300, ...)
    """

    def __init__(self, root: Path, meta: ProjectMeta) -> None:
        """Low-level constructor — prefer ``create()`` or ``from_path()``."""
        self.root: Path = root.resolve()
        self.meta: ProjectMeta = meta

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        parent_dir: Path,
        name: str,
        *,
        exist_ok: bool = False,
    ) -> "GSProject":
        """Create a new project directory and initialise ``project.json``.

        Parameters
        ----------
        parent_dir:
            Directory in which to create the ``.gsproject`` folder.
            Typically the current working directory.
        name:
            Human-readable scene name (e.g. ``"MyVPScene"``).
            The folder will be named ``{name}.gsproject``.
        exist_ok:
            If True, silently reuse an existing project directory.
            If False (default), raise an error if the folder already exists.

        Returns
        -------
        GSProject
            The newly created (or reused) project instance.
        """
        # Sanitise name — strip whitespace, replace spaces with underscores
        safe_name = name.strip().replace(" ", "_")
        project_dir = parent_dir.resolve() / f"{safe_name}.gsproject"

        if project_dir.exists():
            if not exist_ok:
                log_error(
                    f"Project directory already exists: {project_dir}\n"
                    "  Use a different name or delete the existing project."
                )
            log_warning(f"Reusing existing project directory: {project_dir}")
            return cls.from_path(project_dir)

        # Create the root directory
        project_dir.mkdir(parents=True)

        # Create all canonical subdirectories
        for subfolder in SUBFOLDERS:
            (project_dir / subfolder).mkdir()

        # Create sfm/sparse/ but NOT sfm/sparse/0/ — the COLMAP mapper creates
        # its own numbered sub-directories (0/, 1/, …).  Pre-creating sparse/0/
        # can confuse GLOMAP's global mapper.  import-colmap creates sparse/0/
        # itself via ensure_dir(project.sparse_dir).
        (project_dir / "sfm" / "sparse").mkdir(parents=True)

        # Initialise metadata
        meta = ProjectMeta(name=safe_name)
        project = cls(project_dir, meta)
        project.save()

        log_success(f"Created project: {project_dir}")
        return project

    @classmethod
    def from_path(cls, path: Path) -> "GSProject":
        """Load an existing project from a directory path.

        Parameters
        ----------
        path:
            Path to the ``.gsproject`` directory (or any directory containing
            ``project.json``).

        Raises
        ------
        SystemExit
            If the path is not a valid project directory.
        """
        path = path.resolve()

        if not path.is_dir():
            log_error(f"Project path is not a directory: {path}")

        json_path = path / PROJECT_FILENAME
        if not json_path.exists():
            log_error(
                f"No '{PROJECT_FILENAME}' found in: {path}\n"
                "  This directory does not look like a gsforge project.\n"
                "  Run [bold]gsforge init-project[/bold] to initialise it."
            )

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            log_error(f"Corrupt project.json in {path}: {exc}")

        meta = ProjectMeta.from_dict(data)
        return cls(path, meta)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write current metadata to ``project.json``.

        We write to a temp file first then rename atomically so that a crash
        mid-write never leaves a corrupt project.json.
        """
        json_path = self.root / PROJECT_FILENAME
        tmp_path = json_path.with_suffix(".json.tmp")

        payload = json.dumps(self.meta.to_dict(), indent=2, ensure_ascii=False)
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(json_path)  # atomic on POSIX; best-effort on Windows

    # ------------------------------------------------------------------
    # Subfolder accessors — always return absolute Path objects
    # ------------------------------------------------------------------

    @property
    def source_dir(self) -> Path:
        """Directory for original source files (video / images)."""
        return self.root / "source"

    @property
    def preprocess_dir(self) -> Path:
        """Directory for extracted frames (frame_000001.png …)."""
        return self.root / "preprocess"

    @property
    def sfm_dir(self) -> Path:
        """Root of the SfM output tree."""
        return self.root / "sfm"

    @property
    def sparse_dir(self) -> Path:
        """Standard COLMAP sparse/0/ directory inside sfm/.

        This is the *default* sub-model path used when no SfM has been run yet
        (e.g. for ``import-colmap`` which always writes to sparse/0/).
        For the *best* model selected after a full SfM run, use
        ``best_sparse_dir`` instead.
        """
        return self.root / "sfm" / "sparse" / "0"

    @property
    def best_sparse_dir(self) -> Path:
        """Path to the best sparse sub-model selected after SfM.

        If ``project.json`` records a ``sparse_model_dir`` (set by
        ``update_after_sfm``), that path is returned.  Otherwise falls back
        to ``sparse/0/`` for backward compatibility with projects created
        before this field was introduced.
        """
        if self.meta.sparse_model_dir is not None:
            return self.root / self.meta.sparse_model_dir
        return self.sparse_dir

    @property
    def models_dir(self) -> Path:
        """Directory for 3DGS checkpoints and final .ply."""
        return self.root / "models"

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for 3DGS training checkpoints (models/checkpoints/)."""
        return self.root / "models" / "checkpoints"

    @property
    def renders_dir(self) -> Path:
        """Directory for preview renders during training."""
        return self.root / "renders"

    @property
    def logs_dir(self) -> Path:
        """Directory for per-step log files."""
        return self.root / "logs"

    # ------------------------------------------------------------------
    # Ensure subdirectories exist (idempotent — safe to call any time)
    # ------------------------------------------------------------------

    def ensure_subdirs(self) -> None:
        """Create any missing canonical subdirectories.

        Useful when loading a project that was created by an older version of
        gsforge that didn't have all the subdirectories.

        Note: creates ``sfm/sparse/`` but NOT ``sfm/sparse/0/`` — the mapper
        creates its own numbered sub-directories.
        """
        for subfolder in SUBFOLDERS:
            ensure_dir(self.root / subfolder)
        # Ensure sfm/sparse/ exists (parent of sub-models), but not sparse/0/
        ensure_dir(self.root / "sfm" / "sparse")

    # ------------------------------------------------------------------
    # Metadata update helpers — called by pipeline steps
    # ------------------------------------------------------------------

    def update_after_ingest(
        self,
        *,
        input_type: InputType,
        input_path: str,
        num_images_requested: int,
        downscale: int,
        num_extracted_frames: int,
    ) -> None:
        """Record ingest results in project.json.

        Called by ``gsforge/ingest.py`` after frame extraction completes.
        """
        self.meta.input_type = input_type
        self.meta.input_path = input_path
        self.meta.num_images_requested = num_images_requested
        self.meta.downscale = downscale
        self.meta.num_extracted_frames = num_extracted_frames
        self.save()
        log_info(f"project.json updated — {num_extracted_frames} frames ingested.")

    def update_after_sfm(
        self,
        *,
        sfm_method: SfmMethod,
        sfm_status: PipelineStatus,
        camera_count: int,
        sparse_model_dir: Optional[str] = None,
    ) -> None:
        """Record SfM results in project.json.

        Called by ``gsforge/sfm.py`` after reconstruction completes (or fails).

        Parameters
        ----------
        sparse_model_dir:
            Path to the best sparse sub-model, relative to the project root
            (e.g. ``"sfm/sparse/0"``).  If ``None``, the field is not updated.
        """
        self.meta.sfm_method = sfm_method
        self.meta.sfm_status = sfm_status
        self.meta.camera_count = camera_count
        if sparse_model_dir is not None:
            self.meta.sparse_model_dir = sparse_model_dir
        self.save()
        log_info(f"project.json updated — SfM {sfm_status} ({camera_count} cameras).")

    def update_after_training(
        self,
        *,
        training_status: PipelineStatus,
        final_ply: Optional[str] = None,
        last_iteration: Optional[int] = None,
    ) -> None:
        """Record training results in project.json.

        Called by ``gsforge/train.py`` after training completes (or fails).
        """
        self.meta.training_status = training_status
        if final_ply is not None:
            self.meta.final_ply = final_ply
        if last_iteration is not None:
            self.meta.last_iteration = last_iteration
        self.save()
        log_info(f"project.json updated — training {training_status}.")

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, str]:
        """Return a human-readable status dict for all pipeline stages.

        Used by ``gsforge info`` to print the summary table.
        """

        def _fmt(val: Optional[str], default: str = "—") -> str:
            return val if val is not None else default

        frames = (
            str(self.meta.num_extracted_frames)
            if self.meta.num_extracted_frames is not None
            else "—"
        )

        return {
            "Name": _fmt(self.meta.name),
            "Version": self.meta.version,
            "Created": self.meta.created,
            "Input type": _fmt(self.meta.input_type),
            "Input path": _fmt(self.meta.input_path),
            "Images requested": _fmt(
                str(self.meta.num_images_requested)
                if self.meta.num_images_requested is not None
                else None
            ),
            "Downscale": _fmt(
                str(self.meta.downscale) if self.meta.downscale else None
            ),
            "Extracted frames": frames,
            "SfM method": _fmt(self.meta.sfm_method),
            "SfM status": _fmt(self.meta.sfm_status),
            "Camera count": _fmt(
                str(self.meta.camera_count)
                if self.meta.camera_count is not None
                else None
            ),
            "Sparse model": _fmt(self.meta.sparse_model_dir),
            "Training status": _fmt(self.meta.training_status),
            "Final PLY": _fmt(self.meta.final_ply),
            "Last iteration": _fmt(
                str(self.meta.last_iteration)
                if self.meta.last_iteration is not None
                else None
            ),
        }

    def is_ingest_done(self) -> bool:
        """Return True if frame extraction has completed successfully."""
        return (
            self.meta.num_extracted_frames is not None
            and self.meta.num_extracted_frames > 0
        )

    def is_sfm_done(self) -> bool:
        """Return True if SfM has completed successfully."""
        return self.meta.sfm_status == "completed"

    def is_training_done(self) -> bool:
        """Return True if 3DGS training has completed successfully."""
        return self.meta.training_status == "completed"

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Return the path to the most recent checkpoint in models/checkpoints/.

        Resolution order:
          1. If ``meta.last_iteration`` is set and the matching file exists,
             return it directly (fast path — avoids a directory scan).
          2. Otherwise scan the checkpoints directory for ``ckpt_*.pth`` files
             and return the one with the highest iteration number.
          3. Return ``None`` if no checkpoints exist.

        This method is safe to call on old projects that have no checkpoints
        directory or no training metadata — it will simply return ``None``.
        """
        ckpt_dir = self.checkpoints_dir
        if not ckpt_dir.is_dir():
            return None

        # Fast path: trust last_iteration from project.json
        if self.meta.last_iteration is not None:
            candidate = ckpt_dir / f"ckpt_{self.meta.last_iteration:06d}.pth"
            if candidate.exists():
                return candidate

        # Fallback: scan directory for highest-numbered checkpoint
        ckpt_files = sorted(ckpt_dir.glob("ckpt_*.pth"))
        if not ckpt_files:
            return None

        # Sort by the numeric iteration embedded in the filename
        def _iter_from_name(p: Path) -> int:
            try:
                return int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                return -1

        return max(ckpt_files, key=_iter_from_name)

    def should_resume(self) -> bool:
        """Return True when smart auto-resume should trigger by default.

        Conditions (all must be true):
          - ``training_status`` is ``"completed"`` (a previous run finished)
          - ``last_iteration`` is recorded in project.json
          - At least one checkpoint file exists in models/checkpoints/

        This is intentionally conservative: if any condition is missing we
        fall back to a fresh COLMAP initialisation rather than silently
        resuming from an unexpected state.
        """
        if self.meta.training_status != "completed":
            return False
        if self.meta.last_iteration is None:
            return False
        return self.get_latest_checkpoint() is not None

    def require_ingest_done(self) -> None:
        """Exit with a helpful error if ingest has not been run yet."""
        if not self.is_ingest_done():
            log_error(
                "No frames have been extracted for this project.\n"
                "  Run [bold]gsforge ingest --input PATH[/bold] first."
            )

    def require_sfm_done(self) -> None:
        """Exit with a helpful error if SfM has not been run yet."""
        if not self.is_sfm_done():
            log_error(
                "SfM has not completed for this project.\n"
                "  Run [bold]gsforge sfm[/bold] or [bold]gsforge import-colmap[/bold] first."
            )

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def print_info(self) -> None:
        """Print a full project status table to the terminal."""
        status = self.get_status()
        rows = [(k, v) for k, v in status.items()]
        print_summary_table(
            title=f"[bold cyan]gsforge project — {self.root.name}[/bold cyan]",
            rows=rows,
        )

    def __repr__(self) -> str:
        return f"GSProject(root={self.root!r}, name={self.meta.name!r})"
