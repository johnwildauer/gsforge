"""
src/project.py — GSProject: the "smart folder" at the heart of gsforge.

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
    target_fps: Optional[int] = None
    max_frames: Optional[int] = None
    downscale: Optional[int] = None
    num_extracted_frames: Optional[int] = None

    # --- Set after SfM ---
    sfm_method: Optional[SfmMethod] = None
    sfm_status: Optional[PipelineStatus] = None
    camera_count: Optional[int] = None

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
    project.update_after_ingest(num_frames=312, target_fps=5, ...)
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

        # Also create the nested sfm/sparse/0/ structure that COLMAP expects
        (project_dir / "sfm" / "sparse" / "0").mkdir(parents=True)

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
        """Standard COLMAP sparse/0/ directory inside sfm/."""
        return self.root / "sfm" / "sparse" / "0"

    @property
    def models_dir(self) -> Path:
        """Directory for 3DGS checkpoints and final .ply."""
        return self.root / "models"

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
        """
        for subfolder in SUBFOLDERS:
            ensure_dir(self.root / subfolder)
        ensure_dir(self.sparse_dir)

    # ------------------------------------------------------------------
    # Metadata update helpers — called by pipeline steps
    # ------------------------------------------------------------------

    def update_after_ingest(
        self,
        *,
        input_type: InputType,
        input_path: str,
        target_fps: int,
        max_frames: int,
        downscale: int,
        num_extracted_frames: int,
    ) -> None:
        """Record ingest results in project.json.

        Called by ``src/ingest.py`` after frame extraction completes.
        """
        self.meta.input_type = input_type
        self.meta.input_path = input_path
        self.meta.target_fps = target_fps
        self.meta.max_frames = max_frames
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
    ) -> None:
        """Record SfM results in project.json.

        Called by ``src/sfm.py`` after reconstruction completes (or fails).
        """
        self.meta.sfm_method = sfm_method
        self.meta.sfm_status = sfm_status
        self.meta.camera_count = camera_count
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

        Called by ``src/train.py`` after training completes (or fails).
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
            "Target FPS": _fmt(
                str(self.meta.target_fps) if self.meta.target_fps else None
            ),
            "Max frames": _fmt(
                str(self.meta.max_frames) if self.meta.max_frames else None
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

    def require_ingest_done(self) -> None:
        """Exit with a helpful error if ingest has not been run yet."""
        if not self.is_ingest_done():
            log_error(
                "No frames have been extracted for this project.\n"
                "  Run [bold]gsforge ingest --video PATH[/bold] first."
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
