"""
src/utils.py — Shared utilities for gsforge.

Covers:
  - Rich-based console logging (info, success, warning, error)
  - tqdm progress bar factory
  - Path validation helpers
  - Shared constants (default FPS, max frames, subfolder names, etc.)

Why these defaults?
  - DEFAULT_TARGET_FPS = 5: For a typical VP shoot at 24–60 fps, extracting
    every frame gives thousands of near-identical images. COLMAP/GLOMAP only
    needs ~60–80% overlap between adjacent frames. At 5 fps you get one frame
    every 200 ms — plenty of overlap for a slow camera move, and dramatically
    faster SfM (fewer images = fewer feature matches = less RAM + time).
  - DEFAULT_MAX_FRAMES = 400: Even at 5 fps a long clip can produce 1000+
    frames. 300–400 images is the sweet spot for GLOMAP: fast enough to run
    on a workstation in minutes, dense enough for a high-quality reconstruction.
    Users can raise this for very large scenes.
  - DEFAULT_DOWNSCALE = 1: No downscale by default. Users on tight hardware
    can pass --downscale 2 to halve resolution, which cuts COLMAP feature
    extraction time by ~4x.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Iterator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from tqdm import tqdm as _tqdm

# ---------------------------------------------------------------------------
# Constants — VP-tuned defaults
# ---------------------------------------------------------------------------

#: Extract one frame every N seconds of video (5 fps = 1 frame per 200 ms).
#: This is the primary knob for controlling how many images COLMAP/GLOMAP sees.
DEFAULT_TARGET_FPS: int = 5

#: Hard cap on extracted frames. Even at 5 fps a 90-second clip = 450 frames.
#: Capping at 400 keeps SfM tractable on a single workstation.
DEFAULT_MAX_FRAMES: int = 400

#: Spatial downscale factor applied to extracted frames (1 = full resolution).
#: Downscaling by 2 halves width/height, cutting feature extraction time ~4x.
DEFAULT_DOWNSCALE: int = 1

#: Default SfM method. GLOMAP (global SfM via COLMAP 4.x) is dramatically
#: faster than classic incremental COLMAP and more robust on VP footage where
#: the camera path is smooth and well-constrained.
DEFAULT_SFM_METHOD: str = "glomap"

#: Default number of 3DGS training iterations. 15 000 is a good balance
#: between quality and training time (~10–20 min on a modern GPU).
DEFAULT_TRAIN_ITERATIONS: int = 15_000

#: Save a preview render every N training iterations.
DEFAULT_PREVIEW_EVERY: int = 500

#: Canonical subfolder names inside a .gsproject directory.
#: Keeping these as constants prevents typos and makes refactoring easy.
SUBFOLDERS: tuple[str, ...] = (
    "source",  # original video / image files copied here
    "preprocess",  # extracted frames (frame_000001.png …)
    "sfm",  # COLMAP/GLOMAP sparse reconstruction output
    "models",  # 3DGS checkpoints + final .ply
    "renders",  # preview renders during training
    "logs",  # per-step log files
)

#: The metadata file that makes a directory a "smart" gsforge project folder.
PROJECT_FILENAME: str = "project.json"

#: Frame filename template — zero-padded to 6 digits so lexicographic sort
#: matches temporal order for any clip up to ~277 hours at 1 fps.
FRAME_TEMPLATE: str = "frame_{:06d}.png"

# ---------------------------------------------------------------------------
# Rich console — single shared instance so all modules share the same theme
# ---------------------------------------------------------------------------

_THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "highlight": "bold magenta",
        "path": "underline cyan",
        "dim": "dim white",
    }
)

#: Module-level console. Import this in other modules:
#:   from src.utils import console
console = Console(theme=_THEME, highlight=False)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_info(message: str) -> None:
    """Print an informational message in cyan."""
    console.print(f"[info]ℹ  {message}[/info]")


def log_success(message: str) -> None:
    """Print a success message in bold green."""
    console.print(f"[success]✔  {message}[/success]")


def log_warning(message: str) -> None:
    """Print a warning in bold yellow — non-fatal, but user should know."""
    console.print(f"[warning]⚠  {message}[/warning]")


def log_error(message: str) -> None:
    """Print an error in bold red and exit with code 1.

    We exit immediately because gsforge commands are sequential pipelines;
    continuing after a fatal error would produce confusing downstream failures.
    """
    console.print(f"[error]✖  {message}[/error]")
    sys.exit(1)


def log_step(step: str, detail: str = "") -> None:
    """Print a pipeline step header — used to visually separate major phases."""
    if detail:
        console.print(f"\n[highlight]▶  {step}[/highlight]  [dim]{detail}[/dim]")
    else:
        console.print(f"\n[highlight]▶  {step}[/highlight]")


def print_panel(title: str, body: str, style: str = "cyan") -> None:
    """Render a Rich Panel — used for command summaries and welcome banners."""
    console.print(Panel(body, title=title, border_style=style, expand=False))


def print_summary_table(title: str, rows: list[tuple[str, str]]) -> None:
    """Render a two-column key/value summary table using Rich.

    Parameters
    ----------
    title:
        Table title shown above the grid.
    rows:
        List of (key, value) string pairs.
    """
    table = Table(title=title, show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim", no_wrap=True)
    table.add_column("Value", style="bold")
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)


# ---------------------------------------------------------------------------
# Progress bar factory
# ---------------------------------------------------------------------------


def make_progress(
    iterable: Iterable,
    *,
    desc: str = "",
    total: int | None = None,
    unit: str = "it",
) -> Iterator:
    """Wrap an iterable with a tqdm progress bar styled for gsforge.

    We use tqdm (not Rich's Progress) here because tqdm integrates cleanly
    with subprocess output and is familiar to most ML practitioners.

    Parameters
    ----------
    iterable:
        The sequence to iterate over.
    desc:
        Short description shown to the left of the bar.
    total:
        Override the total count (useful when iterable has no __len__).
    unit:
        Unit label shown after the count (e.g. "frames", "iters").
    """
    return _tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        ncols=90,
        colour="cyan",
        dynamic_ncols=False,
    )


# ---------------------------------------------------------------------------
# Path validation helpers
# ---------------------------------------------------------------------------


def require_path_exists(path: Path, label: str = "Path") -> Path:
    """Raise a user-friendly error if *path* does not exist.

    Parameters
    ----------
    path:
        The filesystem path to check.
    label:
        Human-readable name used in the error message (e.g. "Video file").

    Returns
    -------
    Path
        The same path, so callers can chain: ``p = require_path_exists(p)``.
    """
    if not path.exists():
        log_error(f"{label} not found: {path}")
    return path


def require_dir(path: Path, label: str = "Directory") -> Path:
    """Raise a user-friendly error if *path* is not an existing directory."""
    if not path.is_dir():
        log_error(f"{label} is not a directory: {path}")
    return path


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it does not already exist.

    Returns the path so callers can chain:
        frames_dir = ensure_dir(project_root / "preprocess")
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_gsproject(path: Path) -> bool:
    """Return True if *path* looks like a valid gsforge project directory.

    A directory is considered a project if:
      1. Its name ends with ``.gsproject``, OR
      2. It contains a ``project.json`` file.

    The second condition lets users rename their project folder without
    breaking gsforge's ability to find the metadata.
    """
    if not path.is_dir():
        return False
    return path.name.endswith(".gsproject") or (path / PROJECT_FILENAME).exists()


def find_gsproject(start: Path) -> Path | None:
    """Walk *start* and its parents looking for a .gsproject directory.

    This mirrors how git finds .git — so users can run gsforge commands from
    any subdirectory inside their project without specifying --project.

    Returns None if no project is found.
    """
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if is_gsproject(candidate):
            return candidate
    return None


def resolve_project_path(project_arg: Path | None) -> Path:
    """Resolve the --project argument to an absolute project directory.

    Resolution order:
      1. If --project was given explicitly, use it (must be a valid project).
      2. Otherwise, walk up from CWD looking for a .gsproject directory.
      3. If nothing found, exit with a helpful error.

    Parameters
    ----------
    project_arg:
        The value passed to --project, or None if omitted.
    """
    if project_arg is not None:
        p = project_arg.resolve()
        if not is_gsproject(p):
            log_error(
                f"'{p}' does not look like a gsforge project.\n"
                "  Expected a directory ending in '.gsproject' or containing 'project.json'.\n"
                "  Run [bold]gsforge init-project[/bold] to create one."
            )
        return p

    # Auto-detect from CWD
    found = find_gsproject(Path.cwd())
    if found is None:
        log_error(
            "No gsforge project found in the current directory or any parent.\n"
            "  Run [bold]gsforge init-project --name MyScene[/bold] to create one,\n"
            "  or pass [bold]--project PATH[/bold] explicitly."
        )
    return found  # type: ignore[return-value]  # log_error exits, mypy doesn't know


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def format_count(n: int, singular: str, plural: str | None = None) -> str:
    """Return a grammatically correct count string.

    Examples
    --------
    >>> format_count(1, "frame")
    '1 frame'
    >>> format_count(312, "frame")
    '312 frames'
    """
    word = singular if n == 1 else (plural or f"{singular}s")
    return f"{n} {word}"


def human_size(num_bytes: int) -> str:
    """Convert a byte count to a human-readable string (KB / MB / GB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0  # type: ignore[assignment]
    return f"{num_bytes:.1f} PB"
