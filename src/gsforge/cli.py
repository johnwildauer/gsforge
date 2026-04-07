"""
cli.py — Main Typer CLI entrypoint for gsforge.

All user-facing commands live here.  Each command is a thin wrapper that:
  1. Resolves the project path.
  2. Delegates to the appropriate gsforge/ module.
  3. Prints a Rich summary table on success.

Command reference
-----------------
  gsforge init-project   Create a new .gsproject directory
  gsforge ingest         Extract frames from a video file
  gsforge import-colmap  Import an existing COLMAP sparse reconstruction
  gsforge sfm            Run GLOMAP or COLMAP SfM on extracted frames
  gsforge export-colmap  Export a clean COLMAP-compatible folder
  gsforge train          Run 3DGS training with gsplat
  gsforge info           Print project status
  gsforge run-all        Convenience: ingest + sfm + train in one shot

Design notes
------------
- We use Typer (built on Click) because it gives us automatic --help,
  shell completion, and type-safe argument parsing with minimal boilerplate.
- Every command that operates on a project accepts --project (Path).
  If omitted, gsforge walks up from CWD looking for a .gsproject directory
  (same UX as git).
- Rich is used for all terminal output — no bare print() calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from gsforge.utils import (
    DEFAULT_DOWNSCALE,
    DEFAULT_NUM_IMAGES,
    DEFAULT_PREVIEW_EVERY,
    DEFAULT_SEQUENCE_FPS,
    DEFAULT_SFM_METHOD,
    DEFAULT_TRAIN_ITERATIONS,
    console,
    log_error,
    log_info,
    log_step,
    log_success,
    log_warning,
    print_panel,
    print_summary_table,
    resolve_project_path,
)

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="gsforge",
    help=(
        "[bold cyan]gsforge[/bold cyan] — CLI-first 3D Gaussian Splatting for virtual production.\n\n"
        "Typical workflow:\n"
        "  1. [bold]gsforge init-project --name MyScene[/bold]\n"
        "  2. [bold]gsforge ingest --input footage.mp4[/bold]\n"
        "  3. [bold]gsforge sfm[/bold]\n"
        "  4. [bold]gsforge train[/bold]\n\n"
        "Or run everything at once:\n"
        "  [bold]gsforge run-all --input footage.mp4[/bold]"
    ),
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=True,
)


# ---------------------------------------------------------------------------
# init-project
# ---------------------------------------------------------------------------


@app.command("init-project")
def init_project(
    name: str = typer.Option(
        ..., "--name", "-n", help="Scene name (e.g. 'MyVPScene')."
    ),
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Parent directory in which to create the .gsproject folder. Defaults to CWD.",
        exists=False,  # The project dir doesn't exist yet — Typer must not validate it
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    """Create a new gsforge project directory.

    Creates [bold]{name}.gsproject/[/bold] with all canonical subdirectories
    and an initial [bold]project.json[/bold].

    Example:
        gsforge init-project --name MyVPScene
        gsforge init-project --name MyVPScene --project /mnt/projects
    """
    from gsforge.project import GSProject

    parent = project or Path.cwd()
    if not parent.exists():
        log_error(f"Parent directory does not exist: {parent}")

    log_step("init-project", f"name={name!r}  parent={parent}")

    proj = GSProject.create(parent, name=name)

    print_summary_table(
        title="[bold cyan]Project created[/bold cyan]",
        rows=[
            ("Project dir", str(proj.root)),
            ("project.json", str(proj.root / "project.json")),
            ("Subfolders", "source/, preprocess/, sfm/, models/, renders/, logs/"),
        ],
    )
    log_success("Done. Next step: [bold]gsforge ingest --input PATH[/bold]")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@app.command("ingest")
def ingest(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help=(
            "Path to a video file (.mp4, .mov) or the first frame of a numbered "
            "image sequence (.png, .jpg, .jpeg, .tif, .tiff, .exr). "
            "For image sequences, sibling frames are detected automatically from "
            "the filename pattern."
        ),
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Path to the .gsproject directory. Auto-detected from CWD if omitted.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    num_images: int = typer.Option(
        DEFAULT_NUM_IMAGES,
        "--num-images",
        "-n",
        help=(
            f"Number of images to extract from the source. "
            f"Default {DEFAULT_NUM_IMAGES} is tuned for VP footage: enough for "
            "GLOMAP to produce a high-quality reconstruction on a workstation. "
            "Frames are sampled evenly across the full sequence so temporal "
            "coverage is maximised regardless of source length. "
            "If the source has fewer frames than requested, all frames are used "
            "and you will be informed. "
            "Applies to both video and image-sequence inputs."
        ),
    ),
    downscale: int = typer.Option(
        DEFAULT_DOWNSCALE,
        "--downscale",
        help=(
            "Spatial downscale factor (1 = full res, 2 = half res). "
            "Downscaling by 2 cuts COLMAP feature extraction time ~4x. "
            "Applies to both video and image-sequence inputs."
        ),
    ),
    sequence_fps: int = typer.Option(
        DEFAULT_SEQUENCE_FPS,
        "--sequence-fps",
        help=(
            f"Assumed frame rate of an image sequence (default {DEFAULT_SEQUENCE_FPS} fps). "
            "Used for reporting (effective FPS) and stored in project.json. "
            "Does not affect which frames are selected. "
            "Has no effect on video inputs."
        ),
    ),
) -> None:
    """Extract frames from a video file or image sequence into preprocess/.

    [bold]Frame selection[/bold]
    Use [bold]--num-images[/bold] to specify exactly how many frames to extract.
    Frames are sampled evenly across the full source sequence so you always get
    the best temporal coverage.  If the source has fewer frames than requested,
    all frames are used and you are informed.

    [bold]Video inputs (.mp4, .mov)[/bold]
    Uses FFmpeg with a select filter to extract exactly the requested frames
    at evenly-spaced intervals across the clip.

    [bold]Image-sequence inputs (.png, .jpg, .jpeg, .tif, .tiff, .exr)[/bold]
    Provide the first frame of a numbered sequence (e.g. frame_001.png).
    gsforge detects all sibling frames automatically, applies even-stride
    subsampling to reach --num-images, and re-exports them as frame_NNNNNN.png
    into preprocess/.

    Example:
        gsforge ingest --input footage.mp4
        gsforge ingest --input footage.mp4 --num-images 500
        gsforge ingest --input footage.mp4 --downscale 2
        gsforge ingest --input frame_001.png
        gsforge ingest --input frame_001.exr --num-images 200 --sequence-fps 30
    """
    from gsforge.project import GSProject
    from gsforge import ingest as ingest_module

    proj = GSProject.from_path(resolve_project_path(project))

    log_step(
        "ingest",
        f"input={input.name}  num-images={num_images}  "
        f"seq-fps={sequence_fps}  scale=1/{downscale}",
    )

    result = ingest_module.extract_frames(
        project=proj,
        input_path=input,
        num_images=num_images,
        downscale=downscale,
        sequence_fps=sequence_fps,
    )

    print_summary_table(
        title="[bold cyan]Ingest complete[/bold cyan]",
        rows=[
            ("Input", str(input)),
            ("Images requested", str(num_images)),
            ("Frames extracted", str(result.num_frames)),
            ("Effective FPS", f"{result.effective_fps:.2f}"),
            ("Resolution", result.resolution),
            ("Output dir", str(proj.preprocess_dir)),
        ],
    )
    log_success("Done. Next step: [bold]gsforge sfm[/bold]")


# ---------------------------------------------------------------------------
# import-colmap
# ---------------------------------------------------------------------------


@app.command("import-colmap")
def import_colmap(
    source: Path = typer.Option(
        ...,
        "--source",
        "-s",
        help="Path to an existing COLMAP sparse/0/ directory (or its parent).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Path to the .gsproject directory. Auto-detected from CWD if omitted.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    """Import an existing COLMAP sparse reconstruction into this project.

    Useful when you've already run COLMAP/GLOMAP externally (e.g. in
    LichtFeld Studio or Reality Capture) and just want to use gsforge
    for training.

    The reconstruction is copied into [bold]sfm/sparse/0/[/bold] and
    project.json is updated so subsequent steps work normally.

    Example:
        gsforge import-colmap --source /path/to/sparse/0
    """
    from gsforge.project import GSProject
    from gsforge import sfm as sfm_module

    proj = GSProject.from_path(resolve_project_path(project))

    log_step("import-colmap", f"source={source}")

    camera_count = sfm_module.import_colmap_reconstruction(
        project=proj,
        source_path=source,
    )

    print_summary_table(
        title="[bold cyan]Import complete[/bold cyan]",
        rows=[
            ("Source", str(source)),
            ("Destination", str(proj.sparse_dir)),
            ("Cameras imported", str(camera_count)),
        ],
    )
    log_success("Done. Next step: [bold]gsforge train[/bold]")


# ---------------------------------------------------------------------------
# sfm
# ---------------------------------------------------------------------------


@app.command("sfm")
def sfm(
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Path to the .gsproject directory. Auto-detected from CWD if omitted.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    method: str = typer.Option(
        DEFAULT_SFM_METHOD,
        "--method",
        "-m",
        help=(
            "SfM method: 'glomap' (default, fast global SfM via COLMAP 4.x) "
            "or 'colmap' (classic incremental SfM, slower but more robust on "
            "difficult footage)."
        ),
    ),
) -> None:
    """Run Structure-from-Motion on the extracted frames.

    Defaults to [bold]GLOMAP[/bold] (global SfM via COLMAP 4.x binary) which
    is 5–20x faster than classic incremental COLMAP and more robust on the
    smooth camera paths typical of VP shoots.

    Requires the COLMAP 4.x binary on PATH or in [bold]./bin/[/bold].

    Example:
        gsforge sfm
        gsforge sfm --method colmap
    """
    from gsforge.project import GSProject
    from gsforge import sfm as sfm_module

    if method not in ("glomap", "colmap"):
        log_error(f"Unknown SfM method: {method!r}. Choose 'glomap' or 'colmap'.")

    proj = GSProject.from_path(resolve_project_path(project))
    proj.require_ingest_done()

    log_step("sfm", f"method={method}  frames={proj.meta.num_extracted_frames}")

    result = sfm_module.run_sfm(project=proj, method=method)  # type: ignore[arg-type]

    print_summary_table(
        title="[bold cyan]SfM complete[/bold cyan]",
        rows=[
            ("Method", method),
            ("Status", result.status),
            ("Cameras registered", str(result.camera_count)),
            ("Sparse model", str(result.sparse_dir)),
        ],
    )
    log_success("Done. Next step: [bold]gsforge train[/bold]")


# ---------------------------------------------------------------------------
# export-colmap
# ---------------------------------------------------------------------------


@app.command("export-colmap")
def export_colmap(
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Path to the .gsproject directory. Auto-detected from CWD if omitted.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Destination directory for the exported COLMAP folder. "
            "Defaults to [project]/export_colmap/."
        ),
        resolve_path=True,
    ),
) -> None:
    """Export a clean, standard COLMAP folder that any tool can open.

    Creates a directory with the canonical COLMAP layout:
        output/
        ├── images/       ← symlinks (or copies) of the extracted frames
        └── sparse/
            └── 0/        ← cameras.bin, images.bin, points3D.bin

    This output can be opened directly in:
      - LichtFeld Studio
      - nerfstudio (ns-train)
      - Gaussian Splatting viewers
      - COLMAP GUI

    Example:
        gsforge export-colmap
        gsforge export-colmap --output /mnt/exports/MyScene_colmap
    """
    from gsforge.project import GSProject
    from gsforge import sfm as sfm_module

    proj = GSProject.from_path(resolve_project_path(project))
    proj.require_sfm_done()

    dest = output or (proj.root / "export_colmap")

    log_step("export-colmap", f"output={dest}")

    sfm_module.export_colmap(project=proj, output_path=dest)

    print_summary_table(
        title="[bold cyan]Export complete[/bold cyan]",
        rows=[
            ("Output dir", str(dest)),
            ("images/", str(dest / "images")),
            ("sparse/0/", str(dest / "sparse" / "0")),
        ],
    )
    log_success("Done. Open the export folder in LichtFeld Studio, nerfstudio, etc.")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@app.command("train")
def train(
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Path to the .gsproject directory. Auto-detected from CWD if omitted.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    backend: str = typer.Option(
        "gsplat",
        "--backend",
        "-b",
        help="Training backend. Currently only 'gsplat' is supported.",
    ),
    iterations: int = typer.Option(
        DEFAULT_TRAIN_ITERATIONS,
        "--iterations",
        "-i",
        help=f"Number of training iterations. Default {DEFAULT_TRAIN_ITERATIONS}.",
    ),
    preview_every: int = typer.Option(
        DEFAULT_PREVIEW_EVERY,
        "--preview-every",
        help=f"Save a preview render every N iterations. Default {DEFAULT_PREVIEW_EVERY}.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help=(
            "Resume training from the latest checkpoint in models/checkpoints/. "
            "If no checkpoint exists, falls back to fresh COLMAP initialisation. "
            "By default gsforge auto-resumes when training_status is 'completed' "
            "in project.json — use this flag to force resume even when status is "
            "'failed' or 'pending'."
        ),
        is_flag=True,
    ),
    resume_from: Optional[Path] = typer.Option(
        None,
        "--resume-from",
        help=(
            "Resume from a specific checkpoint file. "
            "Accepts an absolute path or a filename relative to models/checkpoints/ "
            "(e.g. --resume-from ckpt_012000.pth). "
            "Takes precedence over --resume and smart auto-resume."
        ),
        exists=False,  # validated manually so we can check checkpoints_dir too
        file_okay=True,
        dir_okay=False,
        resolve_path=False,
    ),
    no_resume: bool = typer.Option(
        False,
        "--no-resume",
        "--restart",
        help=(
            "Force a fresh start from COLMAP, ignoring any existing checkpoints "
            "and overriding smart auto-resume. Alias: --restart."
        ),
        is_flag=True,
    ),
) -> None:
    """Train a 3D Gaussian Splatting model on the reconstructed scene.

    Loads the COLMAP sparse model from [bold]sfm/sparse/0/[/bold], trains
    with [bold]gsplat[/bold], and saves:
      - Checkpoints in [bold]models/checkpoints/[/bold]
      - Final scene as [bold]models/final_scene.ply[/bold]
      - Preview renders in [bold]renders/[/bold]

    [bold]Smart resume (default)[/bold]: if a previous training run completed
    and a checkpoint exists, gsforge automatically continues from where it left
    off.  Pass [bold]--restart[/bold] to override this and start fresh.

    [bold]Resume precedence[/bold] (highest → lowest):
      1. [bold]--restart[/bold] / [bold]--no-resume[/bold] → always start fresh
      2. [bold]--resume-from PATH[/bold]                   → use that checkpoint
      3. [bold]--resume[/bold]                             → use latest checkpoint
      4. smart auto-resume from project.json               → if training completed
      5. fallback                                          → fresh COLMAP init

    Requires a CUDA-capable GPU and gsplat installed with the correct
    PyTorch + CUDA version.

    Example:
        gsforge train
        gsforge train --iterations 30000
        gsforge train --resume
        gsforge train --resume-from ckpt_015000.pth
        gsforge train --restart --iterations 30000
        gsforge train --iterations 30000 --preview-every 1000
    """
    from gsforge.project import GSProject
    from gsforge import train as train_module

    proj = GSProject.from_path(resolve_project_path(project))
    proj.require_sfm_done()

    # Build a human-readable mode string for the log line
    if no_resume:
        mode_str = "restart"
    elif resume_from is not None:
        mode_str = f"resume-from={resume_from}"
    elif resume:
        mode_str = "resume-latest"
    else:
        mode_str = "auto"

    log_step(
        "train",
        f"backend={backend}  iterations={iterations}  "
        f"preview_every={preview_every}  mode={mode_str}",
    )

    train_module.run_training(
        project_path=proj.root,
        backend=backend,
        iterations=iterations,
        preview_every=preview_every,
        resume=resume,
        resume_from=resume_from,
        restart=no_resume,
    )


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@app.command("info")
def info(
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Path to the .gsproject directory. Auto-detected from CWD if omitted.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    """Print the current status of a gsforge project.

    Shows all pipeline stages (ingest, SfM, training) and their current
    state, plus key metadata like frame count, camera count, and output paths.

    Example:
        gsforge info
        gsforge info --project /mnt/projects/MyScene.gsproject
    """
    from gsforge.project import GSProject

    proj = GSProject.from_path(resolve_project_path(project))
    proj.print_info()


# ---------------------------------------------------------------------------
# run-all  (convenience: ingest + sfm + train)
# ---------------------------------------------------------------------------


@app.command("run-all")
def run_all(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help=(
            "Path to a video file (.mp4, .mov) or the first frame of a numbered "
            "image sequence (.png, .jpg, .jpeg, .tif, .tiff, .exr)."
        ),
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Path to the .gsproject directory. Auto-detected from CWD if omitted.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    num_images: int = typer.Option(
        DEFAULT_NUM_IMAGES,
        "--num-images",
        "-n",
        help=(
            f"Number of images to extract from the source (default {DEFAULT_NUM_IMAGES}). "
            "Frames are sampled evenly across the full sequence."
        ),
    ),
    downscale: int = typer.Option(DEFAULT_DOWNSCALE, "--downscale"),
    sequence_fps: int = typer.Option(DEFAULT_SEQUENCE_FPS, "--sequence-fps"),
    method: str = typer.Option(DEFAULT_SFM_METHOD, "--method"),
    iterations: int = typer.Option(DEFAULT_TRAIN_ITERATIONS, "--iterations"),
) -> None:
    """Run the full pipeline: ingest → SfM → train.

    Convenience command for when you want to go from raw footage or an image
    sequence to a trained 3DGS model in one shot with sensible defaults.

    Example:
        gsforge run-all --input footage.mp4
        gsforge run-all --input footage.mp4 --num-images 500
        gsforge run-all --input footage.mov --method colmap --iterations 30000
        gsforge run-all --input frame_001.png --sequence-fps 24
    """
    from gsforge.project import GSProject
    from gsforge import ingest as ingest_module
    from gsforge import sfm as sfm_module

    print_panel(
        title="gsforge run-all",
        body=(
            f"Input:       {input}\n"
            f"Num images:  {num_images}\n"
            f"Seq FPS:     {sequence_fps}\n"
            f"Downscale:   1/{downscale}\n"
            f"SfM method:  {method}\n"
            f"Iterations:  {iterations}"
        ),
    )

    proj = GSProject.from_path(resolve_project_path(project))

    # --- Step 1: Ingest ---
    log_step("Step 1/3 — ingest")
    result = ingest_module.extract_frames(
        project=proj,
        input_path=input,
        num_images=num_images,
        downscale=downscale,
        sequence_fps=sequence_fps,
    )
    log_success(f"Ingest done — {result.num_frames} frames extracted.")

    # --- Step 2: SfM ---
    log_step("Step 2/3 — sfm")
    sfm_result = sfm_module.run_sfm(project=proj, method=method)  # type: ignore[arg-type]
    log_success(f"SfM done — {sfm_result.camera_count} cameras registered.")

    # --- Step 3: Train ---
    log_step("Step 3/3 — train")
    from gsforge import train as train_module

    train_module.run_training(
        project_path=proj.root,
        backend="gsplat",
        iterations=iterations,
        preview_every=DEFAULT_PREVIEW_EVERY,
    )

    print_summary_table(
        title="[bold cyan]run-all complete[/bold cyan]",
        rows=[
            ("Images requested", str(num_images)),
            ("Frames extracted", str(result.num_frames)),
            ("Cameras registered", str(sfm_result.camera_count)),
            ("SfM model", str(proj.sparse_dir)),
            ("Final PLY", str(proj.models_dir / "final_scene.ply")),
        ],
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
