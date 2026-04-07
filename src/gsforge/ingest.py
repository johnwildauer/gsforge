"""
gsforge/ingest.py — Frame extraction for VP workflows.

Supported inputs
----------------
* **MP4 / MOV video files** — treated as a single video; frames are extracted
  with FFmpeg using user-directed even sampling (see below).
* **Image sequences** — the user provides the *first* frame of a numbered
  sequence (e.g. ``frame_001.exr``).  gsforge detects the sibling frames
  automatically from the filename pattern, copies them into ``source/``, and
  re-exports them as numbered PNGs into ``preprocess/``.

User-directed frame count selection
------------------------------------
The user explicitly requests how many images to extract/use via
``--num-images``.  The system samples evenly across the full sequence so
temporal coverage is maximised.

  * If the source has **more** frames than requested, frames are sampled at
    a uniform stride: ``stride = total_frames / num_images``.  The selected
    indices are spread as evenly as possible across the full sequence using
    ``round(i * stride)`` for i in 0..num_images-1, avoiding clustering at
    the start.
  * If the source has **fewer** frames than requested, all frames are used
    and the user is informed via a warning message.
  * Edge cases handled: num_images=1, num_images==total, num_images>total,
    tiny sequences (2 frames), and any integer num_images > 0.

Video path
----------
  1. Probe the video with FFprobe to get duration, native FPS, and total
     frame count.
  2. Compute the target frame count (clamped to available frames if needed).
  3. Select ``num_images`` frame indices evenly distributed across the clip.
  4. Use FFmpeg's ``select`` filter with the computed indices to extract
     exactly the frames we need — no intermediate full extraction.
  5. Optionally scale frames down with FFmpeg's ``scale`` filter (--downscale).

Image-sequence path
-------------------
  1. Parse the stem of the provided first frame to extract a numeric suffix.
  2. Glob the parent directory for siblings matching the same prefix + digits.
  3. Sort numerically; require at least 2 matching frames.
  4. Apply ``num_images`` even-stride subsampling if needed.
  5. Copy selected frames into ``source/`` and re-export as ``frame_NNNNNN.png``
     into ``preprocess/`` using Pillow (already a project dependency).
  6. Apply --downscale via Pillow's Lanczos resampler if requested.

Frame naming
------------
Frames are saved as ``preprocess/frame_000001.png`` (zero-padded to 6 digits).
This ensures lexicographic sort == temporal sort for any clip up to ~277 hours
at 1 fps, and is compatible with COLMAP's image list format.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import ffmpeg  # ffmpeg-python bindings
from PIL import Image

from gsforge.utils import (
    DEFAULT_DOWNSCALE,
    DEFAULT_NUM_IMAGES,
    DEFAULT_SEQUENCE_FPS,
    FRAME_TEMPLATE,
    console,
    ensure_dir,
    log_error,
    log_info,
    log_step,
    log_success,
    log_warning,
    make_progress,
)

# ---------------------------------------------------------------------------
# Format classification constants
# ---------------------------------------------------------------------------

#: File extensions treated as video files.  Never subject to sequence inference.
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov"})

#: File extensions treated as image-sequence frames.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"}
)

# ---------------------------------------------------------------------------
# Result dataclass — returned to cli.py for summary table
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Summary of a completed frame extraction run."""

    num_frames: int  # total frames written to disk
    effective_fps: float  # actual frames-per-second extracted (0.0 for sequences)
    resolution: str  # "WxH" string of the output frames
    output_dir: Path  # absolute path to preprocess/


# ---------------------------------------------------------------------------
# Format classification
# ---------------------------------------------------------------------------


def classify_input(path: Path) -> Literal["video", "image_sequence"]:
    """Determine whether *path* is a supported video file or an image-sequence frame.

    Parameters
    ----------
    path:
        The path provided by the user (must exist — Typer validates this).

    Returns
    -------
    Literal["video", "image_sequence"]

    Raises
    ------
    SystemExit
        If the file extension is not in ``VIDEO_EXTENSIONS`` or
        ``IMAGE_EXTENSIONS``.
    """
    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in IMAGE_EXTENSIONS:
        return "image_sequence"
    log_error(
        f"Unsupported input format: '{path.suffix}' (file: {path.name}).\n"
        f"  Supported video formats : {', '.join(sorted(VIDEO_EXTENSIONS))}\n"
        f"  Supported image formats : {', '.join(sorted(IMAGE_EXTENSIONS))}"
    )
    # log_error calls sys.exit — this line is unreachable but satisfies type checkers
    raise SystemExit(1)  # pragma: no cover


# ---------------------------------------------------------------------------
# Image-sequence helpers
# ---------------------------------------------------------------------------


def resolve_image_sequence(first_frame: Path) -> list[Path]:
    """Discover all frames in a numbered image sequence from the first frame.

    Algorithm
    ---------
    1. Split the stem of *first_frame* into a text prefix and a trailing
       integer using a regex (e.g. ``"frame_001"`` → prefix ``"frame_"``,
       number ``1``).
    2. Glob the parent directory for files with the same extension.
    3. Keep only siblings whose stem matches ``^{prefix}\\d+$`` exactly
       (same prefix, only digits after — no extra text).
    4. Sort numerically by the trailing integer.
    5. Require at least 2 matching frames; fail clearly otherwise.

    Parameters
    ----------
    first_frame:
        Absolute path to the first (or any) frame of the sequence.
        The file must exist.

    Returns
    -------
    list[Path]
        All matching frames sorted in ascending numeric order.

    Raises
    ------
    SystemExit
        If the stem has no trailing digits, or fewer than 2 frames are found.
    """
    stem = first_frame.stem
    ext = first_frame.suffix.lower()
    parent = first_frame.parent

    # Split stem into prefix + trailing digits
    match = re.match(r"^(.*?)(\d+)$", stem)
    if match is None:
        log_error(
            f"Cannot detect a numbered sequence from '{first_frame.name}'.\n"
            "  The filename must end with digits (e.g. 'frame_001.png', 'shot0042.exr').\n"
            "  Provide the first frame of the sequence."
        )
        raise SystemExit(1)  # pragma: no cover

    prefix = match.group(1)
    # Build a regex that matches exactly: prefix + one-or-more digits (nothing else)
    sibling_pattern = re.compile(r"^" + re.escape(prefix) + r"\d+$")

    # Collect all siblings with the same extension (case-insensitive)
    candidates: list[Path] = []
    for sibling in parent.iterdir():
        if sibling.suffix.lower() != ext:
            continue
        if sibling_pattern.match(sibling.stem):
            candidates.append(sibling)

    if len(candidates) < 2:
        log_error(
            f"Image sequence detection found only {len(candidates)} frame(s) "
            f"matching the pattern '{prefix}*.{ext.lstrip('.')}' "
            f"in '{parent}'.\n"
            "  At least 2 matching frames are required to treat the input as a sequence.\n"
            "  Check that the sibling frames are in the same directory and share the same "
            "numbered prefix."
        )
        raise SystemExit(1)  # pragma: no cover

    # Sort numerically by the trailing integer in the stem
    def _frame_number(p: Path) -> int:
        m = re.search(r"(\d+)$", p.stem)
        return int(m.group(1)) if m else 0

    candidates.sort(key=_frame_number)
    return candidates


# ---------------------------------------------------------------------------
# Even-stride frame selection — the core sampling logic
# ---------------------------------------------------------------------------


def select_frames_evenly(total_frames: int, num_images: int) -> list[int]:
    """Select *num_images* frame indices evenly distributed across *total_frames*.

    This is the authoritative sampling function used by both the video path
    (to build an FFmpeg ``select`` expression) and the image-sequence path
    (to pick which source frames to copy).

    Algorithm
    ---------
    Uses a floating-point stride so that selected indices are spread as
    uniformly as possible across the full range [0, total_frames-1]:

        stride = total_frames / num_images
        indices = [round(i * stride) for i in range(num_images)]

    This avoids the clustering-at-start bias of integer-stride approaches
    (e.g. ``frames[::stride]``) and handles non-integer ratios cleanly.

    Edge cases
    ----------
    * ``num_images >= total_frames``: returns ``list(range(total_frames))``.
    * ``num_images == 1``: returns ``[0]`` (first frame).
    * ``num_images == total_frames``: returns all indices in order.
    * ``total_frames == 0``: returns ``[]``.

    Parameters
    ----------
    total_frames:
        Total number of source frames available.
    num_images:
        Desired number of output frames.

    Returns
    -------
    list[int]
        Sorted list of 0-based frame indices to select.  Length is
        ``min(num_images, total_frames)``.
    """
    if total_frames <= 0 or num_images <= 0:
        return []

    if num_images >= total_frames:
        return list(range(total_frames))

    stride = total_frames / num_images
    indices = []
    seen: set[int] = set()
    for i in range(num_images):
        idx = min(round(i * stride), total_frames - 1)
        # Deduplicate in the rare case rounding produces the same index twice
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)

    return sorted(indices)


# ---------------------------------------------------------------------------
# Video probe helpers
# ---------------------------------------------------------------------------


def probe_video(video_path: Path) -> dict:
    """Use ffprobe to extract stream metadata from a video file.

    Returns the raw ffprobe JSON dict.  We parse it ourselves so we can give
    clear error messages when fields are missing.

    Parameters
    ----------
    video_path:
        Absolute path to the video file.

    Raises
    ------
    SystemExit
        If ffprobe is not on PATH or the file cannot be probed.
    """
    try:
        probe = ffmpeg.probe(str(video_path))
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        log_error(
            f"ffprobe failed on '{video_path.name}'.\n"
            f"  Make sure FFmpeg is installed and on PATH.\n"
            f"  ffprobe output:\n{stderr}"
        )
    return probe  # type: ignore[return-value]


def get_video_info(video_path: Path) -> tuple[float, float, int, int, int]:
    """Return (duration_seconds, native_fps, width, height, total_frames) for a video.

    We look for the first video stream in the ffprobe output.

    Parameters
    ----------
    video_path:
        Absolute path to the video file.

    Returns
    -------
    tuple
        (duration_s, native_fps, width, height, total_frames)
        ``total_frames`` is derived from duration × native_fps and may be
        approximate for VFR sources; it is used only for sampling planning.
    """
    probe = probe_video(video_path)

    # Find the first video stream
    video_streams = [
        s for s in probe.get("streams", []) if s.get("codec_type") == "video"
    ]
    if not video_streams:
        log_error(
            f"No video stream found in '{video_path.name}'.\n"
            "  Is this actually a video file?"
        )

    stream = video_streams[0]

    # Duration — prefer stream duration, fall back to container duration
    duration_str = stream.get("duration") or probe.get("format", {}).get("duration")
    if duration_str is None:
        log_error(
            f"Cannot determine duration of '{video_path.name}'.\n"
            "  Try re-encoding the file with FFmpeg first."
        )
    duration_s = float(duration_str)

    # Native FPS — stored as a fraction string like "24000/1001" (23.976 fps)
    fps_str = stream.get("r_frame_rate", "0/1")
    try:
        num, den = fps_str.split("/")
        native_fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        log_error(f"Cannot parse frame rate '{fps_str}' from '{video_path.name}'.")

    width: int = int(stream.get("width", 0))
    height: int = int(stream.get("height", 0))

    if width == 0 or height == 0:
        log_error(f"Cannot determine resolution of '{video_path.name}'.")

    # Derive total frame count from duration × native_fps.
    # ffprobe's nb_frames field is unreliable for many container formats.
    total_frames = int(math.floor(duration_s * native_fps))

    return duration_s, native_fps, width, height, total_frames


# ---------------------------------------------------------------------------
# FFmpeg extraction (video path)
# ---------------------------------------------------------------------------


def _build_ffmpeg_command(
    video_path: Path,
    output_pattern: Path,
    selected_indices: list[int],
    downscale: int,
    width: int,
    height: int,
) -> list[str]:
    """Build the FFmpeg command to extract exactly the requested frame indices.

    Uses FFmpeg's ``select`` filter with an expression that matches only the
    0-based frame numbers in *selected_indices*.  This avoids extracting all
    frames to disk and then deleting the unwanted ones.

    Parameters
    ----------
    video_path:
        Source video file.
    output_pattern:
        Output filename pattern (e.g. ``preprocess/frame_%06d.png``).
    selected_indices:
        Sorted list of 0-based frame indices to extract.
    downscale:
        Spatial downscale factor (1 = full resolution).
    width, height:
        Native video resolution (used to compute downscaled dimensions).
    """
    filters: list[str] = []

    # Build a select expression: eq(n,0)+eq(n,5)+eq(n,10)+...
    # FFmpeg evaluates this per-frame; only frames where the expression is
    # non-zero are passed through.
    if selected_indices:
        select_expr = "+".join(f"eq(n\\,{idx})" for idx in selected_indices)
        filters.append(f"select='{select_expr}'")
        # vsync=0 (or vfr) prevents FFmpeg from duplicating frames to fill
        # gaps in the selected set.
        vsync_flag = ["-vsync", "0"]
    else:
        # Fallback: extract all frames (should not happen in normal usage)
        vsync_flag = []

    if downscale > 1:
        out_w = (width // downscale) & ~1
        out_h = (height // downscale) & ~1
        filters.append(f"scale={out_w}:{out_h}:flags=lanczos")

    vf = ",".join(filters)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-q:v",
        "1",
        "-start_number",
        "1",
    ]
    cmd.extend(vsync_flag)
    cmd.append(str(output_pattern))
    return cmd


def _run_ffmpeg(cmd: list[str], expected_frames: int) -> int:
    """Execute an FFmpeg command and return the actual number of frames written.

    We parse FFmpeg's stderr to count ``frame=`` lines for a rough progress
    indicator.  FFmpeg writes progress to stderr, not stdout.

    Parameters
    ----------
    cmd:
        The FFmpeg command list (from ``_build_ffmpeg_command``).
    expected_frames:
        Approximate expected frame count — used to size the progress bar.

    Returns
    -------
    int
        The return code from FFmpeg (0 = success).
    """
    log_info(f"Running FFmpeg: {' '.join(cmd[:6])} …")

    try:
        proc = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        log_error(
            "FFmpeg binary not found.\n"
            "  Please install FFmpeg and add it to your PATH.\n"
            "  Download: https://ffmpeg.org/download.html"
        )

    # Stream stderr and count progress lines
    with make_progress(
        range(expected_frames),
        desc="Extracting frames",
        unit="frames",
        total=expected_frames,
    ) as pbar:
        # tqdm doesn't iterate here — we use it purely for the display
        pass

    # Re-run with a simpler progress approach: just wait and report
    assert proc is not None
    _, stderr_output = proc.communicate()

    if proc.returncode != 0:
        log_error(
            f"FFmpeg exited with code {proc.returncode}.\n"
            f"  stderr:\n{stderr_output[-2000:]}"  # last 2000 chars to avoid flooding
        )

    return proc.returncode


# ---------------------------------------------------------------------------
# Image-sequence ingest
# ---------------------------------------------------------------------------


def ingest_image_sequence(
    project: "GSProject",  # type: ignore[name-defined]
    frames: list[Path],
    downscale: int = DEFAULT_DOWNSCALE,
    num_images: int = DEFAULT_NUM_IMAGES,
    sequence_fps: int = DEFAULT_SEQUENCE_FPS,
) -> IngestResult:
    """Copy and re-export an image sequence into the project's preprocess/ directory.

    Steps
    -----
    1. Apply ``num_images`` even-stride subsampling: if ``len(frames) > num_images``,
       select frames evenly distributed across the full sequence.
       If ``len(frames) < num_images``, use all frames and warn the user.
    2. Copy selected source frames into ``project/source/`` for self-containment.
    3. Re-export each frame as ``frame_NNNNNN.png`` into ``project/preprocess/``,
       applying ``downscale`` via Pillow's Lanczos resampler if requested.
    4. Read resolution from the first frame.
    5. Update ``project.json`` via ``project.update_after_ingest()``.

    Parameters
    ----------
    project:
        The loaded GSProject instance.
    frames:
        Sorted list of source frame paths (from ``resolve_image_sequence``).
    downscale:
        Spatial downscale factor (1 = full resolution).
    num_images:
        Target number of frames to extract.  Frames are sampled evenly across
        the full sequence.  If the source has fewer frames than requested, all
        frames are used and the user is warned.
    sequence_fps:
        Assumed frame rate of the sequence (used for ``effective_fps`` reporting
        and stored in project.json).  Does not affect which frames are selected.

    Returns
    -------
    IngestResult
        Summary of the extraction for the CLI summary table.
    """
    total_available = len(frames)

    # Determine which frames to use
    if total_available > num_images:
        # Even-stride subsampling across the full sequence
        indices = select_frames_evenly(total_available, num_images)
        selected = [frames[i] for i in indices]
        log_info(
            f"Image sequence: {total_available} frames available, "
            f"{len(selected)} requested → selecting {len(selected)} evenly spaced frames."
        )
    elif total_available < num_images:
        # User requested more frames than exist — use all and warn
        selected = frames
        log_warning(
            f"Requested {num_images} images but the sequence only has "
            f"{total_available} frames.\n"
            f"  Using all {total_available} available frames instead."
        )
    else:
        # Exact match — use all frames
        selected = frames
        log_info(
            f"Image sequence: {total_available} frames available, "
            f"{num_images} requested → using all frames."
        )

    log_info(
        f"Image sequence: {len(selected)} frames selected from {total_available} available."
    )

    # Ensure output directories exist
    preprocess_dir = ensure_dir(project.preprocess_dir)
    source_dir = ensure_dir(project.source_dir)

    # Read resolution from the first selected frame
    with Image.open(selected[0]) as img:
        orig_w, orig_h = img.size

    if downscale > 1:
        out_w = (orig_w // downscale) & ~1
        out_h = (orig_h // downscale) & ~1
    else:
        out_w, out_h = orig_w, orig_h

    log_step("Copying and exporting frames", f"→ {preprocess_dir}")

    num_written = 0
    for idx, src_frame in enumerate(selected, start=1):
        # Copy source frame into project/source/ for self-containment
        dest_source = source_dir / src_frame.name
        if not dest_source.exists():
            shutil.copy2(src_frame, dest_source)

        # Re-export as frame_NNNNNN.png into preprocess/
        dest_preprocess = preprocess_dir / f"frame_{idx:06d}.png"
        with Image.open(src_frame) as img:
            # Convert to RGB so we can always save as PNG regardless of source format
            # (e.g. EXR is float, TIFF may be 16-bit — PNG output is 8-bit RGB)
            rgb = img.convert("RGB")
            if downscale > 1:
                rgb = rgb.resize((out_w, out_h), Image.LANCZOS)
            rgb.save(dest_preprocess, format="PNG")
        num_written += 1

    if num_written == 0:
        log_error(
            "No frames were written to disk.\n"
            f"  Output directory: {preprocess_dir}\n"
            "  Check that the source frames are readable."
        )

    log_success(f"Exported {num_written} frames to {preprocess_dir}")

    # Compute resolution string
    if downscale > 1:
        resolution = f"{out_w}x{out_h} (downscaled from {orig_w}x{orig_h})"
    else:
        resolution = f"{out_w}x{out_h}"

    # effective_fps is the assumed sequence FPS (user-supplied or default 24)
    effective_fps = float(sequence_fps)

    # Store the source directory path relative to project root
    relative_source = str(source_dir.relative_to(project.root))
    project.update_after_ingest(
        input_type="images",
        input_path=relative_source,
        num_images_requested=num_images,
        downscale=downscale,
        num_extracted_frames=num_written,
    )

    return IngestResult(
        num_frames=num_written,
        effective_fps=effective_fps,
        resolution=resolution,
        output_dir=preprocess_dir,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_frames(
    project: "GSProject",  # type: ignore[name-defined]  # avoid circular import
    input_path: Path,
    num_images: int = DEFAULT_NUM_IMAGES,
    downscale: int = DEFAULT_DOWNSCALE,
    sequence_fps: int = DEFAULT_SEQUENCE_FPS,
) -> IngestResult:
    """Extract frames from a video file or image sequence into preprocess/.

    This is the main public function called by ``cli.py``.  It classifies the
    input, then dispatches to either the video (FFmpeg) path or the
    image-sequence (Pillow) path.

    The ``num_images`` parameter is the single authoritative control for how
    many frames are extracted.  Frames are always sampled evenly across the
    full source sequence to maximise temporal coverage.

    Video path steps
    ----------------
    1. Probe the video to get duration, native FPS, total frames, and resolution.
    2. Determine the actual frame count to extract (clamped to available frames
       if num_images > total_frames, with a warning).
    3. Compute evenly-spaced frame indices using ``select_frames_evenly()``.
    4. Copy the source file into ``project/source/``.
    5. Run FFmpeg with a ``select`` filter to extract exactly those frames.
    6. Count the actual frames written.
    7. Update ``project.json`` via ``project.update_after_ingest()``.

    Image-sequence path steps
    -------------------------
    1. Resolve all sibling frames from the provided first frame.
    2. Apply num_images even-stride subsampling (or warn if over-requested).
    3. Copy source frames into ``project/source/``.
    4. Re-export as ``frame_NNNNNN.png`` into ``project/preprocess/``.
    5. Update ``project.json`` via ``project.update_after_ingest()``.

    Parameters
    ----------
    project:
        The loaded GSProject instance.
    input_path:
        Absolute path to the source file — a video (.mp4, .mov) or the first
        frame of a numbered image sequence (.png, .jpg, .jpeg, .tif, .tiff, .exr).
    num_images:
        Target number of images to extract.  Frames are sampled evenly across
        the full source sequence.  If the source has fewer frames than requested,
        all frames are used and the user is warned.
    downscale:
        Spatial downscale factor (1 = full resolution).
    sequence_fps:
        Assumed frame rate of the image sequence.  Used for ``effective_fps``
        reporting and stored in project.json.  Ignored for video inputs.

    Returns
    -------
    IngestResult
        Summary of the extraction for the CLI summary table.
    """
    # Lazy import to avoid circular dependency
    from gsforge.project import GSProject

    input_kind = classify_input(input_path)

    # ------------------------------------------------------------------
    # Image-sequence path
    # ------------------------------------------------------------------
    if input_kind == "image_sequence":
        if sequence_fps != DEFAULT_SEQUENCE_FPS:
            log_info(
                f"Using --sequence-fps={sequence_fps} for image sequence "
                f"'{input_path.name}'."
            )
        log_step("Detecting image sequence", str(input_path.name))
        frames = resolve_image_sequence(input_path)
        log_info(f"Detected {len(frames)} frames in sequence.")
        return ingest_image_sequence(
            project=project,
            frames=frames,
            downscale=downscale,
            num_images=num_images,
            sequence_fps=sequence_fps,
        )

    # ------------------------------------------------------------------
    # Video path (MP4 / MOV)
    # ------------------------------------------------------------------
    # sequence_fps is irrelevant for video — warn if user passed a non-default value
    if sequence_fps != DEFAULT_SEQUENCE_FPS:
        log_warning(
            f"--sequence-fps={sequence_fps} is ignored for video inputs "
            f"('{input_path.name}' is a video file)."
        )

    log_step("Probing source file", str(input_path.name))
    duration_s, native_fps, width, height, total_frames = get_video_info(input_path)

    log_info(
        f"Source: {input_path.name}  |  "
        f"Duration: {duration_s:.1f}s  |  "
        f"Native FPS: {native_fps:.3f}  |  "
        f"Resolution: {width}x{height}  |  "
        f"Total frames: ~{total_frames}"
    )

    # Validate inputs
    if num_images <= 0:
        log_error(f"--num-images must be > 0, got {num_images}.")
    if downscale < 1:
        log_error(f"--downscale must be >= 1, got {downscale}.")

    # Handle over-request: user asked for more frames than the video has
    if num_images > total_frames:
        log_warning(
            f"Requested {num_images} images but the video only has ~{total_frames} frames "
            f"({duration_s:.1f}s at {native_fps:.2f} fps).\n"
            f"  Using all ~{total_frames} available frames instead."
        )
        actual_num_images = total_frames
    else:
        actual_num_images = num_images

    # Compute evenly-spaced frame indices
    selected_indices = select_frames_evenly(total_frames, actual_num_images)
    expected_count = len(selected_indices)

    log_info(
        f"Extraction plan: {expected_count} frames selected from ~{total_frames} total  |  "
        f"Stride: ~{total_frames / max(expected_count, 1):.1f} source frames per output frame"
    )

    # Ensure output directory exists
    preprocess_dir = ensure_dir(project.preprocess_dir)
    source_dir = ensure_dir(project.source_dir)

    # Copy source file into project/source/ for self-containment
    dest_source = source_dir / input_path.name
    if not dest_source.exists():
        log_info(f"Copying source file to {dest_source} …")
        shutil.copy2(input_path, dest_source)
    else:
        log_info(f"Source file already in project: {dest_source.name}")

    # Build output pattern for FFmpeg
    output_pattern = preprocess_dir / "frame_%06d.png"

    # Build and run FFmpeg command
    log_step("Extracting frames", f"→ {preprocess_dir}")
    cmd = _build_ffmpeg_command(
        video_path=input_path,
        output_pattern=output_pattern,
        selected_indices=selected_indices,
        downscale=downscale,
        width=width,
        height=height,
    )

    _run_ffmpeg(cmd, expected_frames=expected_count)

    # Count actual frames written (ground truth — FFmpeg may write slightly
    # more or fewer than expected due to rounding in the select filter)
    actual_frames = sorted(preprocess_dir.glob("frame_*.png"))
    num_frames = len(actual_frames)

    if num_frames == 0:
        log_error(
            "FFmpeg ran but no frames were written to disk.\n"
            f"  Output directory: {preprocess_dir}\n"
            "  Check that the source file is not corrupt and FFmpeg is working."
        )

    log_success(f"Extracted {num_frames} frames to {preprocess_dir}")

    # Compute effective FPS for the summary
    effective_fps = num_frames / duration_s if duration_s > 0 else 0.0

    # Compute output resolution
    if downscale > 1:
        out_w = (width // downscale) & ~1
        out_h = (height // downscale) & ~1
        resolution = f"{out_w}x{out_h} (downscaled from {width}x{height})"
    else:
        resolution = f"{width}x{height}"

    # Update project.json
    # Store the source path relative to the project root for portability
    relative_source = str(dest_source.relative_to(project.root))
    project.update_after_ingest(
        input_type="video",
        input_path=relative_source,
        num_images_requested=num_images,
        downscale=downscale,
        num_extracted_frames=num_frames,
    )

    return IngestResult(
        num_frames=num_frames,
        effective_fps=effective_fps,
        resolution=resolution,
        output_dir=preprocess_dir,
    )
