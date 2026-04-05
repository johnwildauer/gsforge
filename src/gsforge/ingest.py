"""
gsforge/ingest.py — Frame extraction for VP workflows.

Supported inputs
----------------
* **MP4 / MOV video files** — treated as a single video; frames are extracted
  with FFmpeg using VP-tuned smart downsampling (see below).
* **Image sequences** — the user provides the *first* frame of a numbered
  sequence (e.g. ``frame_001.exr``).  gsforge detects the sibling frames
  automatically from the filename pattern, copies them into ``source/``, and
  re-exports them as numbered PNGs into ``preprocess/``.

Why smart downsampling matters for VP workflows (video path)
------------------------------------------------------------
A typical VP shoot records at 24–60 fps.  Feeding every frame to COLMAP or
GLOMAP is counterproductive:

  1. Near-duplicate frames add almost no new information for SfM — the camera
     barely moves between consecutive frames at 24 fps.
  2. Feature extraction (SIFT/SuperPoint) is the bottleneck: it scales linearly
     with image count.  400 frames → ~2 min; 4 000 frames → ~20 min.
  3. GLOMAP's global bundle adjustment memory usage grows quadratically with
     image count.  Keeping it under ~500 images is strongly recommended.

Our strategy (video path)
-------------------------
  1. Probe the video with FFprobe to get duration and native FPS.
  2. Compute how many frames we'd get at --target-fps.
  3. If that count exceeds --max-frames, compute a sparser interval so we
     extract exactly max_frames evenly spaced across the clip.
  4. Use FFmpeg's ``fps`` filter with the computed interval to extract
     only the frames we need — no intermediate full extraction.
  5. Optionally scale frames down with FFmpeg's ``scale`` filter (--downscale).

Image-sequence path
-------------------
  1. Parse the stem of the provided first frame to extract a numeric suffix.
  2. Glob the parent directory for siblings matching the same prefix + digits.
  3. Sort numerically; require at least 2 matching frames.
  4. Apply --max-frames cap (evenly spaced subsample if needed).
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
    DEFAULT_MAX_FRAMES,
    DEFAULT_SEQUENCE_FPS,
    DEFAULT_TARGET_FPS,
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


def get_video_info(video_path: Path) -> tuple[float, float, int, int]:
    """Return (duration_seconds, native_fps, width, height) for a video file.

    We look for the first video stream in the ffprobe output.

    Parameters
    ----------
    video_path:
        Absolute path to the video file.

    Returns
    -------
    tuple
        (duration_s, native_fps, width, height)
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

    return duration_s, native_fps, width, height


# ---------------------------------------------------------------------------
# Frame interval calculation — the core VP-tuned logic (video path)
# ---------------------------------------------------------------------------


def compute_frame_interval(
    duration_s: float,
    native_fps: float,
    target_fps: int,
    max_frames: int,
) -> tuple[float, int]:
    """Compute the FFmpeg frame selection interval and expected frame count.

    This is the heart of the smart downsampling logic.

    Strategy
    --------
    1. Compute ``naive_count`` = how many frames we'd get at ``target_fps``.
    2. If ``naive_count <= max_frames``, use ``target_fps`` directly.
    3. Otherwise, compute a sparser interval so we get exactly ``max_frames``
       evenly distributed across the clip.

    The FFmpeg ``fps`` filter accepts a target FPS value directly, so we
    convert: ``effective_fps = max_frames / duration_s`` when capping.

    Parameters
    ----------
    duration_s:
        Video duration in seconds.
    native_fps:
        Native frame rate of the video stream.
    target_fps:
        Desired extraction rate (frames per second of video).
    max_frames:
        Hard cap on total extracted frames.

    Returns
    -------
    tuple
        (interval_frames, expected_count) where ``interval_frames`` is the
        FFmpeg select interval (keep 1 frame every N source frames) and
        ``expected_count`` is the approximate number of frames that will be
        written.
    """
    # How many frames would we get at the requested fps?
    naive_count = int(duration_s * target_fps)

    if naive_count <= max_frames:
        # Happy path: target_fps is fine as-is
        effective_fps = float(target_fps)
        expected_count = naive_count
    else:
        # We'd exceed max_frames — compute a sparser effective fps
        # so we get exactly max_frames spread evenly across the clip.
        effective_fps = max_frames / duration_s
        expected_count = max_frames
        log_info(
            f"At {target_fps} fps this clip would yield {naive_count} frames "
            f"(exceeds --max-frames={max_frames}).\n"
            f"  Adjusting to {effective_fps:.2f} fps → ~{expected_count} frames."
        )

    # Convert effective fps to a source-frame interval for FFmpeg's select filter.
    # e.g. native=24, effective=5 → interval=4.8 → keep every ~5th source frame.
    interval_frames = native_fps / effective_fps

    return interval_frames, expected_count


# ---------------------------------------------------------------------------
# FFmpeg extraction (video path)
# ---------------------------------------------------------------------------


def _build_ffmpeg_command(
    video_path: Path,
    output_pattern: Path,
    effective_fps: float,
    downscale: int,
    width: int,
    height: int,
) -> list[str]:
    filters: list[str] = []

    # Use the 'fps' filter. It handles all the 'mod' math for you
    # and ensures frames are picked at even time intervals.
    filters.append(f"fps={round(effective_fps, 4)}")

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
        str(output_pattern),
    ]
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
    max_frames: int = DEFAULT_MAX_FRAMES,
    sequence_fps: int = DEFAULT_SEQUENCE_FPS,
) -> IngestResult:
    """Copy and re-export an image sequence into the project's preprocess/ directory.

    Steps
    -----
    1. Apply ``max_frames`` cap: if ``len(frames) > max_frames``, subsample
       evenly using a stride so we keep the best temporal spread.
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
    max_frames:
        Hard cap on total frames written to preprocess/.
    sequence_fps:
        Assumed frame rate of the sequence (used for ``effective_fps`` reporting
        and stored in project.json).  Does not affect which frames are selected.

    Returns
    -------
    IngestResult
        Summary of the extraction for the CLI summary table.
    """
    total_available = len(frames)

    # Apply max_frames cap via even stride
    if total_available > max_frames:
        stride = math.ceil(total_available / max_frames)
        selected = frames[::stride][:max_frames]
        log_info(
            f"Image sequence has {total_available} frames "
            f"(exceeds --max-frames={max_frames}).\n"
            f"  Subsampling every {stride} frames → {len(selected)} frames."
        )
    else:
        selected = frames

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
        target_fps=sequence_fps,
        max_frames=max_frames,
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
    target_fps: int = DEFAULT_TARGET_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    downscale: int = DEFAULT_DOWNSCALE,
    sequence_fps: int = DEFAULT_SEQUENCE_FPS,
) -> IngestResult:
    """Extract frames from a video file or image sequence into preprocess/.

    This is the main public function called by ``cli.py``.  It classifies the
    input, then dispatches to either the video (FFmpeg) path or the
    image-sequence (Pillow) path.

    Video path steps
    ----------------
    1. Probe the video to get duration, native FPS, and resolution.
    2. Compute the optimal frame interval (smart downsampling).
    3. Copy the source file into ``project/source/``.
    4. Run FFmpeg to extract frames into ``project/preprocess/``.
    5. Count the actual frames written.
    6. Update ``project.json`` via ``project.update_after_ingest()``.

    Image-sequence path steps
    -------------------------
    1. Resolve all sibling frames from the provided first frame.
    2. Apply max_frames cap.
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
    target_fps:
        Desired extraction rate for video files (frames per second of video).
        Ignored for image sequences.
    max_frames:
        Hard cap on total extracted frames.  Applied to both video and sequence
        inputs.
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
            max_frames=max_frames,
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
    duration_s, native_fps, width, height = get_video_info(input_path)

    log_info(
        f"Source: {input_path.name}  |  "
        f"Duration: {duration_s:.1f}s  |  "
        f"Native FPS: {native_fps:.3f}  |  "
        f"Resolution: {width}x{height}"
    )

    # Validate inputs
    if target_fps <= 0:
        log_error(f"--target-fps must be > 0, got {target_fps}.")
    if max_frames <= 0:
        log_error(f"--max-frames must be > 0, got {max_frames}.")
    if downscale < 1:
        log_error(f"--downscale must be >= 1, got {downscale}.")

    # Warn if target_fps > native_fps — we can't extract more frames than exist
    if target_fps > native_fps:
        log_warning(
            f"--target-fps ({target_fps}) > native FPS ({native_fps:.2f}).\n"
            f"  Clamping to native FPS."
        )
        target_fps = int(math.floor(native_fps))

    # Compute frame interval
    interval_frames, expected_count = compute_frame_interval(
        duration_s=duration_s,
        native_fps=native_fps,
        target_fps=target_fps,
        max_frames=max_frames,
    )

    log_info(
        f"Extraction plan: ~{expected_count} frames  |  "
        f"Interval: every {interval_frames:.1f} source frames  |  "
        f"Effective FPS: {native_fps / interval_frames:.2f}"
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
        effective_fps=native_fps / interval_frames,
        downscale=downscale,
        width=width,
        height=height,
    )

    _run_ffmpeg(cmd, expected_frames=expected_count)

    # Count actual frames written (ground truth — FFmpeg may write slightly
    # more or fewer than expected due to rounding in the fps filter)
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
        target_fps=target_fps,
        max_frames=max_frames,
        downscale=downscale,
        num_extracted_frames=num_frames,
    )

    return IngestResult(
        num_frames=num_frames,
        effective_fps=effective_fps,
        resolution=resolution,
        output_dir=preprocess_dir,
    )
