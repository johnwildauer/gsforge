"""
gsforge/ingest.py — Video frame extraction with VP-tuned smart downsampling.

Why smart downsampling matters for VP workflows
-----------------------------------------------
A typical VP shoot records at 24–60 fps.  Feeding every frame to COLMAP or
GLOMAP is counterproductive:

  1. Near-duplicate frames add almost no new information for SfM — the camera
     barely moves between consecutive frames at 24 fps.
  2. Feature extraction (SIFT/SuperPoint) is the bottleneck: it scales linearly
     with image count.  400 frames → ~2 min; 4 000 frames → ~20 min.
  3. GLOMAP's global bundle adjustment memory usage grows quadratically with
     image count.  Keeping it under ~500 images is strongly recommended.

Our strategy
------------
  1. Probe the video with FFprobe to get duration and native FPS.
  2. Compute how many frames we'd get at --target-fps.
  3. If that count exceeds --max-frames, compute a sparser interval so we
     extract exactly max_frames evenly spaced across the clip.
  4. Use FFmpeg's ``select`` filter with the computed interval to extract
     only the frames we need — no intermediate full extraction.
  5. Optionally scale frames down with FFmpeg's ``scale`` filter (--downscale).

Frame naming
------------
Frames are saved as ``preprocess/frame_000001.png`` (zero-padded to 6 digits).
This ensures lexicographic sort == temporal sort for any clip up to ~277 hours
at 1 fps, and is compatible with COLMAP's image list format.
"""

from __future__ import annotations

import math
import subprocess
import json as _json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ffmpeg  # ffmpeg-python bindings

from gsforge.utils import (
    DEFAULT_DOWNSCALE,
    DEFAULT_MAX_FRAMES,
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
# Result dataclass — returned to cli.py for summary table
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Summary of a completed frame extraction run."""

    num_frames: int  # total frames written to disk
    effective_fps: float  # actual frames-per-second extracted
    resolution: str  # "WxH" string of the output frames
    output_dir: Path  # absolute path to preprocess/


# ---------------------------------------------------------------------------
# Video probe helpers
# ---------------------------------------------------------------------------


def probe_video(video_path: Path) -> dict:
    """Use ffprobe to extract stream metadata from a video file.

    Returns the raw ffprobe JSON dict.  We parse it ourselves so we can give
    clear error messages when fields are missing (e.g. image sequences).

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
# Frame interval calculation — the core VP-tuned logic
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

    The FFmpeg ``select`` filter accepts a frame interval in *source* frames
    (not seconds), so we convert: ``interval_frames = native_fps / effective_fps``.

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
# FFmpeg extraction
# ---------------------------------------------------------------------------


def _build_ffmpeg_command(
    video_path: Path,
    output_pattern: Path,
    effective_fps: float,  # Pass the calculated FPS here
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
        The number of output files actually written (counted from the output
        directory after the run).
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
    frames_seen = 0
    stderr_lines: list[str] = []

    with make_progress(
        range(expected_frames),
        desc="Extracting frames",
        unit="frames",
        total=expected_frames,
    ) as pbar:
        # tqdm doesn't iterate here — we use it purely for the display
        # We need to manually update it as FFmpeg reports progress
        pass

    # Re-run with a simpler progress approach: just wait and report
    # (FFmpeg's stderr is not line-buffered in a way that's easy to parse live)
    assert proc is not None
    _, stderr_output = proc.communicate()

    if proc.returncode != 0:
        log_error(
            f"FFmpeg exited with code {proc.returncode}.\n"
            f"  stderr:\n{stderr_output[-2000:]}"  # last 2000 chars to avoid flooding
        )

    return proc.returncode


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_frames(
    project: "GSProject",  # type: ignore[name-defined]  # avoid circular import
    video_path: Path,
    target_fps: int = DEFAULT_TARGET_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    downscale: int = DEFAULT_DOWNSCALE,
) -> IngestResult:
    """Extract frames from a video file into the project's preprocess/ directory.

    This is the main public function called by ``cli.py``.

    Steps
    -----
    1. Probe the video to get duration, native FPS, and resolution.
    2. Compute the optimal frame interval (smart downsampling).
    3. Copy the source video into ``project/source/``.
    4. Run FFmpeg to extract frames into ``project/preprocess/``.
    5. Count the actual frames written.
    6. Update ``project.json`` via ``project.update_after_ingest()``.

    Parameters
    ----------
    project:
        The loaded GSProject instance.
    video_path:
        Absolute path to the source video file.
    target_fps:
        Desired extraction rate (frames per second of video).
    max_frames:
        Hard cap on total extracted frames.
    downscale:
        Spatial downscale factor (1 = full resolution).

    Returns
    -------
    IngestResult
        Summary of the extraction for the CLI summary table.
    """
    # Lazy import to avoid circular dependency (project imports utils, utils doesn't import project)
    from gsforge.project import GSProject

    log_step("Probing video", str(video_path.name))
    duration_s, native_fps, width, height = get_video_info(video_path)

    log_info(
        f"Video: {video_path.name}  |  "
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

    # Copy source video into project/source/ for self-containment
    dest_video = source_dir / video_path.name
    if not dest_video.exists():
        log_info(f"Copying source video to {dest_video} …")
        import shutil

        shutil.copy2(video_path, dest_video)
    else:
        log_info(f"Source video already in project: {dest_video.name}")

    # Build output pattern for FFmpeg
    output_pattern = preprocess_dir / "frame_%06d.png"

    # Build and run FFmpeg command
    log_step("Extracting frames", f"→ {preprocess_dir}")
    cmd = _build_ffmpeg_command(
        video_path=video_path,
        output_pattern=output_pattern,
        effective_fps=native_fps / interval_frames,
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
            "  Check that the video file is not corrupt and FFmpeg is working."
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
    relative_source = str(dest_video.relative_to(project.root))
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
