"""
tests/test_ingest.py — Unit tests for gsforge.ingest

Coverage
--------
* classify_input()          — MP4, MOV, PNG, unsupported extension
* resolve_image_sequence()  — happy path, no digits, only one frame, prefix isolation
* select_frames_evenly()    — even distribution, edge cases (1, equal, over-request, tiny)
* ingest_image_sequence()   — end-to-end copy+rename, num_images sampling, downscale
* extract_frames()          — dispatch to video path (MP4/MOV) and sequence path

Design notes
------------
* No real FFmpeg calls are made.  The video path is tested by mocking
  ``subprocess.Popen`` and ``ffmpeg.probe`` so the tests run without FFmpeg
  installed.
* The image-sequence path uses real temporary PNG files written with Pillow,
  so it exercises the actual Pillow resize/save logic without any mocking.
* All tests use ``tmp_path`` (pytest's built-in temporary directory fixture)
  so they are fully isolated and leave no files behind.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from gsforge.ingest import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    IngestResult,
    classify_input,
    extract_frames,
    ingest_image_sequence,
    resolve_image_sequence,
    select_frames_evenly,
)
from gsforge.utils import DEFAULT_NUM_IMAGES, DEFAULT_SEQUENCE_FPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png(path: Path, width: int = 64, height: int = 64) -> Path:
    """Write a small solid-colour PNG to *path* and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    img.save(path, format="PNG")
    return path


def _make_sequence(
    directory: Path,
    prefix: str = "frame_",
    ext: str = ".png",
    count: int = 5,
    width: int = 64,
    height: int = 64,
) -> list[Path]:
    """Create *count* numbered PNG frames in *directory* and return sorted paths."""
    frames = []
    for i in range(1, count + 1):
        p = directory / f"{prefix}{i:03d}{ext}"
        _make_png(p, width=width, height=height)
        frames.append(p)
    return frames


def _make_gsproject(tmp_path: Path) -> "GSProject":  # type: ignore[name-defined]
    """Create a minimal GSProject in *tmp_path* for testing."""
    from gsforge.project import GSProject

    return GSProject.create(tmp_path, name="TestProject", exist_ok=True)


# ---------------------------------------------------------------------------
# classify_input
# ---------------------------------------------------------------------------


class TestClassifyInput:
    def test_mp4_is_video(self, tmp_path: Path) -> None:
        f = tmp_path / "clip.mp4"
        f.touch()
        assert classify_input(f) == "video"

    def test_mp4_uppercase_is_video(self, tmp_path: Path) -> None:
        f = tmp_path / "clip.MP4"
        f.touch()
        assert classify_input(f) == "video"

    def test_mov_is_video(self, tmp_path: Path) -> None:
        f = tmp_path / "clip.mov"
        f.touch()
        assert classify_input(f) == "video"

    def test_mov_uppercase_is_video(self, tmp_path: Path) -> None:
        f = tmp_path / "clip.MOV"
        f.touch()
        assert classify_input(f) == "video"

    def test_png_is_image_sequence(self, tmp_path: Path) -> None:
        f = tmp_path / "frame_001.png"
        f.touch()
        assert classify_input(f) == "image_sequence"

    def test_jpg_is_image_sequence(self, tmp_path: Path) -> None:
        f = tmp_path / "frame_001.jpg"
        f.touch()
        assert classify_input(f) == "image_sequence"

    def test_exr_is_image_sequence(self, tmp_path: Path) -> None:
        f = tmp_path / "frame_001.exr"
        f.touch()
        assert classify_input(f) == "image_sequence"

    def test_tiff_is_image_sequence(self, tmp_path: Path) -> None:
        f = tmp_path / "frame_001.tiff"
        f.touch()
        assert classify_input(f) == "image_sequence"

    def test_unsupported_extension_exits(self, tmp_path: Path) -> None:
        f = tmp_path / "clip.avi"
        f.touch()
        with pytest.raises(SystemExit):
            classify_input(f)

    def test_unsupported_extension_mxf_exits(self, tmp_path: Path) -> None:
        f = tmp_path / "clip.mxf"
        f.touch()
        with pytest.raises(SystemExit):
            classify_input(f)


# ---------------------------------------------------------------------------
# resolve_image_sequence
# ---------------------------------------------------------------------------


class TestResolveImageSequence:
    def test_happy_path_returns_sorted_frames(self, tmp_path: Path) -> None:
        frames = _make_sequence(tmp_path, prefix="frame_", count=5)
        result = resolve_image_sequence(frames[0])
        assert len(result) == 5
        # Verify numeric sort order
        numbers = [int(p.stem.replace("frame_", "")) for p in result]
        assert numbers == sorted(numbers)

    def test_happy_path_all_siblings_found(self, tmp_path: Path) -> None:
        frames = _make_sequence(tmp_path, prefix="shot", count=10)
        result = resolve_image_sequence(frames[0])
        assert len(result) == 10

    def test_no_trailing_digits_exits(self, tmp_path: Path) -> None:
        f = tmp_path / "frame.png"
        _make_png(f)
        with pytest.raises(SystemExit):
            resolve_image_sequence(f)

    def test_only_one_matching_frame_exits(self, tmp_path: Path) -> None:
        # Only one frame with this prefix — should fail
        f = tmp_path / "unique_001.png"
        _make_png(f)
        with pytest.raises(SystemExit):
            resolve_image_sequence(f)

    def test_prefix_isolation_no_cross_contamination(self, tmp_path: Path) -> None:
        """frame_001.png must NOT match frame_extra_001.png."""
        # Create two separate sequences in the same directory
        _make_sequence(tmp_path, prefix="frame_", count=3)
        _make_sequence(tmp_path, prefix="frame_extra_", count=3)

        first_frame = tmp_path / "frame_001.png"
        result = resolve_image_sequence(first_frame)
        # Should only find the 3 "frame_NNN" files, not the "frame_extra_NNN" ones
        assert len(result) == 3
        assert all(p.stem.startswith("frame_") for p in result)
        assert not any("extra" in p.stem for p in result)

    def test_mixed_extensions_not_included(self, tmp_path: Path) -> None:
        """frame_001.png should not pull in frame_002.jpg."""
        _make_png(tmp_path / "frame_001.png")
        _make_png(tmp_path / "frame_002.png")
        # Create a .jpg with the same prefix — should be ignored
        img = Image.new("RGB", (64, 64))
        img.save(tmp_path / "frame_003.jpg", format="JPEG")

        result = resolve_image_sequence(tmp_path / "frame_001.png")
        assert len(result) == 2
        assert all(p.suffix == ".png" for p in result)

    def test_numeric_sort_not_lexicographic(self, tmp_path: Path) -> None:
        """frame_9.png must come before frame_10.png (numeric, not lex)."""
        for i in [1, 2, 9, 10, 11]:
            _make_png(tmp_path / f"frame_{i}.png")

        result = resolve_image_sequence(tmp_path / "frame_1.png")
        numbers = [int(p.stem.replace("frame_", "")) for p in result]
        assert numbers == [1, 2, 9, 10, 11]


# ---------------------------------------------------------------------------
# select_frames_evenly — the core sampling function
# ---------------------------------------------------------------------------


class TestSelectFramesEvenly:
    def test_empty_source_returns_empty(self) -> None:
        assert select_frames_evenly(0, 10) == []

    def test_zero_requested_returns_empty(self) -> None:
        assert select_frames_evenly(100, 0) == []

    def test_request_one_returns_first_frame(self) -> None:
        result = select_frames_evenly(100, 1)
        assert result == [0]

    def test_request_equals_total_returns_all(self) -> None:
        result = select_frames_evenly(5, 5)
        assert result == [0, 1, 2, 3, 4]

    def test_request_greater_than_total_returns_all(self) -> None:
        result = select_frames_evenly(5, 1000)
        assert result == [0, 1, 2, 3, 4]

    def test_even_distribution_no_clustering_at_start(self) -> None:
        """1000 frames, request 200 → every 5th frame, evenly spread."""
        result = select_frames_evenly(1000, 200)
        assert len(result) == 200
        # First index should be 0, last should be near 999
        assert result[0] == 0
        assert result[-1] >= 990  # last selected frame is near the end
        # Gaps between consecutive selected frames should be approximately equal
        gaps = [result[i + 1] - result[i] for i in range(len(result) - 1)]
        # All gaps should be 4 or 5 (stride ≈ 5.0)
        assert all(4 <= g <= 6 for g in gaps), f"Unexpected gaps: {set(gaps)}"

    def test_750_from_1000(self) -> None:
        """1000 frames, request 750 → stride ≈ 1.33, every 1–2 frames."""
        result = select_frames_evenly(1000, 750)
        assert len(result) == 750
        assert result[0] == 0
        assert result[-1] >= 990
        # No duplicates
        assert len(set(result)) == len(result)
        # Sorted
        assert result == sorted(result)

    def test_tiny_sequence_two_frames_request_one(self) -> None:
        result = select_frames_evenly(2, 1)
        assert result == [0]

    def test_tiny_sequence_two_frames_request_two(self) -> None:
        result = select_frames_evenly(2, 2)
        assert result == [0, 1]

    def test_tiny_sequence_two_frames_request_many(self) -> None:
        result = select_frames_evenly(2, 100)
        assert result == [0, 1]

    def test_result_is_sorted(self) -> None:
        result = select_frames_evenly(500, 100)
        assert result == sorted(result)

    def test_no_duplicates(self) -> None:
        result = select_frames_evenly(500, 100)
        assert len(set(result)) == len(result)

    def test_all_indices_in_valid_range(self) -> None:
        total = 300
        result = select_frames_evenly(total, 100)
        assert all(0 <= idx < total for idx in result)

    def test_single_frame_source(self) -> None:
        """Edge case: only 1 frame available."""
        result = select_frames_evenly(1, 1)
        assert result == [0]

    def test_single_frame_source_over_request(self) -> None:
        result = select_frames_evenly(1, 50)
        assert result == [0]


# ---------------------------------------------------------------------------
# ingest_image_sequence
# ---------------------------------------------------------------------------


class TestIngestImageSequence:
    def test_copies_and_renames_frames(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=5)
        proj = _make_gsproject(tmp_path)

        result = ingest_image_sequence(project=proj, frames=frames)

        # Check output files exist with correct naming
        output_files = sorted(proj.preprocess_dir.glob("frame_*.png"))
        assert len(output_files) == 5
        assert output_files[0].name == "frame_000001.png"
        assert output_files[4].name == "frame_000005.png"

    def test_returns_ingest_result(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=3)
        proj = _make_gsproject(tmp_path)

        result = ingest_image_sequence(project=proj, frames=frames)

        assert isinstance(result, IngestResult)
        assert result.num_frames == 3
        assert result.output_dir == proj.preprocess_dir

    def test_effective_fps_uses_sequence_fps(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=3)
        proj = _make_gsproject(tmp_path)

        result = ingest_image_sequence(project=proj, frames=frames, sequence_fps=30)

        assert result.effective_fps == 30.0

    def test_default_sequence_fps_is_24(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=3)
        proj = _make_gsproject(tmp_path)

        result = ingest_image_sequence(project=proj, frames=frames)

        assert result.effective_fps == float(DEFAULT_SEQUENCE_FPS)

    def test_num_images_subsamples_evenly(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=10)
        proj = _make_gsproject(tmp_path)

        result = ingest_image_sequence(project=proj, frames=frames, num_images=5)

        output_files = sorted(proj.preprocess_dir.glob("frame_*.png"))
        assert len(output_files) == 5
        assert result.num_frames == 5

    def test_num_images_no_cap_when_under_limit(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=4)
        proj = _make_gsproject(tmp_path)

        result = ingest_image_sequence(project=proj, frames=frames, num_images=10)

        # Source has 4 frames, requested 10 → use all 4 (with warning)
        assert result.num_frames == 4

    def test_num_images_over_request_uses_all_frames(self, tmp_path: Path) -> None:
        """Requesting more images than available should use all frames."""
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=3)
        proj = _make_gsproject(tmp_path)

        with patch("gsforge.ingest.log_warning") as mock_warn:
            result = ingest_image_sequence(project=proj, frames=frames, num_images=1000)

        assert result.num_frames == 3
        # Should have warned the user
        mock_warn.assert_called_once()
        warning_text = mock_warn.call_args[0][0]
        assert "1000" in warning_text  # requested count mentioned
        assert "3" in warning_text  # available count mentioned

    def test_num_images_exactly_equal_uses_all(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=5)
        proj = _make_gsproject(tmp_path)

        result = ingest_image_sequence(project=proj, frames=frames, num_images=5)

        assert result.num_frames == 5

    def test_num_images_one_extracts_single_frame(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=10)
        proj = _make_gsproject(tmp_path)

        result = ingest_image_sequence(project=proj, frames=frames, num_images=1)

        assert result.num_frames == 1

    def test_downscale_halves_resolution(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=2, width=64, height=64)
        proj = _make_gsproject(tmp_path)

        result = ingest_image_sequence(project=proj, frames=frames, downscale=2)

        # Output frames should be 32x32 (64 // 2 = 32, & ~1 = 32)
        output_file = proj.preprocess_dir / "frame_000001.png"
        with Image.open(output_file) as img:
            assert img.size == (32, 32)

        assert "32x32" in result.resolution
        assert "64x64" in result.resolution  # original size mentioned

    def test_no_downscale_preserves_resolution(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=2, width=64, height=64)
        proj = _make_gsproject(tmp_path)

        result = ingest_image_sequence(project=proj, frames=frames, downscale=1)

        output_file = proj.preprocess_dir / "frame_000001.png"
        with Image.open(output_file) as img:
            assert img.size == (64, 64)

        assert result.resolution == "64x64"

    def test_source_frames_copied_to_source_dir(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=3)
        proj = _make_gsproject(tmp_path)

        ingest_image_sequence(project=proj, frames=frames)

        # All source frames should be copied into project/source/
        for frame in frames:
            assert (proj.source_dir / frame.name).exists()

    def test_project_json_updated_with_images_type(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=3)
        proj = _make_gsproject(tmp_path)

        ingest_image_sequence(project=proj, frames=frames)

        # Reload project to check persisted metadata
        from gsforge.project import GSProject

        proj_reloaded = GSProject.from_path(proj.root)
        assert proj_reloaded.meta.input_type == "images"
        assert proj_reloaded.meta.num_extracted_frames == 3

    def test_project_json_records_num_images_requested(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=10)
        proj = _make_gsproject(tmp_path)

        ingest_image_sequence(project=proj, frames=frames, num_images=5)

        from gsforge.project import GSProject

        proj_reloaded = GSProject.from_path(proj.root)
        assert proj_reloaded.meta.num_images_requested == 5

    def test_even_distribution_large_sequence(self, tmp_path: Path) -> None:
        """100 frames, request 10 → every 10th frame, evenly spread."""
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=100)
        proj = _make_gsproject(tmp_path)

        result = ingest_image_sequence(project=proj, frames=frames, num_images=10)

        assert result.num_frames == 10
        output_files = sorted(proj.preprocess_dir.glob("frame_*.png"))
        assert len(output_files) == 10


# ---------------------------------------------------------------------------
# extract_frames — dispatch tests (video path mocked, sequence path real)
# ---------------------------------------------------------------------------


class TestExtractFramesDispatch:
    """Verify that extract_frames() dispatches correctly based on file extension.

    The video path is tested by mocking ffmpeg.probe and subprocess.Popen so
    no real FFmpeg binary is required.  The sequence path uses real temp files.
    """

    def _mock_ffprobe_result(self) -> dict:
        """Return a minimal ffprobe-style dict for a 10-second 24fps 1920x1080 clip."""
        return {
            "streams": [
                {
                    "codec_type": "video",
                    "duration": "10.0",
                    "r_frame_rate": "24/1",
                    "width": 1920,
                    "height": 1080,
                }
            ],
            "format": {"duration": "10.0"},
        }

    def _mock_popen(self, tmp_path: Path, num_frames: int = 5) -> MagicMock:
        """Return a mock Popen that writes dummy PNG frames to preprocess/."""

        def _side_effect(cmd, **kwargs):
            # Parse the output pattern from the command to find preprocess dir
            # cmd[-1] is the output pattern like /path/preprocess/frame_%06d.png
            output_pattern = cmd[-1]
            out_dir = Path(output_pattern).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(1, num_frames + 1):
                _make_png(out_dir / f"frame_{i:06d}.png")
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate.return_value = ("", "")
            return mock_proc

        mock = MagicMock(side_effect=_side_effect)
        return mock

    def test_mp4_dispatches_to_video_path(self, tmp_path: Path) -> None:
        video = tmp_path / "clip.mp4"
        video.touch()
        proj = _make_gsproject(tmp_path)

        with (
            patch(
                "gsforge.ingest.ffmpeg.probe", return_value=self._mock_ffprobe_result()
            ),
            patch("gsforge.ingest.subprocess.Popen", self._mock_popen(tmp_path)),
        ):
            result = extract_frames(project=proj, input_path=video)

        assert result.num_frames > 0
        assert proj.meta.input_type == "video"

    def test_mov_dispatches_to_video_path(self, tmp_path: Path) -> None:
        video = tmp_path / "clip.MOV"
        video.touch()
        proj = _make_gsproject(tmp_path)

        with (
            patch(
                "gsforge.ingest.ffmpeg.probe", return_value=self._mock_ffprobe_result()
            ),
            patch("gsforge.ingest.subprocess.Popen", self._mock_popen(tmp_path)),
        ):
            result = extract_frames(project=proj, input_path=video)

        assert result.num_frames > 0
        assert proj.meta.input_type == "video"

    def test_png_dispatches_to_sequence_path(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=4)
        proj = _make_gsproject(tmp_path)

        result = extract_frames(project=proj, input_path=frames[0])

        assert result.num_frames == 4
        assert proj.meta.input_type == "images"

    def test_sequence_fps_ignored_for_video_with_warning(self, tmp_path: Path) -> None:
        """Passing --sequence-fps with a video input should warn but not fail."""
        video = tmp_path / "clip.mp4"
        video.touch()
        proj = _make_gsproject(tmp_path)

        # The mock video is 10s at 24fps → ~240 total frames.
        # Use num_images=100 (well under 240) so no over-request warning fires,
        # leaving only the sequence-fps warning to assert on.
        with (
            patch(
                "gsforge.ingest.ffmpeg.probe", return_value=self._mock_ffprobe_result()
            ),
            patch("gsforge.ingest.subprocess.Popen", self._mock_popen(tmp_path)),
            patch("gsforge.ingest.log_warning") as mock_warn,
        ):
            result = extract_frames(
                project=proj,
                input_path=video,
                num_images=100,  # well under ~240 available frames
                sequence_fps=30,  # non-default value — should trigger warning
            )

        # Exactly one warning: the sequence-fps-ignored warning
        mock_warn.assert_called_once()
        assert (
            "sequence-fps" in mock_warn.call_args[0][0].lower()
            or "sequence_fps" in mock_warn.call_args[0][0].lower()
        )

    def test_sequence_fps_used_for_image_sequence(self, tmp_path: Path) -> None:
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=3)
        proj = _make_gsproject(tmp_path)

        result = extract_frames(
            project=proj,
            input_path=frames[0],
            sequence_fps=48,
        )

        assert result.effective_fps == 48.0

    def test_unsupported_extension_exits(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "clip.avi"
        bad_file.touch()
        proj = _make_gsproject(tmp_path)

        with pytest.raises(SystemExit):
            extract_frames(project=proj, input_path=bad_file)

    def test_num_images_stored_in_project_json(self, tmp_path: Path) -> None:
        """num_images_requested must be persisted in project.json after ingest."""
        seq_dir = tmp_path / "seq"
        frames = _make_sequence(seq_dir, count=20)
        proj = _make_gsproject(tmp_path)

        extract_frames(project=proj, input_path=frames[0], num_images=7)

        from gsforge.project import GSProject

        proj_reloaded = GSProject.from_path(proj.root)
        assert proj_reloaded.meta.num_images_requested == 7

    def test_video_over_request_warns_and_uses_all(self, tmp_path: Path) -> None:
        """Requesting more images than video frames should warn and use all frames."""
        video = tmp_path / "clip.mp4"
        video.touch()
        proj = _make_gsproject(tmp_path)

        # 10-second 24fps clip → ~240 total frames; request 9999
        with (
            patch(
                "gsforge.ingest.ffmpeg.probe", return_value=self._mock_ffprobe_result()
            ),
            patch(
                "gsforge.ingest.subprocess.Popen",
                self._mock_popen(tmp_path, num_frames=5),
            ),
            patch("gsforge.ingest.log_warning") as mock_warn,
        ):
            result = extract_frames(
                project=proj,
                input_path=video,
                num_images=9999,
            )

        # Should have warned about over-request
        warning_calls = [str(c) for c in mock_warn.call_args_list]
        assert any("9999" in w for w in warning_calls)
