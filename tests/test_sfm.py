"""
tests/test_sfm.py — Unit tests for gsforge.sfm sparse-model selection

Coverage
--------
* enumerate_sparse_models()   — finds numbered sub-dirs with COLMAP files,
                                ignores non-numeric dirs, ignores empty dirs
* analyze_sparse_model()      — parses model_analyzer output, handles failures
* select_best_sparse_model()  — picks model with most images, single-model
                                fast path, graceful fallback on all-fail

Design notes
------------
* No real COLMAP binary is invoked.  ``subprocess.run`` is mocked to return
  controlled stdout/stderr so the tests run without COLMAP installed.
* All tests use ``tmp_path`` for full isolation.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gsforge.sfm import (
    ColmapCapabilityProbe,
    analyze_sparse_model,
    enumerate_sparse_models,
    probe_colmap_capabilities,
    run_mapper,
    run_sfm,
    select_best_sparse_model,
)
from gsforge.utils import DEFAULT_SFM_METHOD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sparse_model(parent: Path, name: str, num_images: int = 10) -> Path:
    """Create a minimal COLMAP sparse sub-model directory.

    Writes a stub ``images.bin`` with the correct 8-byte uint64 header so
    ``count_registered_cameras`` can read it.
    """
    model_dir = parent / name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Write a minimal images.bin: 8-byte little-endian uint64 = num_images,
    # followed by nothing (count_registered_cameras only reads the header).
    images_bin = model_dir / "images.bin"
    images_bin.write_bytes(struct.pack("<Q", num_images))

    # Also write stub cameras.bin and points3D.bin so the dir is "valid"
    (model_dir / "cameras.bin").write_bytes(struct.pack("<Q", 1))
    (model_dir / "points3D.bin").write_bytes(struct.pack("<Q", 0))

    return model_dir


def _make_colmap_bin() -> Path:
    """Return a fake colmap binary path (does not need to exist for mocked tests)."""
    return Path("/fake/colmap")


COLMAP_4_1_1_GLOBAL_MAPPER_HELP = """COLMAP 4.1.1 (Commit a0d785f on 2026-07-17 with CUDA)
Options:
  -h [ --help ]
--GlobalMapper.gp_use_gpu arg (=1)
--GlobalMapper.gp_gpu_index arg (=-1)
--GlobalMapper.ba_ceres_use_gpu arg (=1)
--GlobalMapper.ba_ceres_gpu_index arg (=-1)
"""


def _capture_mapper_command(method: str) -> list[str]:
    captured: list[list[str]] = []

    def _capture(cmd, **kwargs):
        captured.append(cmd)
        result = MagicMock()
        result.returncode = 0
        return result

    with patch("gsforge.sfm.subprocess.run", side_effect=_capture):
        run_mapper(
            _make_colmap_bin(),
            Path("/project/database.db"),
            Path("/project/preprocess"),
            Path("/project/sfm/sparse"),
            method,  # type: ignore[arg-type]
        )

    assert len(captured) == 1
    return captured[0]


def _assert_shared_mapper_paths(cmd: list[str]) -> None:
    assert cmd[cmd.index("--database_path") + 1] == str(Path("/project/database.db"))
    assert cmd[cmd.index("--image_path") + 1] == str(Path("/project/preprocess"))
    assert cmd[cmd.index("--output_path") + 1] == str(Path("/project/sfm/sparse"))


# ---------------------------------------------------------------------------
# run_mapper dispatch
# ---------------------------------------------------------------------------


class TestRunMapper:
    def test_glomap_uses_global_mapper_and_global_options(self) -> None:
        cmd = _capture_mapper_command("glomap")

        assert cmd[0] == str(_make_colmap_bin())
        assert cmd[1] == "global_mapper"
        _assert_shared_mapper_paths(cmd)
        assert "--GlobalMapper.ba_refine_focal_length" in cmd
        assert "--GlobalMapper.ba_refine_principal_point" in cmd
        assert "--GlobalMapper.ba_refine_extra_params" in cmd
        assert not any(arg.startswith("--Mapper.") for arg in cmd)
        assert "--Mapper.mapper_type" not in cmd
        assert "--GlobalMapper.gp_use_gpu" in cmd
        assert "--GlobalMapper.ba_ceres_use_gpu" in cmd

    def test_colmap_uses_incremental_mapper_and_mapper_options(self) -> None:
        cmd = _capture_mapper_command("colmap")

        assert cmd[0] == str(_make_colmap_bin())
        assert cmd[1] == "mapper"
        _assert_shared_mapper_paths(cmd)
        assert "--Mapper.ba_refine_focal_length" in cmd
        assert "--Mapper.ba_refine_principal_point" in cmd
        assert "--Mapper.ba_refine_extra_params" in cmd
        assert not any(arg.startswith("--GlobalMapper.") for arg in cmd)
        assert "--Mapper.mapper_type" not in cmd

    def test_glomap_is_the_full_pipeline_default_method(self) -> None:
        assert DEFAULT_SFM_METHOD == "glomap"
        assert inspect.signature(run_sfm).parameters["method"].default == "glomap"


class TestColmapCapabilityProbe:
    def test_help_is_authoritative_when_version_is_unsupported(self) -> None:
        calls: list[list[str]] = []

        def _probe(cmd, **kwargs):
            calls.append(cmd)
            if cmd[-1] == "--version":
                return MagicMock(
                    returncode=1, stdout="", stderr="--version not recognized"
                )
            if cmd[-1] == "help":
                return MagicMock(
                    returncode=0,
                    stdout="COLMAP 4.0.2\nAvailable commands:\n global_mapper\n mapper",
                    stderr="",
                )
            if cmd[-1] == "version":
                return MagicMock(returncode=0, stdout="COLMAP 4.0.2", stderr="")
            return MagicMock(returncode=0, stdout="Options:\n --help", stderr="")

        with patch("gsforge.sfm.subprocess.run", side_effect=_probe):
            result = probe_colmap_capabilities(_make_colmap_bin())

        assert result.binary_available is True
        assert result.version == "4.0.2"
        assert result.version_status == "unsupported"
        assert result.commands["global_mapper"] == "supported"
        assert result.raw_evidence["--version"]["returncode"] == 1
        assert ["--version"] in [call[1:] for call in calls]
        assert ["version"] in [call[1:] for call in calls]
        assert ["-h"] in [call[1:] for call in calls]

    def test_unusable_help_marks_binary_unavailable(self) -> None:
        with patch(
            "gsforge.sfm.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="colmap", timeout=10),
        ):
            result = probe_colmap_capabilities(_make_colmap_bin())

        assert result.binary_available is False
        assert result.version_status == "unavailable"

    def test_empty_successful_help_is_unusable(self) -> None:
        def _probe(cmd, **kwargs):
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("gsforge.sfm.subprocess.run", side_effect=_probe):
            result = probe_colmap_capabilities(_make_colmap_bin())

        assert result.binary_available is False
        assert result.version_status == "unavailable"

    def test_cpu_fallback_warning_is_reported_for_glomap(self) -> None:
        warning = (
            "Requested to use GPU for bundle adjustment, but Ceres was compiled "
            "without CUDA support. Falling back to CPU-based dense solvers."
        )
        result = MagicMock(returncode=0, stdout=warning, stderr="")
        with (
            patch("gsforge.sfm.subprocess.run", return_value=result),
            patch("gsforge.sfm.log_warning") as log_warning_mock,
        ):
            run_mapper(
                _make_colmap_bin(),
                Path("/project/database.db"),
                Path("/project/preprocess"),
                Path("/project/sfm/sparse"),
                "glomap",
            )

        log_warning_mock.assert_called_once()
        assert "CPU bundle adjustment" in log_warning_mock.call_args.args[0]

    def test_normal_ceres_report_does_not_emit_cpu_fallback_warning(self) -> None:
        result = MagicMock(
            returncode=0,
            stdout=(
                "Ceres Solver Report: Iterations: 52, Initial cost: "
                "5.3e+06, Final cost: 2.7e+01, Termination: CONVERGENCE"
            ),
            stderr="",
        )
        with (
            patch("gsforge.sfm.subprocess.run", return_value=result),
            patch("gsforge.sfm.log_warning") as log_warning_mock,
        ):
            run_mapper(
                _make_colmap_bin(),
                Path("/project/database.db"),
                Path("/project/preprocess"),
                Path("/project/sfm/sparse"),
                "glomap",
            )

        log_warning_mock.assert_not_called()

    def test_colmap_411_global_mapper_gpu_contract_is_captured(self) -> None:
        def _probe(cmd, **kwargs):
            if cmd[1] == "help":
                return MagicMock(
                    returncode=0,
                    stdout=(
                        "COLMAP 4.1.1 -- Structure-from-Motion and Multi-View Stereo\n"
                        "Available commands:\n mapper\n global_mapper\n"
                    ),
                    stderr="",
                )
            if cmd[1] == "global_mapper":
                return MagicMock(
                    returncode=0,
                    stdout=COLMAP_4_1_1_GLOBAL_MAPPER_HELP,
                    stderr="",
                )
            if cmd[1] == "-h":
                return MagicMock(
                    returncode=0,
                    stdout=(
                        "COLMAP 4.1.1\nAvailable commands:\n mapper\n global_mapper\n"
                    ),
                    stderr="",
                )
            return MagicMock(returncode=0, stdout="COLMAP 4.1.1", stderr="")

        with patch("gsforge.sfm.subprocess.run", side_effect=_probe):
            result = probe_colmap_capabilities(_make_colmap_bin())

        assert result.version == "4.1.1"
        assert result.commands["global_mapper"] == "supported"
        assert result.commands["-h"] == "supported"
        assert (
            "--GlobalMapper.ba_ceres_use_gpu"
            in result.raw_evidence["global_mapper"]["stdout"]
        )
        assert "Caspar" not in result.raw_evidence["global_mapper"]["stdout"]


class TestRunSfmProbeIntegration:
    def _project(self, tmp_path: Path) -> MagicMock:
        project = MagicMock()
        project.root = tmp_path
        project.preprocess_dir = tmp_path / "preprocess"
        project.sfm_dir = tmp_path / "sfm"
        project.logs_dir = tmp_path / "logs"
        project.preprocess_dir.mkdir()
        (project.preprocess_dir / "frame_000001.png").write_bytes(b"frame")
        return project

    def _probe(self) -> ColmapCapabilityProbe:
        return ColmapCapabilityProbe(
            binary_available=True,
            version="4.1.1",
            version_status="unsupported",
            commands={"help": "supported", "global_mapper": "supported"},
            diagnostic="help is authoritative",
        )

    def test_rejected_version_metadata_does_not_stop_pipeline(
        self, tmp_path: Path
    ) -> None:
        project = self._project(tmp_path)
        sparse = tmp_path / "sfm" / "sparse" / "0"
        sparse.mkdir(parents=True)
        with (
            patch("gsforge.sfm.find_colmap_binary", return_value=Path("colmap")),
            patch("gsforge.sfm.probe_colmap_capabilities", return_value=self._probe()),
            patch("gsforge.sfm.run_feature_extraction"),
            patch("gsforge.sfm.run_feature_matching"),
            patch("gsforge.sfm.run_mapper"),
            patch("gsforge.sfm.select_best_sparse_model", return_value=sparse),
            patch("gsforge.sfm.count_registered_cameras", return_value=1),
            patch("gsforge.sfm.log_info"),
        ):
            result = run_sfm(project, method="glomap")

        assert result.status == "completed"
        evidence_path = project.logs_dir / "colmap-capability-probe.json"
        assert evidence_path.exists()
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert evidence["version_status"] == "unsupported"
        assert "raw_evidence" in evidence
        project.update_after_sfm.assert_called_once()
        assert project.update_after_sfm.call_args.kwargs["sfm_status"] == "completed"

    def test_mapper_failure_persists_failed_state_without_switching_method(
        self, tmp_path: Path
    ) -> None:
        project = self._project(tmp_path)
        with (
            patch("gsforge.sfm.find_colmap_binary", return_value=Path("colmap")),
            patch("gsforge.sfm.probe_colmap_capabilities", return_value=self._probe()),
            patch("gsforge.sfm.run_feature_extraction"),
            patch("gsforge.sfm.run_feature_matching"),
            patch("gsforge.sfm.run_mapper", side_effect=SystemExit(1)),
            patch("gsforge.sfm.log_info"),
        ):
            with pytest.raises(SystemExit):
                run_sfm(project, method="glomap")

        update = project.update_after_sfm.call_args.kwargs
        assert update["sfm_method"] == "glomap"
        assert update["sfm_status"] == "failed"
        assert update["camera_count"] == 0

    def test_unusable_probe_persists_failed_state(self, tmp_path: Path) -> None:
        project = self._project(tmp_path)
        probe = ColmapCapabilityProbe(
            binary_available=False,
            version=None,
            version_status="unavailable",
            diagnostic="unusable help",
        )
        with (
            patch("gsforge.sfm.find_colmap_binary", return_value=Path("colmap")),
            patch("gsforge.sfm.probe_colmap_capabilities", return_value=probe),
            patch("gsforge.sfm.log_info"),
            patch("gsforge.sfm.run_feature_extraction") as extract,
            patch("gsforge.sfm.run_feature_matching") as match,
            patch("gsforge.sfm.run_mapper") as mapper,
            patch("gsforge.sfm.log_error", side_effect=SystemExit(1)),
        ):
            with pytest.raises(SystemExit):
                run_sfm(project, method="glomap")

        update = project.update_after_sfm.call_args.kwargs
        assert update["sfm_status"] == "failed"
        assert update["camera_count"] == 0
        extract.assert_not_called()
        match.assert_not_called()
        mapper.assert_not_called()

    def test_probe_preserves_unavailable_subcommand_state(self) -> None:
        def _probe(cmd, **kwargs):
            if cmd[1] == "mapper":
                raise FileNotFoundError("mapper probe failed")
            if cmd[1] in {"help", "-h"}:
                return MagicMock(
                    returncode=0,
                    stdout="COLMAP 4.1.1\nAvailable commands:\n mapper\n global_mapper\n",
                    stderr="",
                )
            return MagicMock(returncode=0, stdout="COLMAP 4.1.1", stderr="")

        with patch("gsforge.sfm.subprocess.run", side_effect=_probe):
            result = probe_colmap_capabilities(_make_colmap_bin())

        assert result.binary_available is True
        assert result.commands["mapper"] == "unavailable"


# ---------------------------------------------------------------------------
# enumerate_sparse_models
# ---------------------------------------------------------------------------


class TestEnumerateSparseModels:
    def test_finds_numbered_subdirs_with_colmap_files(self, tmp_path: Path) -> None:
        sparse = tmp_path / "sparse"
        _make_sparse_model(sparse, "0", num_images=5)
        _make_sparse_model(sparse, "1", num_images=10)

        result = enumerate_sparse_models(sparse)

        assert len(result) == 2
        assert result[0].name == "0"
        assert result[1].name == "1"

    def test_sorted_numerically(self, tmp_path: Path) -> None:
        sparse = tmp_path / "sparse"
        _make_sparse_model(sparse, "2")
        _make_sparse_model(sparse, "0")
        _make_sparse_model(sparse, "1")

        result = enumerate_sparse_models(sparse)

        assert [m.name for m in result] == ["0", "1", "2"]

    def test_ignores_non_numeric_dirs(self, tmp_path: Path) -> None:
        sparse = tmp_path / "sparse"
        _make_sparse_model(sparse, "0")
        # Non-numeric directory — should be ignored
        bad = sparse / "backup"
        bad.mkdir()
        (bad / "cameras.bin").write_bytes(b"\x00" * 8)

        result = enumerate_sparse_models(sparse)

        assert len(result) == 1
        assert result[0].name == "0"

    def test_ignores_empty_numeric_dirs(self, tmp_path: Path) -> None:
        sparse = tmp_path / "sparse"
        _make_sparse_model(sparse, "0")
        # Empty numeric directory — no COLMAP files
        (sparse / "1").mkdir()

        result = enumerate_sparse_models(sparse)

        assert len(result) == 1
        assert result[0].name == "0"

    def test_returns_empty_when_sparse_parent_missing(self, tmp_path: Path) -> None:
        result = enumerate_sparse_models(tmp_path / "nonexistent")
        assert result == []

    def test_returns_empty_when_no_valid_models(self, tmp_path: Path) -> None:
        sparse = tmp_path / "sparse"
        sparse.mkdir()
        # Only empty dirs
        (sparse / "0").mkdir()
        (sparse / "1").mkdir()

        result = enumerate_sparse_models(sparse)
        assert result == []

    def test_accepts_txt_format_models(self, tmp_path: Path) -> None:
        sparse = tmp_path / "sparse"
        model = sparse / "0"
        model.mkdir(parents=True)
        # Text format instead of binary
        (model / "cameras.txt").write_text("# cameras\n")
        (model / "images.txt").write_text("# images\n")
        (model / "points3D.txt").write_text("# points\n")

        result = enumerate_sparse_models(sparse)

        assert len(result) == 1
        assert result[0].name == "0"


# ---------------------------------------------------------------------------
# analyze_sparse_model
# ---------------------------------------------------------------------------


class TestAnalyzeSparseModel:
    def _make_run_result(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        mock = MagicMock()
        mock.stdout = stdout
        mock.stderr = stderr
        mock.returncode = returncode
        return mock

    def test_parses_registered_images_from_stderr(self, tmp_path: Path) -> None:
        """COLMAP 4.x writes log lines to stderr in the format used by glog."""
        model_dir = tmp_path / "0"
        model_dir.mkdir()
        colmap_bin = _make_colmap_bin()

        # Realistic COLMAP 4.x glog output (written to stderr)
        output = (
            "I20260406 22:32:40.215421 12804 model.cc:436] Cameras: 1\n"
            "I20260406 22:32:40.215519 12804 model.cc:438] Frames: 398\n"
            "I20260406 22:32:40.215553 12804 model.cc:440] Images: 398\n"
            "I20260406 22:32:40.215594 12804 model.cc:441] Registered images: 398\n"
            "I20260406 22:32:40.215626 12804 model.cc:443] Points: 100165\n"
        )
        with patch(
            "gsforge.sfm.subprocess.run",
            return_value=self._make_run_result(stderr=output),
        ):
            count = analyze_sparse_model(colmap_bin, model_dir)

        assert count == 398

    def test_registered_images_preferred_over_bare_images(self, tmp_path: Path) -> None:
        """When both 'Images:' and 'Registered images:' appear, use Registered images."""
        model_dir = tmp_path / "0"
        model_dir.mkdir()
        colmap_bin = _make_colmap_bin()

        # Images: 500 (total in DB), Registered images: 312 (actually localised)
        output = "Images: 500\nRegistered images: 312\n"
        with patch(
            "gsforge.sfm.subprocess.run",
            return_value=self._make_run_result(stdout=output),
        ):
            count = analyze_sparse_model(colmap_bin, model_dir)

        assert count == 312

    def test_fallback_to_bare_images_when_no_registered_line(
        self, tmp_path: Path
    ) -> None:
        """Older COLMAP versions may not print 'Registered images:' — fall back."""
        model_dir = tmp_path / "0"
        model_dir.mkdir()
        colmap_bin = _make_colmap_bin()

        output = "Cameras: 1\nImages: 87\nPoints: 5000\n"
        with patch(
            "gsforge.sfm.subprocess.run",
            return_value=self._make_run_result(stdout=output),
        ):
            count = analyze_sparse_model(colmap_bin, model_dir)

        assert count == 87

    def test_returns_none_when_output_unparseable(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "0"
        model_dir.mkdir()
        colmap_bin = _make_colmap_bin()

        with patch(
            "gsforge.sfm.subprocess.run",
            return_value=self._make_run_result(stdout="No useful output here"),
        ):
            count = analyze_sparse_model(colmap_bin, model_dir)

        assert count is None

    def test_returns_none_on_timeout(self, tmp_path: Path) -> None:
        import subprocess

        model_dir = tmp_path / "0"
        model_dir.mkdir()
        colmap_bin = _make_colmap_bin()

        with patch(
            "gsforge.sfm.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="colmap", timeout=60),
        ):
            count = analyze_sparse_model(colmap_bin, model_dir)

        assert count is None

    def test_returns_none_on_exception(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "0"
        model_dir.mkdir()
        colmap_bin = _make_colmap_bin()

        with patch(
            "gsforge.sfm.subprocess.run",
            side_effect=OSError("binary not found"),
        ):
            count = analyze_sparse_model(colmap_bin, model_dir)

        assert count is None

    def test_case_insensitive_parsing(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "0"
        model_dir.mkdir()
        colmap_bin = _make_colmap_bin()

        # Case variation
        output = "REGISTERED IMAGES: 42\n"
        with patch(
            "gsforge.sfm.subprocess.run",
            return_value=self._make_run_result(stdout=output),
        ):
            count = analyze_sparse_model(colmap_bin, model_dir)

        assert count == 42

    def test_uses_path_flag_not_input_path(self, tmp_path: Path) -> None:
        """Verify the subprocess call uses --path (COLMAP 4.x) not --input_path."""
        model_dir = tmp_path / "0"
        model_dir.mkdir()
        colmap_bin = _make_colmap_bin()

        captured_cmd: list[list[str]] = []

        def _capture(cmd, **kwargs):
            captured_cmd.append(cmd)
            mock = MagicMock()
            mock.stdout = "Registered images: 10\n"
            mock.stderr = ""
            mock.returncode = 0
            return mock

        with patch("gsforge.sfm.subprocess.run", side_effect=_capture):
            analyze_sparse_model(colmap_bin, model_dir)

        assert len(captured_cmd) == 1
        cmd = captured_cmd[0]
        assert "--path" in cmd
        assert "--input_path" not in cmd


# ---------------------------------------------------------------------------
# select_best_sparse_model
# ---------------------------------------------------------------------------


class TestSelectBestSparseModel:
    def _mock_analyzer(self, counts: dict[str, int | None]):
        """Return a side_effect function for analyze_sparse_model.

        ``counts`` maps model directory name (e.g. "0") to the image count
        that the mock should return (or None to simulate failure).
        """

        def _side_effect(colmap_bin: Path, model_dir: Path):
            return counts.get(model_dir.name)

        return _side_effect

    def test_single_model_returned_directly(self, tmp_path: Path) -> None:
        sparse = tmp_path / "sparse"
        _make_sparse_model(sparse, "0", num_images=50)
        colmap_bin = _make_colmap_bin()

        # With a single model, no analyzer call should be needed
        with patch("gsforge.sfm.analyze_sparse_model") as mock_analyze:
            result = select_best_sparse_model(colmap_bin, sparse)

        mock_analyze.assert_not_called()
        assert result.name == "0"

    def test_selects_model_with_most_images(self, tmp_path: Path) -> None:
        sparse = tmp_path / "sparse"
        _make_sparse_model(sparse, "0", num_images=50)
        _make_sparse_model(sparse, "1", num_images=200)
        _make_sparse_model(sparse, "2", num_images=80)
        colmap_bin = _make_colmap_bin()

        with patch(
            "gsforge.sfm.analyze_sparse_model",
            side_effect=self._mock_analyzer({"0": 50, "1": 200, "2": 80}),
        ):
            result = select_best_sparse_model(colmap_bin, sparse)

        assert result.name == "1"

    def test_fallback_to_first_when_all_analysis_fails(self, tmp_path: Path) -> None:
        sparse = tmp_path / "sparse"
        _make_sparse_model(sparse, "0")
        _make_sparse_model(sparse, "1")
        colmap_bin = _make_colmap_bin()

        with patch(
            "gsforge.sfm.analyze_sparse_model",
            side_effect=self._mock_analyzer({"0": None, "1": None}),
        ):
            result = select_best_sparse_model(colmap_bin, sparse)

        # Falls back to first model
        assert result.name == "0"

    def test_partial_failure_still_selects_best(self, tmp_path: Path) -> None:
        """If analysis fails for some models, use the ones that succeeded."""
        sparse = tmp_path / "sparse"
        _make_sparse_model(sparse, "0")
        _make_sparse_model(sparse, "1")
        _make_sparse_model(sparse, "2")
        colmap_bin = _make_colmap_bin()

        with patch(
            "gsforge.sfm.analyze_sparse_model",
            side_effect=self._mock_analyzer({"0": None, "1": 150, "2": None}),
        ):
            result = select_best_sparse_model(colmap_bin, sparse)

        assert result.name == "1"

    def test_no_models_returns_sparse_0_path(self, tmp_path: Path) -> None:
        """When no sub-models exist, return sparse/0/ as a fallback path."""
        sparse = tmp_path / "sparse"
        sparse.mkdir()
        colmap_bin = _make_colmap_bin()

        result = select_best_sparse_model(colmap_bin, sparse)

        assert result == sparse / "0"

    def test_two_models_equal_count_picks_first_found(self, tmp_path: Path) -> None:
        """Tie-breaking: when counts are equal, the first model (lowest index) wins."""
        sparse = tmp_path / "sparse"
        _make_sparse_model(sparse, "0")
        _make_sparse_model(sparse, "1")
        colmap_bin = _make_colmap_bin()

        with patch(
            "gsforge.sfm.analyze_sparse_model",
            side_effect=self._mock_analyzer({"0": 100, "1": 100}),
        ):
            result = select_best_sparse_model(colmap_bin, sparse)

        # Both have 100 images; model "0" is evaluated first and sets best_count=100,
        # model "1" is NOT strictly greater so "0" remains the winner.
        assert result.name == "0"
