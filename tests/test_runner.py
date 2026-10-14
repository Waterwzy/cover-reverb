"""Command line surface: argument parsing and the self-test path."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from srv.runner import (
    REQUIRED,
    _synthetic_reference,
    build_parser,
    check_env,
    find_free_port,
    main,
    selftest,
)


class TestParser:
    def test_defaults(self):
        args = build_parser().parse_args([])
        assert args.host == "127.0.0.1"
        assert args.port == 7860
        assert args.strategy is None
        assert args.batch is False

    def test_port_is_an_int(self):
        assert build_parser().parse_args(["--port", "8123"]).port == 8123

    def test_strategy_choices_are_enforced(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--strategy", "not_a_strategy"])

    @pytest.mark.parametrize("flag", ["--check", "--selftest", "--batch", "--setup", "--share"])
    def test_flags_parse(self, flag):
        assert getattr(build_parser().parse_args([flag]), flag.strip("-").replace("-", "_"))

    def test_roots_and_out(self):
        args = build_parser().parse_args(["--roots", "/music", "--out", "/tmp/x"])
        assert args.roots == "/music"
        assert args.out == "/tmp/x"


class TestFindFreePort:
    def test_returns_the_start_when_free(self):
        assert find_free_port("127.0.0.1", 7860) is not None

    def test_skips_a_bound_port(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            busy = sock.getsockname()[1]
            found = find_free_port("127.0.0.1", busy, tries=5)
        assert found is not None
        assert found != busy

    def test_returns_none_when_the_range_is_exhausted(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            assert find_free_port("127.0.0.1", port, tries=1) is None


class TestSyntheticReference:
    def test_shapes(self):
        dry, wet, rir = _synthetic_reference(44100, 2.0, seed=1)
        assert dry.ndim == 1
        assert wet.shape == (len(dry), 2)
        assert rir.shape[1] == 2

    def test_is_normalised(self):
        dry, wet, _ = _synthetic_reference(44100, 2.0, seed=2)
        assert np.max(np.abs(dry)) == pytest.approx(1.0, abs=1e-6)
        # wet = dry + reverb, so its peak may exceed the dry peak slightly
        assert np.max(np.abs(wet)) > 0.9

    def test_is_deterministic(self):
        a = _synthetic_reference(44100, 1.0, seed=3)[0]
        b = _synthetic_reference(44100, 1.0, seed=3)[0]
        assert np.allclose(a, b)

    def test_wet_differs_from_dry(self):
        dry, wet, _ = _synthetic_reference(44100, 2.0, seed=4)
        assert not np.allclose(dry, wet[:, 0])


class TestCheckEnv:
    def test_passes_in_this_environment(self):
        assert check_env(autofix=False) is True

    def test_required_packages_are_named(self):
        names = {module for module, _ in REQUIRED}
        assert {"numpy", "scipy", "soundfile", "gradio"} <= names


class TestSelftest:
    def test_runs_and_writes_artifacts(self, tmp_path, capsys):
        args = build_parser().parse_args(["--selftest", "--out", str(tmp_path)])
        assert selftest(args) == 0
        assert (tmp_path / "selftest_wet_vocal.wav").is_file()
        assert (tmp_path / "selftest_rir.wav").is_file()
        assert "Self-test passed" in capsys.readouterr().out

    def test_honours_the_strategy_flag(self, tmp_path):
        args = build_parser().parse_args(
            ["--selftest", "--out", str(tmp_path), "--strategy", "G0_extreme_dry"]
        )
        assert selftest(args) == 0
        data, _ = sf.read(str(tmp_path / "selftest_rir.wav"), always_2d=True)
        assert len(data) > 0


class TestMain:
    def test_check_mode_exits_zero(self, capsys):
        assert main(["--check"]) == 0
        assert "Environment check finished" in capsys.readouterr().out

    def test_selftest_via_main(self, tmp_path):
        assert main(["--selftest", "--out", str(tmp_path)]) == 0

    def test_batch_without_material_reports_failure(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("COVER_REVERB_SCAN", str(tmp_path))
        assert main(["--batch", "--out", str(tmp_path / "out")]) == 1
        assert "no stem folders" in capsys.readouterr().out
