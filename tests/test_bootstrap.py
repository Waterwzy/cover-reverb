"""Environment bootstrap: conda detection, the venv fallback and argument parsing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import bootstrap_env
from bootstrap_env import (
    DEV_DEPS,
    RUNTIME_DEPS,
    build_parser,
    conda_env_python,
    find_conda,
    main,
    run_conda,
)

SKIP_ON_WINDOWS = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX-only path resolution"
)


class TestCondaDetection:
    def test_prefers_conda_exe_over_the_bat_shim(self, monkeypatch, tmp_path):
        """subprocess cannot run a .bat directly, so the .exe must win."""
        real = tmp_path / "conda.exe"
        shim = tmp_path / "conda.bat"
        real.write_text("")
        shim.write_text("")

        def fake_which(name):
            return {"conda.exe": str(real), "conda": str(shim)}.get(name)

        monkeypatch.setattr(bootstrap_env.shutil, "which", fake_which)
        assert find_conda() == str(real)

    def test_falls_back_to_the_shim_when_no_exe_exists(self, monkeypatch, tmp_path):
        shim = tmp_path / "conda.bat"
        shim.write_text("")
        monkeypatch.setattr(
            bootstrap_env.shutil, "which", lambda name: str(shim) if name == "conda" else None
        )
        assert find_conda() == str(shim)

    def test_returns_none_when_conda_is_absent(self, monkeypatch):
        monkeypatch.setattr(bootstrap_env.shutil, "which", lambda name: None)
        monkeypatch.setattr(bootstrap_env.Path, "home", classmethod(lambda cls: Path("/nonexistent")))
        assert find_conda() is None

    def test_finds_conda_on_this_machine_or_reports_none(self):
        assert find_conda() is None or Path(find_conda()).exists()


class TestRunConda:
    def test_returns_the_completed_process_on_success(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            assert cmd[1:] == ["env", "list", "--json"]
            return subprocess.CompletedProcess(cmd, 0, "{}", "")

        monkeypatch.setattr(bootstrap_env.subprocess, "run", fake_run)
        result = run_conda("conda", ["env", "list", "--json"])
        assert result is not None
        assert result.returncode == 0

    def test_returns_none_for_a_missing_executable(self, monkeypatch):
        def boom(*_args, **_kwargs):
            raise FileNotFoundError("no conda")

        monkeypatch.setattr(bootstrap_env.subprocess, "run", boom)
        assert run_conda("/usr/bin/conda", ["env", "list"]) is None

    def test_retries_through_a_shell_for_a_bat_shim_on_windows(self, monkeypatch):
        calls: list[object] = []

        def fake_run(cmd, **kwargs):
            calls.append(kwargs.get("shell", False))
            if not kwargs.get("shell"):
                raise OSError("cannot execute a batch file")
            return subprocess.CompletedProcess(cmd, 0, "{}", "")

        monkeypatch.setattr(bootstrap_env.subprocess, "run", fake_run)
        monkeypatch.setattr(bootstrap_env.os, "name", "nt")
        result = run_conda(r"C:\conda.bat", ["env", "list", "--json"])
        assert result is not None
        assert calls == [False, True]

    def test_does_not_retry_for_a_bat_shim_off_windows(self, monkeypatch):
        def fake_run(*_args, **_kwargs):
            raise OSError("nope")

        monkeypatch.setattr(bootstrap_env.subprocess, "run", fake_run)
        monkeypatch.setattr(bootstrap_env.os, "name", "posix")
        assert run_conda("/usr/bin/conda.bat", ["env", "list"]) is None


class TestCondaEnvPython:
    def _stub_env_list(self, monkeypatch, envs, returncode=0):
        def fake_run(cmd, **_kwargs):
            import json

            return subprocess.CompletedProcess(cmd, returncode, json.dumps({"envs": envs}), "")

        monkeypatch.setattr(bootstrap_env.subprocess, "run", fake_run)

    def test_returns_the_interpreter_of_a_matching_env(self, monkeypatch, tmp_path):
        env = tmp_path / "myenv"
        (env / "bin").mkdir(parents=True)
        python = env / "bin" / "python"
        python.write_text("")
        self._stub_env_list(monkeypatch, [str(env)])
        assert conda_env_python("conda", "myenv") == python

    def test_prefers_python_exe_when_present(self, monkeypatch, tmp_path):
        env = tmp_path / "myenv"
        env.mkdir()
        python = env / "python.exe"
        python.write_text("")
        self._stub_env_list(monkeypatch, [str(env)])
        assert conda_env_python("conda", "myenv") == python

    def test_returns_none_for_an_unknown_env(self, monkeypatch, tmp_path):
        env = tmp_path / "other"
        env.mkdir()
        self._stub_env_list(monkeypatch, [str(env)])
        assert conda_env_python("conda", "myenv") is None

    def test_returns_none_when_conda_fails(self, monkeypatch, tmp_path):
        self._stub_env_list(monkeypatch, [], returncode=1)
        assert conda_env_python("conda", "myenv") is None

    def test_returns_none_on_unparseable_output(self, monkeypatch):
        monkeypatch.setattr(
            bootstrap_env.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "not json", ""),
        )
        assert conda_env_python("conda", "myenv") is None

    def test_returns_none_when_the_interpreter_file_is_missing(self, monkeypatch, tmp_path):
        env = tmp_path / "myenv"
        env.mkdir()
        self._stub_env_list(monkeypatch, [str(env)])
        assert conda_env_python("conda", "myenv") is None

    def test_matches_the_env_by_directory_name_not_by_path(self, monkeypatch, tmp_path):
        nested = tmp_path / "a" / "b" / "wanted"
        (nested / "bin").mkdir(parents=True)
        (nested / "bin" / "python").write_text("")
        self._stub_env_list(monkeypatch, [str(tmp_path / "unrelated"), str(nested)])
        assert conda_env_python("conda", "wanted") == nested / "bin" / "python"


class TestVersionGuard:
    def test_version_guard_is_checked_before_creating_a_venv(self, monkeypatch, tmp_path, capsys):
        from collections import namedtuple

        VersionInfo = namedtuple("version_info", "major minor micro releaselevel serial")
        args = build_parser().parse_args(["--manager", "venv", "--env-name", "x"])
        monkeypatch.setattr(bootstrap_env.sys, "version_info", VersionInfo(3, 8, 0, "final", 0))
        monkeypatch.setattr(bootstrap_env, "VENV_DIR", tmp_path / "v")
        python, kind = bootstrap_env.setup_venv(args)
        assert python is None
        assert kind == ""
        assert "too old" in capsys.readouterr().out
        assert not (tmp_path / "v").exists()


class TestParserAndDeps:
    def test_default_manager_is_auto(self):
        assert build_parser().parse_args([]).manager == "auto"

    def test_manager_choices_are_enforced(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--manager", "docker"])

    def test_conda_flow_is_cancelled_when_conda_was_forced_but_absent(self, monkeypatch, capsys):
        monkeypatch.setattr(bootstrap_env, "find_conda", lambda: None)
        assert main(["--manager", "conda"]) == 1
        assert "not found" in capsys.readouterr().out

    def test_runtime_deps_are_pip_installable_names(self):
        assert all("gradio" not in dep or dep.startswith("gradio") for dep in RUNTIME_DEPS)
        for dep in RUNTIME_DEPS + DEV_DEPS:
            assert any(dep.startswith(name) for name in ("numpy", "scipy", "soundfile",
                                                         "gradio", "pytest", "ruff"))


class TestKindReporting:
    """kind drives the "next steps" message, so it must name the manager."""

    def test_conda_setup_reports_conda(self, monkeypatch, tmp_path):
        env = tmp_path / "cover-reverb"
        env.mkdir()
        (env / "python.exe").write_text("")
        monkeypatch.setattr(bootstrap_env, "find_conda", lambda: "conda")
        monkeypatch.setattr(bootstrap_env, "conda_env_python", lambda c, n: env / "python.exe")
        monkeypatch.setattr(bootstrap_env, "_pip_install", lambda *a, **k: 0)
        args = build_parser().parse_args([])
        python, kind = bootstrap_env.setup_conda(args)
        assert python == str(env / "python.exe")
        assert kind == "conda"

    def test_venv_setup_reports_venv(self, monkeypatch, tmp_path):
        venv_dir = tmp_path / "v"
        python = venv_dir / ("Scripts/python.exe" if bootstrap_env.os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True)
        python.write_text("")
        monkeypatch.setattr(bootstrap_env, "VENV_DIR", venv_dir)
        monkeypatch.setattr(bootstrap_env, "_pip_install", lambda *a, **k: 0)
        args = build_parser().parse_args(["--manager", "venv"])
        _, kind = bootstrap_env.setup_venv(args)
        assert kind == "venv"

    def test_next_steps_mention_conda_when_conda_was_used(self, monkeypatch, tmp_path, capsys):
        env = tmp_path / "cover-reverb"
        env.mkdir()
        (env / "python.exe").write_text("")
        monkeypatch.setattr(bootstrap_env, "find_conda", lambda: "conda")
        monkeypatch.setattr(bootstrap_env, "conda_env_python", lambda c, n: env / "python.exe")
        monkeypatch.setattr(bootstrap_env, "_pip_install", lambda *a, **k: 0)
        assert bootstrap_env.main(["--manager", "conda", "--env-name", "cover-reverb"]) == 0
        out = capsys.readouterr().out
        assert "conda activate cover-reverb" in out
        assert ".venv" not in out


class TestLauncherModule:
    def test_launcher_reuses_the_same_conda_lookup(self):
        import launch

        assert launch.find_conda() == find_conda()

    def test_launcher_env_name_is_overridable(self, monkeypatch):
        monkeypatch.setenv("COVER_REVERB_ENV", "custom-env")
        import importlib

        import launch

        reloaded = importlib.reload(launch)
        assert reloaded.ENV_NAME == "custom-env"
        monkeypatch.delenv("COVER_REVERB_ENV")
        importlib.reload(launch)

    @SKIP_ON_WINDOWS
    def test_venv_python_path_is_posix_style_off_windows(self):
        import launch

        assert launch.VENV_PYTHON.parts[-2:] == ("bin", "python")
