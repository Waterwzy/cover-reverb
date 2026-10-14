"""Thin launcher used by the platform scripts.

Keeps the shell/batch wrappers to pure path resolution: the interactive
"pick an interpreter, bootstrap if needed, then run" flow lives here where it
can be written once and tested.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = REPO_ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
ENV_NAME = os.environ.get("COVER_REVERB_ENV", "cover-reverb")


def find_conda() -> str | None:
    """Locate conda on PATH or in the common install locations.

    Prefers the real ``conda.exe`` over the ``conda.BAT`` shim, because
    ``subprocess`` cannot execute a batch file directly on Windows.
    """
    import shutil

    found = shutil.which("conda.exe") or shutil.which("conda")
    if found:
        return found
    candidates = [
        Path.home() / "anaconda3" / "Scripts" / "conda.exe",
        Path.home() / "miniconda3" / "Scripts" / "conda.exe",
        Path.home() / "anaconda3" / "bin" / "conda",
        Path.home() / "miniconda3" / "bin" / "conda",
        Path("/opt/conda/bin/conda"),
        Path("C:/ProgramData/anaconda3/Scripts/conda.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _conda_env_list(conda: str) -> str | None:
    """Run ``conda env list --json``, working around the Windows .bat shim."""
    cmd = [conda, "env", "list", "--json"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        if os.name != "nt" or not conda.lower().endswith((".bat", ".cmd")):
            return None
        try:
            return subprocess.run(
                subprocess.list2cmdline(cmd), capture_output=True, text=True,
                check=False, shell=True,
            ).stdout
        except OSError:
            return None


def find_conda_python() -> Path | None:
    """Interpreter of the project's conda env, if it has been created."""
    import json

    conda = find_conda()
    if conda is None:
        return None
    stdout = _conda_env_list(conda)
    if not stdout:
        return None
    try:
        envs = json.loads(stdout).get("envs", [])
    except json.JSONDecodeError:
        return None
    for env_path in envs:
        base = Path(env_path)
        if base.name != ENV_NAME:
            continue
        for rel in ("python.exe", "bin/python"):
            candidate = base / rel
            if candidate.exists():
                return candidate
    return None


def environment_ready(python: Path) -> bool:
    """True when this interpreter can actually run the app.

    A conda env or venv can exist while still missing its dependencies -- for
    instance when an earlier bootstrap run was interrupted. Reusing such an
    interpreter produces a confusing ImportError, so verify before trusting it.
    """
    probe = "import numpy, scipy, soundfile, gradio, srv"
    try:
        return subprocess.run(
            [str(python), "-c", probe],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
        ).returncode == 0
    except OSError:
        return False


def resolve_python() -> Path:
    """Return an interpreter that can run the app, bootstrapping one if needed."""
    explicit = os.environ.get("COVER_REVERB_PYTHON")
    if explicit and Path(explicit).exists():
        return Path(explicit)

    for candidate in (find_conda_python(), VENV_PYTHON if VENV_PYTHON.exists() else None):
        if candidate is not None and environment_ready(candidate):
            return candidate

    print()
    print("  No usable environment found - running bootstrap_env.py ...")
    print()
    if subprocess.run([sys.executable, str(REPO_ROOT / "bootstrap_env.py"), "--with-dev"]).returncode:
        raise SystemExit("environment setup failed")

    for candidate in (find_conda_python(), VENV_PYTHON if VENV_PYTHON.exists() else None):
        if candidate is not None and environment_ready(candidate):
            return candidate
    raise SystemExit("environment was created but is still not usable")


def main() -> int:
    python = resolve_python()
    print(f"  python: {python}")
    print()
    return subprocess.run(
        [str(python), "-u", "-m", "srv.runner", *sys.argv[1:]], check=False
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
