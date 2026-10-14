"""Create and populate the runtime environment.

Prefers conda when it is available, otherwise falls back to a plain ``venv``.
The runtime dependencies (numpy, scipy, soundfile, gradio) are all pip-installable
on every platform, so no compiled toolchain is required either way.

Usage::

    python bootstrap_env.py                # auto-detect conda, else venv
    python bootstrap_env.py --manager venv # force a venv
    python bootstrap_env.py --manager conda --env-name my-env
    python bootstrap_env.py --with-dev     # also install pytest and ruff
    python bootstrap_env.py --recreate     # wipe and rebuild
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"
DEFAULT_ENV_NAME = "cover-reverb"

RUNTIME_DEPS = ["numpy>=1.23", "scipy>=1.9", "soundfile>=0.12", "gradio>=4.44"]
DEV_DEPS = ["pytest>=7.4", "pytest-cov>=4.1", "ruff>=0.6"]


def log(message: str) -> None:
    print(f"  {message}")


def run(cmd: list[str], **kwargs) -> int:
    log(" ".join(str(part) for part in cmd))
    try:
        return subprocess.run(cmd, check=False, **kwargs).returncode
    except FileNotFoundError:
        return 127
    except OSError as exc:
        log(f"could not run that command: {exc}")
        return 127


def run_conda(conda: str, args: list[str]) -> subprocess.CompletedProcess | None:
    """Invoke conda, transparently handling the ``conda.bat`` shim on Windows.

    ``shutil.which("conda")`` resolves to ``conda.BAT`` on Windows, and
    ``subprocess`` cannot execute a batch file without a shell. Without this the
    probe fails silently and the caller wrongly concludes that no conda
    environment exists.
    """
    cmd = [conda, *args]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)
    except (FileNotFoundError, OSError):
        if os.name != "nt" or not conda.lower().endswith((".bat", ".cmd")):
            return None
        try:
            return subprocess.run(
                subprocess.list2cmdline(cmd), capture_output=True, text=True,
                check=False, shell=True,
            )
        except OSError:
            return None


def _pip_install(python: str, packages: list[str], mirror: str | None) -> int:
    cmd = [python, "-m", "pip", "install", "--upgrade"]
    if mirror:
        cmd += ["-i", mirror]
    cmd += packages
    return run(cmd)


def find_conda() -> str | None:
    """Locate ``conda`` on PATH or in the usual install locations.

    Prefers the real ``conda.exe`` over the ``conda.BAT`` shim so that
    ``subprocess`` can invoke it directly.
    """
    found = shutil.which("conda.exe") or shutil.which("conda")
    if found:
        return found
    candidates = [
        Path.home() / "anaconda3" / "Scripts" / "conda.exe",
        Path.home() / "miniconda3" / "Scripts" / "conda.exe",
        Path.home() / "anaconda3" / "bin" / "conda",
        Path.home() / "miniconda3" / "bin" / "conda",
        Path("/opt/conda/bin/conda"),
        Path("/usr/local/Caskroom/miniconda/base/bin/conda"),
        Path("C:/ProgramData/anaconda3/Scripts/conda.exe"),
        Path("C:/ProgramData/miniconda3/Scripts/conda.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def conda_env_python(conda: str, env_name: str) -> Path | None:
    """Return the interpreter path of a conda env, if it exists."""
    import json

    result = run_conda(conda, ["env", "list", "--json"])
    if result is None or result.returncode != 0:
        return None
    try:
        envs = json.loads(result.stdout).get("envs", [])
    except (json.JSONDecodeError, AttributeError):
        return None
    for env_path in envs:
        base = Path(env_path)
        if base.name == env_name:
            for rel in ("python.exe", "bin/python"):
                candidate = base / rel
                if candidate.exists():
                    return candidate
    return None


def setup_conda(args) -> tuple[str | None, str]:
    conda = find_conda()
    if conda is None:
        return None, ""
    log(f"conda: {conda}")

    python = conda_env_python(conda, args.env_name)
    if python is None or args.recreate:
        if args.recreate and python is not None:
            log(f"removing conda env '{args.env_name}'")
            run([conda, "env", "remove", "-n", args.env_name, "-y"])
        log(f"creating conda env '{args.env_name}'")
        if run([conda, "create", "-n", args.env_name, "python=3.11", "-y"]) != 0:
            return None, ""
        python = conda_env_python(conda, args.env_name)
    if python is None:
        return None, ""

    log(f"conda env python: {python}")
    packages = list(RUNTIME_DEPS) + (DEV_DEPS if args.with_dev else [])
    if _pip_install(str(python), packages, args.mirror) != 0:
        return None, ""
    return str(python), "conda"


def setup_venv(args) -> tuple[str | None, str]:
    if sys.version_info < (3, 10):
        log(f"Python {sys.version_info.major}.{sys.version_info.minor} is too old; 3.10+ required")
        return None, ""

    if args.recreate and VENV_DIR.exists():
        log(f"removing {VENV_DIR}")
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    if not VENV_DIR.exists():
        log(f"creating venv at {VENV_DIR}")
        try:
            venv.EnvBuilder(with_pip=True, clear=False).create(str(VENV_DIR))
        except Exception as exc:
            log(f"venv creation failed: {exc}")
            return None, ""
    else:
        log(f"reusing venv at {VENV_DIR}")

    python = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        log(f"interpreter not found at {python}")
        return None, ""

    packages = list(RUNTIME_DEPS) + (DEV_DEPS if args.with_dev else [])
    if _pip_install(str(python), packages, args.mirror) != 0:
        return None, ""
    return str(python), "venv"


def build_parser() -> argparse.ArgumentParser:
    """Command line interface for the environment bootstrap."""
    parser = argparse.ArgumentParser(description="Create the cover-reverb environment.")
    parser.add_argument("--manager", choices=["auto", "conda", "venv"], default="auto")
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME, help="conda environment name")
    parser.add_argument("--with-dev", action="store_true", help="also install pytest and ruff")
    parser.add_argument("--recreate", action="store_true", help="rebuild from scratch")
    parser.add_argument(
        "--mirror",
        default=os.environ.get("PIP_MIRROR") or None,
        help="pip index URL, e.g. a domestic mirror",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("=" * 66)
    print("  cover-reverb environment setup")
    print("=" * 66)

    python: str | None = None
    kind = ""

    if args.manager in ("auto", "conda"):
        conda = find_conda()
        if conda:
            python, kind = setup_conda(args)
        elif args.manager == "conda":
            log("conda was requested but not found")
            return 1
        else:
            log("conda not found, falling back to venv")

    if python is None and args.manager in ("auto", "venv"):
        python, kind = setup_venv(args)

    if python is None:
        log("could not create an environment")
        return 1

    print()
    log(f"environment ready ({kind})")
    log(f"python: {python}")
    print()
    print("  Next steps:")
    if kind == "conda":
        print(f"    conda activate {args.env_name}")
        print("    cover-reverb              # or: python -m srv.runner")
    else:
        if os.name == "nt":
            print(r"    .venv\Scripts\activate")
        else:
            print("    source .venv/bin/activate")
        print("    cover-reverb              # or: python -m srv.runner")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
