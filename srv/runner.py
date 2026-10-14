"""Environment check, dependency bootstrap and command line entry point."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REQUIRED = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("soundfile", "soundfile"),
    ("gradio", "gradio"),
)
MIRROR = os.environ.get("COVER_REVERB_PIP_MIRROR", "").strip()


def info(message: str) -> None:
    print(f"  {message}")


def ok(message: str) -> None:
    print(f"  [OK]   {message}")


def warn(message: str) -> None:
    print(f"  [WARN] {message}")


def fail(message: str) -> None:
    print(f"  [FAIL] {message}")


def _pip_cmd(packages: list[str]) -> list[str]:
    cmd = [sys.executable, "-m", "pip", "install"]
    if MIRROR:
        cmd += ["-i", MIRROR]
    return cmd + packages


def check_env(autofix: bool = True, force: bool = False) -> bool:
    """Verify Python and the runtime dependencies, installing what is missing."""
    print("\n=== Environment check ===")
    all_ok = True

    py = sys.version_info
    info(f"Python {py.major}.{py.minor}.{py.micro}  ({sys.executable})")
    if py < (3, 10):
        fail("Python 3.10 or newer is required")
        all_ok = False

    missing: list[str] = []
    for module, package in REQUIRED:
        try:
            __import__(module)
            ok(module)
        except ImportError:
            warn(f"{module} is missing")
            missing.append(package)

    if missing:
        if autofix or force:
            info(f"installing {' '.join(missing)} ...")
            try:
                result = subprocess.run(_pip_cmd(missing), check=False)
            except OSError as exc:
                result = None
                warn(f"could not run pip: {exc}")
            if result is not None and result.returncode == 0:
                ok(f"installed {' '.join(missing)}")
            else:
                fail(f"install failed; run manually: pip install {' '.join(missing)}")
                all_ok = False
        else:
            all_ok = False

    try:
        probe = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
        if probe.returncode == 0 and probe.stdout:
            ok(f"ffmpeg: {probe.stdout.splitlines()[0][:70]}")
        else:
            warn("ffmpeg returned non-zero; MP3 export will not work (WAV output is unaffected)")
    except FileNotFoundError:
        warn("ffmpeg not found; MP3 export disabled (WAV output is unaffected)")

    print("=== check finished ===\n")
    return all_ok


def find_free_port(host: str, start: int, tries: int = 20) -> int | None:
    """Return the first free TCP port at or after ``start``."""
    import socket

    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    return None


def launch(args: argparse.Namespace) -> int:
    """Start the web UI."""
    from .pipeline import STRATEGIES, validate_strategies
    from .ui import default_out_dir
    from .ui import launch as ui_launch

    problems = validate_strategies()
    if problems:
        warn("some preset values fall outside the UI ranges (clamped at runtime):")
        for problem in problems:
            print(f"         {problem}")
    else:
        ok(f"strategy presets validated: {', '.join(STRATEGIES)}")

    out_dir = os.path.abspath(args.out or default_out_dir())
    os.makedirs(out_dir, exist_ok=True)
    allowed = [out_dir]
    if args.extra_allowed:
        for path in args.extra_allowed:
            resolved = os.path.abspath(path)
            os.makedirs(resolved, exist_ok=True)
            allowed.append(resolved)

    port = find_free_port(args.host, args.port)
    if port is None:
        fail(f"{args.host}:{args.port}-{args.port + 19} are all in use; pick another with --port")
        return 1

    print(f"\n{'=' * 66}")
    print(f"  Cover Reverb Matching UI ->  http://{args.host}:{port}")
    print(f"  Output directory: {out_dir}")
    print("  Press Ctrl+C to stop")
    print(f"{'=' * 66}\n")

    ui_launch(
        host=args.host,
        port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        allowed_paths=allowed,
    )
    return 0


def _synthetic_reference(sr: int, seconds: float, seed: int):
    """Build a dry/reverb pair with a plausible reverb level.\n\n    The reverb is scaled so its energy sits well below the dry signal, which puts\n    the active-region ratio in the same range real material lands in.\n    """
    import numpy as np
    from scipy.signal import fftconvolve

    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = 180.0 * (1 + 0.05 * np.sin(2 * np.pi * 3.0 * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    dry = np.sum([(1.0 / k) * np.sin(k * phase) for k in range(1, 16)], axis=0)
    dry *= (0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t - np.pi / 2)) ** 1.5
    dry += 0.01 * rng.standard_normal(n)
    dry = dry / (np.max(np.abs(dry)) + 1e-20)

    t60 = 0.8
    length = int(1.4 * t60 * sr)
    idx = np.arange(length) / sr
    h = rng.standard_normal((length, 2)) * np.exp(-6.9078 * idx / t60)[:, None]
    h /= np.sqrt(np.sum(h**2, axis=0, keepdims=True)) + 1e-30

    reverb = np.stack([fftconvolve(dry, h[:, c])[:n] for c in range(2)], axis=1)
    reverb *= 0.75
    wet = dry[:, None] + reverb
    return dry, wet, h


def selftest(args: argparse.Namespace) -> int:
    """Run the pipeline end to end on synthetic material."""
    import numpy as np
    import soundfile as sf

    from .pipeline import DEFAULT_STRATEGY, STRATEGIES, process

    out_dir = os.path.abspath(args.out or "cover_reverb_selftest")
    os.makedirs(out_dir, exist_ok=True)
    sr = 44100

    ok("generating synthetic reference material")
    dry, wet, _ = _synthetic_reference(sr, 12.0, seed=1234)
    # Split the wet signal back into its two components. Passing the wet vocal as
    # the reverb component would make the reference ratio read near 0 dB and trip
    # the implausible-target guard, which is not what a self-test should exercise.
    reverb = wet - dry[:, None]
    dry_path = os.path.join(out_dir, "_synthetic_dry.wav")
    reverb_path = os.path.join(out_dir, "_synthetic_reverb.wav")
    sf.write(dry_path, dry, sr, subtype="FLOAT")
    sf.write(reverb_path, reverb, sr, subtype="FLOAT")

    cover = np.stack([dry[::-1] * 0.8, dry[::-1] * 0.8], axis=1)
    cover_path = os.path.join(out_dir, "_synthetic_cover.wav")
    sf.write(cover_path, cover, sr, subtype="FLOAT")

    strategy = STRATEGIES.get(args.strategy or DEFAULT_STRATEGY, STRATEGIES[DEFAULT_STRATEGY])
    result = process(
        cover_dry_path=cover_path,
        out_dir=out_dir,
        strategy=strategy,
        reference_dry_path=dry_path,
        reference_reverb_path=reverb_path,
        song_name="selftest",
    )
    if not result.ok:
        fail(f"pipeline failed: {result.message}")
        return 1

    targets = result.report["targets"]
    achieved = result.report["achieved"]
    ok(f"strategy            : {strategy.name}")
    ok(f"reference mode      : {targets['reference_mode']}")
    ok(f"active ratio target : {targets['active_ratio_db']} dB")
    ok(f"active ratio actual : {achieved['active_ratio_db']} dB")
    ok(f"effective T60       : {result.report['rir']['effective_t60_s']} s")
    ok(f"output              : {result.wet_vocal}")
    print(f"\nSelf-test passed. Artifacts in {out_dir}\n")
    return 0


def batch(args: argparse.Namespace) -> int:
    """Process every detected song folder without opening the UI."""
    from .pipeline import STRATEGIES, process
    from .ui import scan_songs

    roots = [args.roots] if args.roots else None
    songs = scan_songs(roots)
    if not songs:
        fail("no stem folders found; pass --roots or set COVER_REVERB_SCAN")
        return 1

    out_root = os.path.abspath(args.out or "cover_reverb_batch")
    strategy = STRATEGIES.get(args.strategy, STRATEGIES["F1_intimate"])
    print(f"\nProcessing {len(songs)} song(s) with {strategy.name} -> {out_root}\n")

    failures = 0
    for entry in songs:
        print(f"--- {entry['song']} ---")
        result = process(
            cover_dry_path=entry["cover_dry"],
            out_dir=os.path.join(out_root, entry["song"]),
            strategy=strategy,
            reference_reverb_path=entry.get("ref_rev"),
            reference_dry_path=entry.get("ref_dry"),
            reference_wet_path=entry.get("ref_wet"),
            instrumental_path=entry.get("instr"),
            song_name=entry["song"],
        )
        if not result.ok:
            failures += 1
            fail(result.message)
            continue
        targets = result.report["targets"]
        achieved = result.report["achieved"]
        ok(
            f"active ratio {targets['active_ratio_db']} -> {achieved['active_ratio_db']} dB, "
            f"T60 {result.report['rir']['effective_t60_s']} s"
        )
        for warning in result.report.get("warnings", []):
            warn(warning)
    print(f"\nDone. Failures: {failures}. Output in {out_root}\n")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cover-reverb",
        description="Match a reference vocal's reverb and apply it to a dry cover vocal.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument(
        "--extra-allowed",
        nargs="*",
        default=None,
        help="additional directories the UI is allowed to serve files from",
    )
    parser.add_argument("--setup", action="store_true", help="force reinstall of dependencies")
    parser.add_argument("--check", action="store_true", help="environment check only")
    parser.add_argument("--selftest", action="store_true", help="run the pipeline on synthetic audio")
    parser.add_argument("--batch", action="store_true", help="process all detected songs, no UI")
    parser.add_argument("--roots", default=None, help="root directory to scan for stems")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["F1_intimate", "G0_extreme_dry"],
        help="reverb strategy (default: F1_intimate)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("=" * 66)
    print("  Cover Reverb Matching")
    print("=" * 66)

    healthy = check_env(autofix=True, force=args.setup)
    if args.check:
        print("Environment check finished.")
        return 0 if healthy else 1
    if not healthy:
        fail("environment is incomplete; fix the issues above first")
        return 1

    if args.selftest:
        return selftest(args)
    if args.batch:
        return batch(args)
    return launch(args)




if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
