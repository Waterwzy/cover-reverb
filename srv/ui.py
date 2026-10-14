"""Gradio web interface for cover reverb matching."""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import replace
from typing import Any

# Keep the app fully offline: the telemetry round-trips HuggingFace on startup and
# only adds latency and tracebacks to the log.
for _key in (
    "HF_HUB_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    "DO_NOT_TRACK",
    "GRADIO_ANALYTICS_ENABLED",
):
    os.environ.setdefault(_key, "1")

from .pipeline import (  # noqa: E402  (env vars must be set before gradio import)
    DEFAULT_STRATEGY,
    STRATEGIES,
    clamp_params,
    encode_mp3,
    process,
    to_json,
)

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - exercised only without gradio
    raise ImportError("gradio is required for the UI. Install with: pip install gradio") from exc


def default_out_dir() -> str:
    """Output directory, overridable with the ``COVER_REVERB_OUT`` env var."""
    env = os.environ.get("COVER_REVERB_OUT")
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(os.getcwd(), "cover_reverb_out"))


def scan_roots() -> list[str]:
    """Directories searched by the folder scanner, overridable with ``COVER_REVERB_SCAN``."""
    env = os.environ.get("COVER_REVERB_SCAN")
    if env:
        return [p for p in env.split(os.pathsep) if p and os.path.isdir(p)]
    home = os.path.expanduser("~")
    candidates = [os.path.join(home, "Music"), os.path.join(home, "music"), home]
    seen: list[str] = []
    for path in candidates:
        if os.path.isdir(path) and path not in seen:
            seen.append(path)
    return seen


#: Filename suffixes a scanned folder may use, in preference order.
_FOLDER_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cover_dry", ("_firefly", "_cover_vocal", "_vocals_dry", "_dry", "")),
    ("ref_rev", ("_vocals_reverb", "_reverb", "_rvb")),
    ("ref_dry", ("_vocals_noreverb", "_noreverb", "_dry_vocal")),
    ("instr", ("_other", "_instrumental", "_inst", "_karaoke")),
)


def _find_in_folder(folder: str, name: str, suffixes: tuple[str, ...]) -> str | None:
    for suffix in suffixes:
        candidate = os.path.join(folder, f"{name}{suffix}.wav")
        if os.path.isfile(candidate):
            return candidate
    return None


def scan_songs(roots: list[str] | None = None) -> list[dict[str, Any]]:
    """Look for complete stem sets in the scan roots.

    A folder qualifies when it contains a dry cover vocal plus at least one
    reference stem. Returns one dict per song with ``ref_wet`` left empty on
    purpose -- a wet reference should only be used when no reverb component is
    available, and auto-filling it would silently degrade the calibration.
    """
    found: list[dict[str, Any]] = []
    for root in roots or scan_roots():
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            folder = os.path.join(root, name)
            if not os.path.isdir(folder):
                continue
            entry: dict[str, Any] = {"song": name, "ref_wet": None}
            for key, suffixes in _FOLDER_PATTERNS:
                entry[key] = _find_in_folder(folder, name, suffixes)
            if entry["cover_dry"] and (entry["ref_rev"] or entry["ref_wet"]):
                # A wet reference is never auto-filled: using it instead of the
                # reverb component silently degrades the level calibration.
                entry["ref_wet"] = None
                found.append(entry)
    return found


def _preset_values(strategy_name: str) -> tuple:
    strategy = STRATEGIES.get(strategy_name, STRATEGIES[DEFAULT_STRATEGY])
    values = clamp_params(
        {
            "predelay_ms": strategy.predelay_ms,
            "n_early": strategy.n_early,
            "early_gain_db": strategy.early_gain_db,
            "early_end_ms": strategy.early_end_ms,
            "late_onset_ms": strategy.late_onset_ms,
            "late_build_ms": strategy.late_build_ms,
            "t60_scale": strategy.t60_scale,
            "base_t60_s": strategy.base_t60_s,
            "wet_trim_db": strategy.wet_trim_db,
        }
    )
    return (
        values["predelay_ms"],
        values["n_early"],
        values["early_gain_db"],
        values["early_end_ms"],
        values["late_onset_ms"],
        values["late_build_ms"],
        values["t60_scale"],
        values["base_t60_s"],
        values["wet_trim_db"],
        strategy.target_mode,
        strategy.match_spectrum,
        strategy.match_stereo,
        strategy.instrumental_match,
    )


def on_strategy_change(strategy_name: str) -> tuple:
    """Update every parameter widget to the chosen preset."""
    return tuple(gr.update(value=v) for v in _preset_values(strategy_name))


def build_strategy(
    strategy_name: str,
    predelay: float,
    n_early: float,
    early_gain: float,
    early_end: float,
    late_onset: float,
    late_build: float,
    t60_scale: float,
    base_t60: float,
    wet_trim: float,
    target_mode: str,
    match_spec: bool,
    match_stereo: bool,
    inst_match: bool,
):
    """Merge UI widget values into a :class:`~srv.pipeline.ReverbStrategy`."""
    base = STRATEGIES.get(strategy_name, STRATEGIES[DEFAULT_STRATEGY])
    params = clamp_params(
        {
            "predelay_ms": predelay,
            "n_early": n_early,
            "early_gain_db": early_gain,
            "early_end_ms": early_end,
            "late_onset_ms": late_onset,
            "late_build_ms": late_build,
            "t60_scale": t60_scale,
            "base_t60_s": base_t60,
            "wet_trim_db": wet_trim,
        }
    )
    return replace(
        base,
        predelay_ms=params["predelay_ms"],
        n_early=params["n_early"],
        early_gain_db=params["early_gain_db"],
        early_end_ms=params["early_end_ms"],
        late_onset_ms=params["late_onset_ms"],
        late_build_ms=params["late_build_ms"],
        t60_scale=params["t60_scale"],
        base_t60_s=params["base_t60_s"],
        wet_trim_db=params["wet_trim_db"],
        target_mode=str(target_mode),
        match_spectrum=bool(match_spec),
        match_stereo=bool(match_stereo),
        instrumental_match=bool(inst_match),
    )


def _metrics_rows(report: dict[str, Any]) -> list[list[Any]]:
    targets = report.get("targets", {})
    achieved = report.get("achieved", {})
    rir = report.get("rir", {})
    rows: list[list[Any]] = [
        ["Active-region reverb/dry (dB)", targets.get("active_ratio_db"), achieved.get("active_ratio_db")],
        ["side - mid (dB)", targets.get("side_mid_db"), achieved.get("side_mid_db")],
        ["L/R correlation", targets.get("lr_corr"), achieved.get("lr_corr")],
        ["Reverb envelope lag (ms)", "-", achieved.get("env_lag_ms")],
        ["Envelope tracking peak", "-", achieved.get("env_peak")],
        ["Early/late energy (dB)", "-", rir.get("early_late_db")],
        ["Early reflections", "-", rir.get("reflection_onsets_0_100ms")],
        ["Envelope peak (ms)", "-", rir.get("env_peak_ms")],
        ["Energy half-time (ms)", "-", rir.get("energy_half_time_ms")],
        ["Effective T60 (s)", "-", rir.get("effective_t60_s")],
    ]
    for band_name, err in achieved.get("octave_spectrum_err_db", {}).items():
        rows.append([f"Spectrum error {band_name} Hz (dB)", 0.0, err])
    return rows


#: Gradio's progress tracker, held as a module-level singleton so that `run`
#: stays callable directly from tests while the UI still gets its progress bar.
_PROGRESS = gr.Progress()


def run(
    cover_dry,
    ref_rev,
    ref_wet,
    ref_dry,
    instr,
    song_name,
    strategy_name,
    predelay,
    n_early,
    early_gain,
    early_end,
    late_onset,
    late_build,
    t60_scale,
    base_t60,
    wet_trim,
    target_mode,
    match_spec,
    match_stereo,
    inst_match,
    out_dir,
    want_mp3,
    progress=_PROGRESS,
):
    """UI entry point. Returns ``(wet, remix, reverb_only, rir, status, report)``."""
    try:
        if not cover_dry:
            return (None, None, None, None, "Please provide the **dry cover vocal**.", "{}")
        if not (ref_rev or ref_wet):
            return (
                None,
                None,
                None,
                None,
                "Please provide reference material: the **original reverb component**, or "
                "the original dry vocal together with the original wet vocal.",
                "{}",
            )

        cfg = build_strategy(
            strategy_name, predelay, n_early, early_gain, early_end, late_onset,
            late_build, t60_scale, base_t60, wet_trim, target_mode, match_spec,
            match_stereo, inst_match,
        )
        target_dir = out_dir or default_out_dir()
        os.makedirs(target_dir, exist_ok=True)

        progress(0.02, desc="starting")
        result = process(
            cover_dry_path=cover_dry,
            out_dir=target_dir,
            strategy=cfg,
            reference_dry_path=ref_dry or None,
            reference_reverb_path=ref_rev or None,
            reference_wet_path=ref_wet or None,
            instrumental_path=instr or None,
            song_name=(song_name or "").strip() or None,
            progress=lambda fraction, desc="": progress(fraction, desc=desc),
        )
        if not result.ok:
            return (None, None, None, None, f"Failed: {result.message}", "{}")

        if want_mp3:
            for path in (result.wet_vocal, result.remix):
                if path:
                    encode_mp3(path)

        report = result.report
        warns = report.get("warnings", [])
        status = (
            f"Done.\n\n"
            f"**Strategy**: `{strategy_name}`  |  "
            f"**effective T60**: {report.get('rir', {}).get('effective_t60_s')} s  |  "
            f"**predelay**: {cfg.predelay_ms} ms  |  **early reflections**: {cfg.n_early}\n\n"
            f"Reference mode: `{report.get('targets', {}).get('reference_mode')}`\n\n"
            f"Output: `{target_dir}`"
        )
        if warns:
            status += "\n\n" + "\n\n".join(f"Warning: {w}" for w in warns)
        return (
            result.wet_vocal,
            result.remix,
            result.reverb_only,
            result.rir,
            status,
            to_json(report),
        )
    except Exception:  # pragma: no cover - defensive, surfaced in the UI
        return (
            None,
            None,
            None,
            None,
            f"Error:\n```\n{traceback.format_exc()[-2000:]}\n```",
            "{}",
        )


def on_scan(roots_text: str):
    """Rescan for stem folders and refresh the quick-pick dropdown."""
    roots = [p.strip() for p in (roots_text or "").split(";") if p.strip()]
    songs = scan_songs(roots or None)
    choices = [s["song"] for s in songs]
    state = {s["song"]: s for s in songs}
    label = (
        f"found {len(choices)} song(s): {', '.join(choices)}"
        if choices
        else "no stem folders found - upload the files manually below"
    )
    return gr.update(choices=choices, value=None), state, label


def on_song_change(song: str, state: dict):
    """Fill the audio widgets from a picked folder."""
    entry = (state or {}).get(song)
    if not entry:
        return None, None, None, None, None, ""
    return (
        entry.get("cover_dry"),
        entry.get("ref_rev"),
        entry.get("ref_dry"),
        entry.get("ref_wet"),
        entry.get("instr"),
        entry.get("song", ""),
    )


def build_ui():
    """Construct the Gradio Blocks application."""

    with gr.Blocks(title="Cover Reverb Matching") as demo:
        gr.Markdown(
            "# Cover Reverb Matching\n"
            "Transfer the reverb character of a reference vocal onto a dry cover vocal.\n\n"
            "Upload the files below, pick a strategy, and press **Generate**. "
            "All processing is local; nothing is uploaded anywhere."
        )
        song_state = gr.State({})

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 1. Material")
                cover_dry = gr.Audio(label="Dry cover vocal (required)", type="filepath")
                gr.Markdown(
                    "**Reference material** - the reverb source. Either provide the "
                    "original reverb component, or the original dry + wet vocals."
                )
                ref_rev = gr.Audio(label="Original reverb component (best)", type="filepath")
                with gr.Row():
                    ref_dry = gr.Audio(label="Original dry vocal (recommended)", type="filepath")
                    ref_wet = gr.Audio(label="...or original wet vocal", type="filepath")
                with gr.Row():
                    instr = gr.Audio(label="Instrumental (optional, for the remix)", type="filepath")
                    song_name = gr.Textbox(label="Output name prefix", value="")

            with gr.Column(scale=2):
                gr.Markdown("### 2. Strategy")
                strategy = gr.Radio(
                    choices=list(STRATEGIES), value=DEFAULT_STRATEGY, label="Reverb strategy"
                )
                gr.Markdown(
                    "\n\n".join(f"**{k}** - {v.desc}" for k, v in STRATEGIES.items())
                )
                out_dir = gr.Textbox(label="Output directory", value=default_out_dir())
                want_mp3 = gr.Checkbox(label="Also export MP3 (needs ffmpeg)", value=False)

                with gr.Accordion("Scan a folder for stems (optional)", open=False):
                    roots = gr.Textbox(
                        label="Search roots (separate multiple with ';')",
                        value=";".join(scan_roots()),
                    )
                    with gr.Row():
                        scan_btn = gr.Button("Scan", size="sm")
                        scan_label = gr.Markdown("")
                    song_dd = gr.Dropdown(label="Detected songs", choices=[], value=None)

        with gr.Accordion("3. Strategy parameters", open=True):
            with gr.Row():
                predelay = gr.Slider(0, 120, value=8.0, step=1.0, label="Predelay (ms)")
                n_early = gr.Slider(0, 24, value=12, step=1, label="Early reflections")
                early_gain = gr.Slider(-60, 0, value=-3.0, step=0.5, label="Early gain (dB)")
            with gr.Row():
                early_end = gr.Slider(20, 250, value=60.0, step=5.0, label="Early end (ms)")
                late_onset = gr.Slider(0, 300, value=40.0, step=5.0, label="Late onset (ms)")
                late_build = gr.Slider(50, 1500, value=420.0, step=10.0, label="Late build (ms)")
            with gr.Row():
                t60_scale = gr.Slider(0.3, 2.5, value=0.80, step=0.05, label="T60 scale")
                base_t60 = gr.Slider(0.3, 5.0, value=1.6, step=0.1, label="Base T60 (s)")
                wet_trim = gr.Slider(-12, 12, value=0.0, step=0.5, label="Reverb trim (dB)")
            with gr.Row():
                target_mode = gr.Radio(
                    choices=["active", "global"], value="active",
                    label="Level calibration (active = recommended)",
                )
                match_spec = gr.Checkbox(label="Match octave spectrum", value=True)
                match_stereo = gr.Checkbox(label="Match stereo width", value=True)
                inst_match = gr.Checkbox(label="Align reverb/instrument balance", value=True)
            reset_btn = gr.Button("Reload strategy defaults", size="sm")

        run_btn = gr.Button("Generate", variant="primary", size="lg")
        status = gr.Markdown()

        with gr.Accordion("4. Results", open=True):
            with gr.Row():
                out_wet = gr.Audio(label="Wet cover vocal", type="filepath")
                out_rev = gr.Audio(
                    label="Reverb only (listen to this to judge the reverb level)",
                    type="filepath",
                )
            with gr.Row():
                out_remix = gr.Audio(label="Remix (instrumental + wet vocal)", type="filepath")
                out_rir = gr.Audio(label="Impulse response used", type="filepath")
            metrics = gr.Dataframe(
                headers=["Metric", "Target", "Achieved"], interactive=False, label="Target vs achieved"
            )
            with gr.Accordion("Full report (JSON)", open=False):
                report_json = gr.Code(label="report", language="json")

        param_outputs = [
            predelay, n_early, early_gain, early_end, late_onset, late_build,
            t60_scale, base_t60, wet_trim, target_mode, match_spec, match_stereo, inst_match,
        ]
        strategy.change(on_strategy_change, inputs=[strategy], outputs=param_outputs)
        strategy.change(
            lambda name: [
                [row[0], row[1], row[2]] for row in _metrics_rows({})
            ],
            inputs=[strategy],
            outputs=[metrics],
        )
        reset_btn.click(on_strategy_change, inputs=[strategy], outputs=param_outputs)
        scan_btn.click(on_scan, inputs=[roots], outputs=[song_dd, song_state, scan_label])
        song_dd.change(
            on_song_change,
            inputs=[song_dd, song_state],
            outputs=[cover_dry, ref_rev, ref_dry, ref_wet, instr, song_name],
        )
        run_btn.click(
            run,
            inputs=[
                cover_dry, ref_rev, ref_wet, ref_dry, instr, song_name, strategy,
                predelay, n_early, early_gain, early_end, late_onset, late_build,
                t60_scale, base_t60, wet_trim, target_mode, match_spec, match_stereo,
                inst_match, out_dir, want_mp3,
            ],
            outputs=[out_wet, out_remix, out_rev, out_rir, status, report_json],
            show_progress="full",
        )
        gr.Markdown(
            "---\n"
            "**Tip**: judge the reverb level by listening to *Reverb only* first. "
            "The `Active-region reverb/dry` row in the metrics table is the key number - "
            "the achieved value should land on the target."
        )
    return demo


def launch(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
    allowed_paths: list[str] | None = None,
):
    """Build and serve the UI. Falls back to the next free port if ``port`` is taken."""
    import socket

    import gradio as gr

    demo = build_ui()

    chosen = None
    for candidate in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
                chosen = candidate
                break
            except OSError:
                continue
    if chosen is None:
        raise OSError(f"no free port in {host}:{port}-{port + 19}")
    if chosen != port:
        print(f"  port {port} is busy, using {chosen} instead")

    paths = allowed_paths or [default_out_dir()]
    demo.queue().launch(
        server_name=host,
        server_port=chosen,
        share=share,
        inbrowser=inbrowser,
        show_error=True,
        allowed_paths=paths,
        theme=gr.themes.Soft(),
    )
    return chosen


__all__ = [
    "build_strategy",
    "build_ui",
    "default_out_dir",
    "launch",
    "on_scan",
    "on_song_change",
    "on_strategy_change",
    "run",
    "scan_roots",
    "scan_songs",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(0 if launch() else 1)
