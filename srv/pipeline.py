"""Core pipeline: match a reference vocal's reverb and apply it to a dry cover vocal.

Why not deconvolution?
----------------------
The straightforward idea -- estimate the impulse response from a reference
dry/wet pair and convolve it with the cover -- was tried and measured. It does
not work on stem-separated material: the reference reverb stem is a
time-varying masked decomposition, not a convolution of the dry stem.
Coherence between the two is 0.06-0.26, while a genuine convolutive pair
measures 0.42-0.56. No static impulse response can reproduce it.

So this module matches three *measurable* and *stable* targets instead:

1. **Active-region reverb/dry ratio** (how much reverb). Critically this is
   measured over active blocks only. Using the global energy ratio instead
   over-reverberates by roughly 7-10 dB while singing, because the reference
   reverb is gated and concentrates its energy in the active regions.
2. **Stereo decorrelation** (side/mid energy ratio, L/R correlation).
3. **Octave-band spectral envelope** (reverb timbre).

See the project README for the full rationale and the numbers behind it.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, medfilt, resample_poly, sosfiltfilt

SR = 44100
BAND_EDGES = np.array([125, 250, 500, 1000, 2000, 4000, 8000, 16000], dtype=float)
BAND_NAMES = [f"{int(BAND_EDGES[i])}-{int(BAND_EDGES[i + 1])}" for i in range(len(BAND_EDGES) - 1)]

ProgressFn = Callable[[float, str], None]


# --------------------------------------------------------------------------- config
@dataclass
class ReverbStrategy:
    """Every tunable knob of the reverb synthesis."""

    name: str = "custom"
    desc: str = ""
    predelay_ms: float = 8.0
    """Delay between direct sound and the reverb onset. Longer = clearer vocal."""
    n_early: int = 12
    """Number of discrete early reflections. 0 disables them entirely."""
    early_gain_db: float = -3.0
    """Early reflection level relative to the direct sound."""
    early_end_ms: float = 60.0
    """Early reflections are spread over [predelay_ms, early_end_ms]."""
    late_onset_ms: float = 40.0
    """When the diffuse late field starts to build up."""
    late_build_ms: float = 420.0
    """Time constant of the late-field build-up. Larger = sparser, later."""
    t60_scale: float = 0.80
    """Scales the per-band T60. Effective T60 = base_t60_s * t60_scale."""
    base_t60_s: float = 1.6
    """Reference reverberation time in seconds."""
    wet_trim_db: float = 0.0
    """Global reverb level trim in dB. Negative = less reverb."""
    t60_factors: tuple[float, ...] = (1.00, 1.00, 1.00, 0.88, 0.68, 0.52, 0.42)
    """Per-band T60 multipliers. High frequencies decay faster, as in a real room."""
    target_mode: str = "active"
    """``active`` calibrates on active blocks (recommended); ``global`` uses total energy."""
    match_spectrum: bool = True
    match_stereo: bool = True
    instrumental_match: bool = True


STRATEGIES: dict[str, ReverbStrategy] = {
    "G0_extreme_dry": ReverbStrategy(
        name="G0_extreme_dry",
        desc="Very dry: no early reflections, late field arrives late and stays thin.",
        predelay_ms=5.0,
        n_early=0,
        early_gain_db=-6.0,
        early_end_ms=20.0,
        late_onset_ms=55.0,
        late_build_ms=700.0,
        t60_scale=0.70,
    ),
    "F1_intimate": ReverbStrategy(
        name="F1_intimate",
        desc="Intimate: short predelay, 12 early reflections, fast late build-up.",
        predelay_ms=8.0,
        n_early=12,
        early_gain_db=-3.0,
        early_end_ms=60.0,
        late_onset_ms=40.0,
        late_build_ms=420.0,
        t60_scale=0.80,
    ),
}
DEFAULT_STRATEGY = "F1_intimate"

PARAM_RANGES: dict[str, tuple[float, float]] = {
    "predelay_ms": (0.0, 120.0),
    "n_early": (0, 24),
    "early_gain_db": (-60.0, 0.0),
    "early_end_ms": (20.0, 250.0),
    "late_onset_ms": (0.0, 300.0),
    "late_build_ms": (50.0, 1500.0),
    "t60_scale": (0.3, 2.5),
    "base_t60_s": (0.3, 5.0),
    "wet_trim_db": (-12.0, 12.0),
}


def clamp_params(values: dict[str, Any]) -> dict[str, Any]:
    """Clamp numeric parameters into :data:`PARAM_RANGES`.

    Guards the UI against out-of-range values: a strategy whose default sits
    outside a widget's range would otherwise raise on widget update.
    """
    out: dict[str, Any] = {}
    for key, value in values.items():
        bounds = PARAM_RANGES.get(key)
        if bounds is not None and value is not None:
            lo, hi = bounds
            value = min(max(float(value), lo), hi)
            if isinstance(lo, int) and isinstance(hi, int):
                value = round(value)
        out[key] = value
    return out


def validate_strategies() -> list[str]:
    """Return a list of preset values that fall outside the UI ranges."""
    problems: list[str] = []
    for name, strategy in STRATEGIES.items():
        for key, (lo, hi) in PARAM_RANGES.items():
            value = getattr(strategy, key, None)
            if value is None:
                continue
            if not lo <= float(value) <= hi:
                problems.append(f"strategy {name}: {key}={value} outside UI range [{lo}, {hi}]")
    return problems


@dataclass
class PipelineResult:
    """Outcome of :func:`process`."""

    ok: bool
    message: str = ""
    wet_vocal: str | None = None
    remix: str | None = None
    rir: str | None = None
    reverb_only: str | None = None
    report: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- dsp helpers
#: ``sosfiltfilt`` pads by three times the number of filter taps; a 4th-order
#: Butterworth band-pass is 9 taps, so anything at or below this length cannot be
#: filtered and must be short-circuited.
_MIN_FILTER_LENGTH = 32


def band(x: np.ndarray, lo: float, hi: float, sr: int = SR) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    Returns zeros for input too short to filter, rather than raising.
    """
    if x.size < _MIN_FILTER_LENGTH:
        return np.zeros_like(x)
    hi = min(hi, sr / 2 * 0.95)
    return sosfiltfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), x)


def band2d(x: np.ndarray, lo: float, hi: float, sr: int = SR) -> np.ndarray:
    """Band-pass every channel of a ``(n_samples, n_channels)`` array."""
    if x.ndim == 1:
        return band(x, lo, hi, sr)
    return np.stack([band(x[:, c], lo, hi, sr) for c in range(x.shape[1])], axis=1)


def load_audio(path: str, target_sr: int = SR) -> tuple[np.ndarray, int]:
    """Read audio as float64 ``(n_samples, 2)`` at ``target_sr``."""
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    if sr != target_sr:
        x = resample_poly(x, target_sr, sr, axis=0)
        sr = target_sr
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    return x, sr


def lr_corr(x: np.ndarray, sr: int = SR) -> float:
    """L/R waveform correlation in the 300-6000 Hz band."""
    b = band2d(x, 300, 6000, sr)
    if b.shape[0] < 8 or b.shape[1] < 2:
        return 0.0
    left, right = b[:, 0], b[:, 1]
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def sidemid_db(x: np.ndarray) -> float:
    """Side-to-mid energy ratio in dB."""
    mid = (x[:, 0] + x[:, 1]) / 2
    side = (x[:, 0] - x[:, 1]) / 2
    return float(10 * np.log10((np.sum(side**2) + 1e-30) / (np.sum(mid**2) + 1e-30)))


def block_energy(x: np.ndarray, sr: int = SR, block_ms: float = 10.0) -> np.ndarray:
    """Energy per non-overlapping block."""
    blk = max(int(block_ms * sr / 1000.0), 1)
    return np.array([np.sum(x[i * blk : (i + 1) * blk] ** 2) for i in range(len(x) // blk)])


def active_ratio_db(
    dry_mono: np.ndarray, rev_mono: np.ndarray, sr: int = SR, rel_db: float = -25.0
) -> float | None:
    """Reverb/dry energy ratio measured over active blocks only.

    This is the single most important calibration number in the project. Using
    the global ratio instead over-reverberates by ~7-10 dB during singing.
    """
    ed = block_energy(dry_mono, sr)
    er = block_energy(rev_mono, sr)
    m = min(len(ed), len(er))
    if m < 20:
        return None
    ed, er = ed[:m], er[:m]
    ed_db = medfilt(10 * np.log10(ed + 1e-30), 5)
    mask = ed_db > (np.percentile(ed_db, 95) + rel_db)
    if mask.sum() < 10:
        return None
    ratio = (np.sum(er[mask]) + 1e-30) / (np.sum(ed[mask]) + 1e-30)
    return round(float(10 * np.log10(ratio)), 2)


def octave_spectrum_db(x: np.ndarray, sr: int = SR) -> dict[str, float]:
    """Absolute energy per octave band, in dB."""
    return {
        name: float(10 * np.log10(np.sum(band2d(x, BAND_EDGES[i], BAND_EDGES[i + 1], sr) ** 2) + 1e-30))
        for i, name in enumerate(BAND_NAMES)
    }


def apply_octave_spectrum(
    wet: np.ndarray, target_spec_db: dict[str, float], sr: int = SR
) -> np.ndarray:
    """Reshape ``wet`` so its octave-band envelope matches ``target_spec_db``.

    Only the *relative* shape is matched: both curves are normalised to their own
    maximum first, so the overall level is preserved.
    """
    cur = octave_spectrum_db(wet, sr)
    cur_max = max(cur.values())
    tgt_max = max(target_spec_db.values())
    out = wet.copy()
    for i, name in enumerate(BAND_NAMES):
        delta = (target_spec_db[name] - tgt_max) - (cur[name] - cur_max)
        if abs(delta) > 0.01:
            out = out + band2d(wet, BAND_EDGES[i], BAND_EDGES[i + 1], sr) * (10 ** (delta / 20.0) - 1.0)
    return out


# --------------------------------------------------------------------------- RIR synthesis
def _stereo_weights(side_mid_db: float) -> tuple[float, float]:
    """Return ``(rho, sign)`` for the shared/independent stereo decomposition.

    A mono source convolved with two impulse responses cannot produce ``side > mid``
    unless the channels are explicitly anti-phase, so ``sign`` flips to -1 when the
    target ratio exceeds 0 dB.

    Derivation for the anti-phase case::

        y_L =  sqrt(rho)*c*d + sqrt(1-rho)*i1*d
        y_R = -sqrt(rho)*c*d - sqrt(1-rho)*i2*d
        E_side/E_mid = (1 + rho) / (1 - rho)

    which inverts to ``rho = (ratio - 1) / (ratio + 1)``.
    """
    ratio = 10 ** (side_mid_db / 10.0)
    if ratio > 1.0:
        rho = (ratio - 1.0) / (ratio + 1.0)
        sign = -1.0
    else:
        rho = (1.0 - ratio) / (1.0 + ratio)
        sign = 1.0
    return float(np.clip(rho, 0.0, 0.97)), sign


def make_rir(
    cfg: ReverbStrategy,
    sr: int = SR,
    seed: int = 0,
    side_mid_db: float = 3.0,
    base_t60_s: float | None = None,
) -> np.ndarray:
    """Synthesise a stereo RIR with the structure described by ``cfg``.

    Direct sound + discrete early reflections + a smoothly building diffuse late
    field, each octave band decaying with its own T60. The late field ramps in
    continuously (never as a step, which would click and inflate echo-density
    metrics).
    """
    b60 = float(base_t60_s if base_t60_s else cfg.base_t60_s)
    rng = np.random.default_rng(seed)
    rho, sign = _stereo_weights(side_mid_db)

    length = int(1.6 * b60 * cfg.t60_scale * sr) + int(0.2 * sr)
    t = np.arange(length) / sr
    predelay = cfg.predelay_ms / 1000.0
    early_end = cfg.early_end_ms / 1000.0
    late_onset = cfg.late_onset_ms / 1000.0
    build = max(cfg.late_build_ms / 1000.0, 1e-3)

    t_after = np.maximum(t - late_onset, 0.0)
    onset_w = np.where(t >= late_onset, 1.0 - np.exp(-t_after / build * 3.0), 0.0)
    factors = cfg.t60_factors

    def shaped_late(noise: np.ndarray) -> np.ndarray:
        acc = np.zeros(length)
        for i in range(len(BAND_EDGES) - 1):
            lo, hi = BAND_EDGES[i], min(BAND_EDGES[i + 1], sr / 2 * 0.95)
            if lo >= hi:
                continue
            t60 = max(b60 * cfg.t60_scale * factors[i], 0.02)
            acc += band(noise, lo, hi, sr) * np.exp(-6.9078 * t_after / t60)
        return acc * onset_w

    def early_taps(seed_offset: int) -> np.ndarray:
        taps = np.zeros(length)
        if cfg.n_early <= 0:
            return taps
        g = np.random.default_rng(seed + seed_offset)
        hi = max(early_end, predelay + 0.005)
        delays = np.sort(g.uniform(predelay, hi, cfg.n_early))
        amps = (
            10 ** (cfg.early_gain_db / 20.0)
            * g.uniform(0.4, 1.0, cfg.n_early)
            * np.exp(-(delays - predelay) / max(hi - predelay, 1e-6))
        )
        signs = g.choice([-1.0, 1.0], cfg.n_early)
        for delay, amp, sgn in zip(delays, amps, signs, strict=True):
            idx = int(delay * sr)
            if idx < length:
                taps[idx] += amp * sgn
        return taps

    common = shaped_late(rng.standard_normal(length)) + early_taps(1)
    indep_l = shaped_late(rng.standard_normal(length)) + early_taps(2)
    indep_r = shaped_late(rng.standard_normal(length)) + early_taps(3)
    common[0] += 1.0  # direct sound

    left = np.sqrt(rho) * common + np.sqrt(1 - rho) * indep_l
    right = sign * (np.sqrt(rho) * common + np.sqrt(1 - rho) * indep_r)
    rir = np.stack([left, right], axis=1)
    rir /= np.sqrt(np.sum(rir**2, axis=0, keepdims=True)) + 1e-30
    return rir


def rir_metrics(rir_mono: np.ndarray, sr: int = SR) -> dict[str, float]:
    """Objective descriptors of a RIR: early/late split, echo onsets, energy build-up."""
    energy = rir_mono**2
    split = int(0.08 * sr)
    early, late = float(np.sum(energy[:split])), float(np.sum(energy[split:]))
    head = np.abs(rir_mono[: int(0.1 * sr)])
    if head.size and np.abs(rir_mono).max() > 0:
        above = head > np.abs(rir_mono).max() * 10 ** (-30 / 20)
        onsets = int(np.sum((~above[:-1]) & above[1:]))
    else:
        onsets = 0
    cumsum = np.cumsum(energy)
    half_time = (
        float(np.searchsorted(cumsum, 0.5 * cumsum[-1]) / sr * 1000) if cumsum[-1] > 0 else 0.0
    )
    win = max(int(0.02 * sr), 1)
    envelope = np.convolve(np.abs(rir_mono), np.ones(win) / win, mode="same")
    return {
        "early_late_db": round(10 * np.log10((early + 1e-30) / (late + 1e-30)), 1),
        "reflection_onsets_0_100ms": onsets,
        "env_peak_ms": round(float(np.argmax(envelope)) / sr * 1000, 1),
        "energy_half_time_ms": round(half_time, 1),
    }


def env_lag_peak(
    dry_mono: np.ndarray,
    rev_mono: np.ndarray,
    sr: int = SR,
    block_ms: float = 10.0,
    max_ms: float = 400.0,
) -> tuple[float | None, float | None]:
    """Lag and peak value of the reverb-envelope / dry-envelope cross-correlation."""
    ed = block_energy(dry_mono, sr, block_ms)
    er = block_energy(rev_mono, sr, block_ms)
    m = min(len(ed), len(er))
    if m < 20:
        return None, None
    ed, er = ed[:m], er[:m]
    a = np.log10(ed + 1e-30)
    b = np.log10(er + 1e-30)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.sqrt(np.sum(a**2)), np.sqrt(np.sum(b**2))
    if na < 1e-12 or nb < 1e-12:
        return None, None
    a, b = a / na, b / nb
    cc = np.correlate(b, a, mode="full")
    mid = len(a) - 1
    window = cc[mid : mid + int(max_ms / block_ms) + 1]
    if window.size == 0:
        return None, None
    idx = int(np.argmax(window))
    return round(idx * block_ms, 1), round(float(window[idx]), 3)


# --------------------------------------------------------------------------- pipeline
def _resolve_reference(
    reference_dry_path: str | None,
    reference_reverb_path: str | None,
    reference_wet_path: str | None,
    report: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Load the reference material in whichever of the four supported ways was given."""
    have_rev = bool(reference_reverb_path and os.path.exists(reference_reverb_path))
    have_dry = bool(reference_dry_path and os.path.exists(reference_dry_path))
    have_wet = bool(reference_wet_path and os.path.exists(reference_wet_path))

    if not (have_rev or have_wet):
        raise ValueError(
            "Missing reference material: provide the original reverb component, "
            "or the original dry vocal together with the original wet vocal."
        )

    if have_rev and have_dry:
        dry, _ = load_audio(reference_dry_path)
        rev, _ = load_audio(reference_reverb_path)
        return dry, rev, "dry+reverb (best)"

    if have_wet and have_dry:
        dry, _ = load_audio(reference_dry_path)
        wet, _ = load_audio(reference_wet_path)
        m = min(len(dry), len(wet))
        return dry[:m], wet[:m] - dry[:m], "dry+wet (reverb derived by subtraction)"

    if have_rev:
        rev, _ = load_audio(reference_reverb_path)
        report["warnings"].append(
            "No reference dry vocal supplied. The active-region gate falls back to the "
            "reverb component's own envelope, which biases the level calibration low by "
            "roughly 1-3 dB. Supply the dry vocal for an exact match."
        )
        return None, rev, "reverb_only"

    wet, _ = load_audio(reference_wet_path)
    report["warnings"].append(
        "Only the wet reference vocal was supplied, so the reverb component cannot be "
        "isolated. Both the spectral envelope and the reverb level will be inaccurate. "
        "Provide the dry vocal or the reverb component instead."
    )
    return None, wet, "only_wet (low confidence)"


def process(
    cover_dry_path: str,
    out_dir: str,
    strategy: ReverbStrategy,
    reference_dry_path: str | None = None,
    reference_reverb_path: str | None = None,
    reference_wet_path: str | None = None,
    instrumental_path: str | None = None,
    base_t60_s: float | None = None,
    wet_trim_db: float = 0.0,
    make_remix: bool = True,
    song_name: str | None = None,
    max_analysis_s: float = 120.0,
    progress: ProgressFn | None = None,
) -> PipelineResult:
    """Run the full matching pipeline.

    Args:
        cover_dry_path: Dry cover vocal. Required.
        out_dir: Directory for the written files.
        strategy: Reverb synthesis configuration.
        reference_dry_path: Original dry vocal. Strongly recommended.
        reference_reverb_path: Original reverb component (best single input).
        reference_wet_path: Original wet vocal; used only if the reverb component
            is unavailable.
        instrumental_path: Optional backing track, used for the remix output and
            for aligning the reverb-to-instrument balance.
        base_t60_s: Overrides ``strategy.base_t60_s``.
        wet_trim_db: Additional reverb level trim in dB.
        make_remix: Write ``<name>_remix.wav`` when an instrumental is available.
        song_name: Output filename prefix. Defaults to the cover vocal's stem.
        max_analysis_s: Length of the analysis window used to estimate targets.
        progress: Optional ``fn(fraction, description)`` callback.

    Returns:
        A :class:`PipelineResult`; check ``.ok`` before using the paths.
    """

    def log(fraction: float, message: str) -> None:
        if progress is not None:
            progress(fraction, message)

    os.makedirs(out_dir, exist_ok=True)
    name = song_name or os.path.splitext(os.path.basename(cover_dry_path))[0]
    report: dict[str, Any] = {"song": name, "strategy": asdict(strategy), "warnings": []}

    log(0.05, "Reading cover vocal...")
    try:
        cover, sr = load_audio(cover_dry_path)
        ref_dry, ref_rev, mode = _resolve_reference(
            reference_dry_path, reference_reverb_path, reference_wet_path, report
        )
    except (ValueError, RuntimeError) as exc:
        return PipelineResult(False, str(exc))
    report["reference_mode"] = mode

    n = len(cover)
    if ref_dry is not None:
        n = min(n, len(ref_dry))
        cover, ref_dry = cover[:n], ref_dry[:n]
    n = min(n, len(ref_rev))
    cover, ref_rev = cover[:n], ref_rev[:n]

    instrumental = None
    if instrumental_path and os.path.exists(instrumental_path):
        instrumental, _ = load_audio(instrumental_path)
        instrumental = instrumental[:n]

    # ---------------------------------------------------------------- targets
    log(0.25, "Estimating reverb targets...")
    win = min(int(max_analysis_s * sr), n)
    start = int(n * 0.25)
    if start + win > n:
        start = max(0, n - win)
    seg = slice(start, start + win)
    if seg.stop - seg.start < int(5 * sr):
        seg = slice(0, n)

    rev_seg = ref_rev[seg]
    dry_seg = ref_dry[seg] if ref_dry is not None else None

    target_side_mid = sidemid_db(rev_seg)
    target_lr = lr_corr(rev_seg)
    spec_raw = octave_spectrum_db(rev_seg)
    target_spec = {k: v - max(spec_raw.values()) for k, v in spec_raw.items()}

    gate = dry_seg.mean(axis=1) if dry_seg is not None else rev_seg.mean(axis=1)
    target_active = active_ratio_db(gate, rev_seg.mean(axis=1))

    low_confidence = dry_seg is None
    if target_active is None or target_active > -1.0:
        low_confidence = True
        if target_active is not None:
            report["warnings"].append(
                f"Measured active reverb/dry ratio is {target_active:+.2f} dB, which is "
                "implausibly high (a normal value is below -1 dB). Falling back to a "
                "conservative -8.0 dB target."
            )
        target_active = -8.0

    report["targets"] = {
        "side_mid_db": round(float(target_side_mid), 2),
        "lr_corr": round(float(target_lr), 3),
        "active_ratio_db": round(float(target_active), 2),
        "octave_spectrum_rel_db": {k: round(float(v), 1) for k, v in target_spec.items()},
        "low_confidence": low_confidence,
        "reference_mode": mode,
    }

    # ---------------------------------------------------------------- synthesis
    log(0.4, "Synthesising impulse response...")
    rir = make_rir(
        strategy,
        sr=sr,
        seed=abs(hash(name)) % 9999,
        side_mid_db=target_side_mid if strategy.match_stereo else 0.0,
        base_t60_s=base_t60_s,
    )
    effective_t60 = float(base_t60_s if base_t60_s else strategy.base_t60_s) * strategy.t60_scale
    report["rir"] = {
        **rir_metrics(rir[:, 0], sr),
        "t60_base_s": float(base_t60_s if base_t60_s else strategy.base_t60_s),
        "t60_scale": strategy.t60_scale,
        "effective_t60_s": round(effective_t60, 3),
    }

    log(0.5, "Convolving...")
    wet_rev = np.stack(
        [fftconvolve(cover[:, c], rir[:, c])[: len(cover)] for c in range(2)], axis=1
    )

    if strategy.match_spectrum:
        log(0.6, "Shaping octave spectrum...")
        wet_rev = apply_octave_spectrum(wet_rev, target_spec, sr)

    if strategy.instrumental_match and instrumental is not None:
        log(0.7, "Aligning reverb-to-instrument balance...")
        inst_energy = float(np.sum(instrumental**2))
        ref_rev_energy = float(np.sum(ref_rev**2))
        if inst_energy > 1e-12:
            want = 10 * np.log10(ref_rev_energy / inst_energy)
            have = 10 * np.log10((np.sum(wet_rev**2) + 1e-30) / inst_energy)
            wet_rev *= 10 ** ((want - have) / 20.0)

    # ---------------------------------------------------------------- calibration
    # Order matters: the instrumental balance above changes the reverb level, so the
    # active-region calibration has to run last or it gets undone.
    if strategy.target_mode == "active":
        log(0.8, "Calibrating reverb level on active regions...")
        current = active_ratio_db(cover.mean(axis=1), wet_rev.mean(axis=1))
        if current is not None:
            wet_rev *= 10 ** ((target_active - current + wet_trim_db) / 20.0)

    wet = cover + wet_rev

    # ---------------------------------------------------------------- output
    log(0.9, "Writing files...")
    wet_path = os.path.join(out_dir, f"{name}_wet_vocal.wav")
    rev_path = os.path.join(out_dir, f"{name}_reverb_only.wav")
    rir_path = os.path.join(out_dir, f"{name}_rir.wav")
    sf.write(wet_path, wet, sr, subtype="FLOAT")
    sf.write(rev_path, wet_rev, sr, subtype="FLOAT")
    sf.write(rir_path, rir, sr, subtype="FLOAT")

    remix_path = None
    if make_remix and instrumental is not None:
        mix = instrumental + wet
        peak = float(np.max(np.abs(mix)))
        if peak > 0.999:
            mix = mix / peak * 0.999
        remix_path = os.path.join(out_dir, f"{name}_remix.wav")
        sf.write(remix_path, mix, sr, subtype="FLOAT")
        report["remix_peak_before_norm"] = round(peak, 4)

    # ---------------------------------------------------------------- verify
    achieved_spec = octave_spectrum_db(wet_rev[seg] if seg.stop <= len(wet_rev) else wet_rev, sr)
    spec_max = max(achieved_spec.values())
    lag, peak = env_lag_peak(cover.mean(axis=1), wet_rev.mean(axis=1))
    report["achieved"] = {
        "active_ratio_db": active_ratio_db(cover.mean(axis=1), wet_rev.mean(axis=1)),
        "side_mid_db": round(sidemid_db(wet_rev), 2),
        "lr_corr": round(lr_corr(wet_rev), 3),
        "env_lag_ms": lag,
        "env_peak": peak,
        "octave_spectrum_err_db": {
            name_: round((achieved_spec[name_] - spec_max) - target_spec[name_], 2)
            for name_ in BAND_NAMES
        },
    }
    report["outputs"] = {
        "wet_vocal": wet_path,
        "remix": remix_path,
        "reverb_only": rev_path,
        "rir": rir_path,
    }

    log(1.0, "Done")
    return PipelineResult(True, "done", wet_path, remix_path, rir_path, rev_path, report)


def encode_mp3(wav_path: str, bitrate: str = "320k") -> str | None:
    """Encode a WAV to MP3 with ffmpeg. Returns the output path, or None on failure."""
    out = os.path.splitext(wav_path)[0] + ".mp3"
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", wav_path,
             "-c:a", "libmp3lame", "-b:a", bitrate, out],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return out if result.returncode == 0 else None


def to_json(report: dict[str, Any], indent: int = 2) -> str:
    """Serialise a report dict, tolerating numpy scalars."""
    return json.dumps(report, ensure_ascii=False, indent=indent, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


__all__ = [
    "BAND_EDGES",
    "BAND_NAMES",
    "DEFAULT_STRATEGY",
    "PARAM_RANGES",
    "STRATEGIES",
    "PipelineResult",
    "ReverbStrategy",
    "active_ratio_db",
    "apply_octave_spectrum",
    "band",
    "band2d",
    "clamp_params",
    "encode_mp3",
    "env_lag_peak",
    "load_audio",
    "lr_corr",
    "make_rir",
    "octave_spectrum_db",
    "process",
    "replace",
    "rir_metrics",
    "sidemid_db",
    "to_json",
    "validate_strategies",
]
