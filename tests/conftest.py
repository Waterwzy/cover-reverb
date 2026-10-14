"""Shared fixtures: synthetic but realistic audio, generated on the fly.

No binary test data is committed. Every fixture builds its material with a fixed
seed so results are reproducible across machines and platforms.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import fftconvolve

SR = 44100


def synth_dry(seconds: float, sr: int = SR, seed: int = 7) -> np.ndarray:
    """A monophonic harmonic signal with a syllable-like envelope and short pauses.

    Roughly vocal-shaped: a drifting fundamental, a few formant-weighted
    harmonics, amplitude modulation at ~4 Hz, and gated silences so that
    active-region metrics have something to discriminate.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = 190.0 * (1 + 0.05 * np.sin(2 * np.pi * 3.0 * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = np.zeros(n)
    formants = ((700.0, 1.0), (1200.0, 0.6), (2600.0, 0.3))
    for k in range(1, 25):
        weight = sum(w * np.exp(-(((k * 190.0) - fc) ** 2) / (2 * 300.0**2)) for fc, w in formants)
        sig += weight * np.sin(k * phase)
    syllable = (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t - np.pi / 2)) ** 1.5
    gate = (np.sin(2 * np.pi * 0.5 * t) > -0.55).astype(float)
    sig = sig * syllable * gate
    sig += 0.005 * rng.standard_normal(n)
    return sig / (np.max(np.abs(sig)) + 1e-20)


def synth_rir(t60: float = 0.8, sr: int = SR, seed: int = 11, channels: int = 2) -> np.ndarray:
    """Exponentially decaying noise, decorrelated between channels."""
    rng = np.random.default_rng(seed)
    length = int(1.4 * t60 * sr)
    idx = np.arange(length) / sr
    envelope = np.exp(-6.9078 * idx / t60)[:, None]
    rir = rng.standard_normal((length, channels)) * envelope
    rir[0] += 1.0
    rir /= np.sqrt(np.sum(rir**2, axis=0, keepdims=True)) + 1e-30
    return rir


def convolve(dry: np.ndarray, rir: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Convolve a mono signal with a multichannel RIR."""
    wet = np.stack([fftconvolve(dry, rir[:, c])[: len(dry)] for c in range(rir.shape[1])], axis=1)
    if scale != 1.0:
        wet = wet * scale
    return wet / (np.max(np.abs(wet)) + 1e-20)


@pytest.fixture(scope="session")
def sr() -> int:
    return SR


@pytest.fixture(scope="session")
def dry_signal() -> np.ndarray:
    """8 second mono dry signal."""
    return synth_dry(8.0)


@pytest.fixture(scope="session")
def rir_stereo() -> np.ndarray:
    """Stereo RIR with T60 = 0.8 s."""
    return synth_rir(0.8)


@pytest.fixture(scope="session")
def reverb_signal(dry_signal, rir_stereo) -> np.ndarray:
    """The reverb component alone, at a realistic level relative to the dry signal.

    Scaled so the active-region reverb/dry ratio lands near -7 dB, which is what
    real stem-separated material measures.
    """
    return convolve(dry_signal, rir_stereo, scale=0.75)


@pytest.fixture(scope="session")
def wet_signal(dry_signal, reverb_signal) -> np.ndarray:
    """Wet vocal = dry + reverb, exactly as a separator would emit it."""
    return dry_signal[:, None] + reverb_signal


@pytest.fixture(scope="session")
def dry_stereo(dry_signal) -> np.ndarray:
    """The dry signal on two channels."""
    return np.stack([dry_signal, dry_signal], axis=1)


@pytest.fixture(scope="session")
def short_stereo(dry_signal) -> np.ndarray:
    """3 second stereo signal, for functions that only need short input."""
    mono = dry_signal[: 3 * SR]
    return np.stack([mono, mono * 0.9], axis=1)


@pytest.fixture
def audio_files(tmp_path, dry_stereo, reverb_signal, dry_signal, rir_stereo):
    """Write a dry/reverb pair plus a cover vocal and an instrumental to disk."""
    sr = SR
    wet = dry_stereo + reverb_signal
    cover = np.stack([dry_signal[::-1] * 0.8, dry_signal[::-1] * 0.7], axis=1)
    instrumental = np.stack(
        [0.3 * np.roll(dry_signal, 5000)[::-1], 0.3 * np.roll(dry_signal, 9000)[::-1]], axis=1
    )

    paths = {}
    for key, data in (
        ("dry", dry_stereo),
        ("wet", wet),
        ("reverb", reverb_signal),
        ("cover", cover),
        ("instrumental", instrumental),
        ("rir", rir_stereo),
    ):
        path = tmp_path / f"{key}.wav"
        sf.write(str(path), data, sr, subtype="FLOAT")
        paths[key] = str(path)
    return paths
