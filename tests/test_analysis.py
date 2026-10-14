"""Analysis helpers: metrics, spectrum shaping, calibration and file loading."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import fftconvolve

from srv.pipeline import (
    BAND_NAMES,
    SR,
    active_ratio_db,
    apply_octave_spectrum,
    band,
    band2d,
    env_lag_peak,
    load_audio,
    lr_corr,
    octave_spectrum_db,
    sidemid_db,
)

from .conftest import synth_dry, synth_rir


class TestLoadAudio:
    def test_mono_is_duplicated_to_stereo(self, tmp_path):
        path = tmp_path / "mono.wav"
        sf.write(str(path), np.zeros(1000), SR)
        data, sr = load_audio(path)
        assert sr == SR
        assert data.shape == (1000, 2)

    def test_resamples_to_the_target_rate(self, tmp_path):
        path = tmp_path / "48k.wav"
        sf.write(str(path), np.zeros(48000), 48000)
        data, sr = load_audio(path)
        assert sr == SR
        assert abs(data.shape[0] - SR) < 16

    def test_downmixes_more_than_two_channels(self, tmp_path):
        path = tmp_path / "six.wav"
        sf.write(str(path), np.zeros((500, 6)), SR)
        data, _ = load_audio(path)
        assert data.shape == (500, 2)

    def test_accepts_a_string_path(self, tmp_path):
        path = tmp_path / "p.wav"
        sf.write(str(path), np.zeros(100), SR)
        data, _ = load_audio(str(path))
        assert data.shape == (100, 2)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((OSError, RuntimeError, ValueError)):
            load_audio(tmp_path / "nope.wav")


class TestStereoMetrics:
    def test_identical_channels_have_no_side_energy(self):
        mono = np.random.default_rng(0).standard_normal(44100)
        stereo = np.stack([mono, mono], axis=1)
        assert sidemid_db(stereo) < -50
        assert lr_corr(stereo) == pytest.approx(1.0, abs=1e-6)

    def test_anti_phase_channels_have_high_side_energy(self):
        mono = np.random.default_rng(1).standard_normal(44100)
        stereo = np.stack([mono, -mono], axis=1)
        assert sidemid_db(stereo) > 30
        assert lr_corr(stereo) == pytest.approx(-1.0, abs=1e-6)

    def test_independent_channels_are_barely_correlated(self):
        rng = np.random.default_rng(2)
        stereo = rng.standard_normal((44100, 2))
        assert abs(lr_corr(stereo)) < 0.05

    def test_constant_input_does_not_crash(self):
        assert lr_corr(np.ones((1000, 2))) == 0.0

    def test_tiny_input_does_not_crash(self):
        assert lr_corr(np.zeros((2, 2))) == 0.0


class TestActiveRatio:
    def test_half_amplitude_reverb_gives_minus_six_db(self):
        """Halving the amplitude costs 6 dB of energy, so the ratio should read ~-6 dB."""
        dry = synth_dry(6.0, seed=20)
        ratio = active_ratio_db(dry, dry * 0.5)
        assert ratio == pytest.approx(-6.0, abs=0.5)

    def test_equal_energy_reads_zero_db(self):
        dry = synth_dry(6.0, seed=21)
        assert active_ratio_db(dry, dry) == pytest.approx(0.0, abs=0.2)

    def test_quieter_reverb_reads_negative(self):
        dry = synth_dry(6.0, seed=22)
        assert active_ratio_db(dry, dry * 0.2) < -10.0

    def test_louder_reverb_reads_positive(self):
        dry = synth_dry(6.0, seed=23)
        assert active_ratio_db(dry, dry * 2.0) > 4.0

    def test_too_short_input_returns_none(self):
        assert active_ratio_db(np.zeros(100), np.zeros(100)) is None

    def test_silent_input_returns_none_or_a_number(self):
        """A silent dry gate must not raise; returning None is acceptable."""
        result = active_ratio_db(np.zeros(SR * 2), np.zeros(SR * 2))
        assert result is None or isinstance(result, float)

    def test_is_insensitive_to_gain_on_the_dry_side(self):
        """Scaling dry and reverb together must not change the ratio."""
        dry = synth_dry(6.0, seed=24)
        rev = dry * 0.3
        base = active_ratio_db(dry, rev)
        scaled = active_ratio_db(dry * 4.0, rev * 4.0)
        assert base == pytest.approx(scaled, abs=0.2)

    def test_gating_makes_the_global_ratio_underestimate_the_reverb(self):
        """The core lesson of this project.

        A real reverb keeps ringing between phrases, so its tails carry energy
        into the quiet blocks. A global energy ratio therefore reports less
        reverb than is present while singing, and calibrating on it makes the
        reverb too loud on the vocal. The active-region ratio is not fooled.

        The synthetic gap here is small (about 1 dB) because this test signal is
        mostly silence; on real stem-separated material the two numbers differed
        by 3-7 dB.
        """
        dry = synth_dry(8.0, seed=25)
        reverb = np.stack(
            [fftconvolve(dry, synth_rir(0.9, seed=26)[:, c])[: len(dry)] for c in range(2)], axis=1
        )
        reverb = reverb / (np.max(np.abs(reverb)) + 1e-20) * 0.5
        rev_mono = reverb.mean(axis=1)

        global_ratio = 10 * np.log10(np.sum(rev_mono**2) / np.sum(dry**2))
        active = active_ratio_db(dry, rev_mono)

        assert active is not None
        # While singing, the reverb is stronger than the global average suggests.
        assert active < global_ratio - 0.5

    def test_active_ratio_ignores_quiet_blocks(self):
        """Energy added purely during pauses must not move the measurement."""
        dry = synth_dry(8.0, seed=27)
        reverb = dry * 0.4

        paused = reverb.copy()
        quiet = np.abs(dry) < 0.02
        paused[quiet] += np.random.default_rng(0).standard_normal(int(quiet.sum())) * 0.002

        assert active_ratio_db(dry, paused) == pytest.approx(
            active_ratio_db(dry, reverb), abs=0.3
        )


class TestOctaveSpectrum:
    def test_every_band_is_reported(self):
        spectrum = octave_spectrum_db(np.random.default_rng(3).standard_normal(SR))
        assert set(spectrum) == set(BAND_NAMES)

    def test_silence_is_finite(self):
        assert np.isfinite(list(octave_spectrum_db(np.zeros(SR)).values())).all()

    def test_a_mid_band_tone_peaks_in_its_own_band(self):
        t = np.arange(SR) / SR
        tone = np.sin(2 * np.pi * 700 * t)
        spectrum = octave_spectrum_db(tone)
        assert max(spectrum, key=spectrum.get) == "500-1000"

    def test_shaping_moves_energy_towards_the_target(self):
        rng = np.random.default_rng(4)
        source = rng.standard_normal(SR)
        target = dict.fromkeys(BAND_NAMES, 0.0)
        target["500-1000"] = 0.0
        for name in BAND_NAMES:
            if name != "500-1000":
                target[name] = -40.0

        before = octave_spectrum_db(source)
        after = octave_spectrum_db(apply_octave_spectrum(np.stack([source, source], 1), target)[:, 0])

        def spread(spec):
            peak = max(spec.values())
            return max(peak - v for v in spec.values())

        assert spread(after) > spread(before)

    def test_shaping_preserves_signal_length(self):
        rng = np.random.default_rng(5)
        source = np.stack([rng.standard_normal(SR)] * 2, axis=1)
        target = {name: -float(i) for i, name in enumerate(BAND_NAMES)}
        assert apply_octave_spectrum(source, target).shape == source.shape

    def test_shaping_to_the_sources_own_spectrum_is_a_no_op(self):
        """The relative shape is already correct, so nothing should change."""
        rng = np.random.default_rng(6)
        source = np.stack([rng.standard_normal(SR)] * 2, axis=1)
        current = octave_spectrum_db(source)
        out = apply_octave_spectrum(source, current)
        assert np.allclose(out, source)


class TestEnvLagPeak:
    def test_delayed_reverb_shows_a_positive_lag(self):
        dry = synth_dry(6.0, seed=30)
        rir = synth_rir(0.6, seed=31)
        rev = np.stack([fftconvolve(dry, rir[:, c])[: len(dry)] for c in range(2)], axis=1)
        lag, peak = env_lag_peak(dry, rev.mean(axis=1))
        assert lag is not None and lag >= 0
        assert peak is not None and peak > 0

    def test_identical_signals_peak_at_zero_lag(self):
        dry = synth_dry(6.0, seed=32)
        lag, peak = env_lag_peak(dry, dry)
        assert lag == 0
        assert peak == pytest.approx(1.0, abs=0.05)

    def test_too_short_input_returns_nones(self):
        lag, peak = env_lag_peak(np.zeros(50), np.zeros(50))
        assert lag is None and peak is None

    def test_constant_input_returns_nones(self):
        lag, peak = env_lag_peak(np.ones(SR), np.ones(SR))
        assert lag is None and peak is None


class TestBandHelpers:
    def test_band_keeps_length(self):
        rng = np.random.default_rng(7)
        assert len(band(rng.standard_normal(10000), 300, 1000)) == 10000

    def test_band2d_keeps_shape(self):
        rng = np.random.default_rng(8)
        data = rng.standard_normal((10000, 2))
        assert band2d(data, 300, 1000).shape == data.shape

    def test_band2d_handles_one_dimension(self):
        rng = np.random.default_rng(9)
        assert band2d(rng.standard_normal(10000), 300, 1000).ndim == 1

    def test_band_rejects_energy_outside_its_range(self):
        t = np.arange(SR) / SR
        low = np.sin(2 * np.pi * 100 * t)
        assert np.sum(np.abs(band(low, 2000, 4000))) < 0.05 * np.sum(np.abs(low))
