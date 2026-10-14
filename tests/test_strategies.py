"""Strategy presets, parameter clamping and the synthetic-RIR builder."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from srv.pipeline import (
    BAND_NAMES,
    DEFAULT_STRATEGY,
    PARAM_RANGES,
    STRATEGIES,
    ReverbStrategy,
    clamp_params,
    lr_corr,
    make_rir,
    rir_metrics,
    sidemid_db,
    validate_strategies,
)


class TestStrategies:
    def test_both_presets_exist(self):
        assert set(STRATEGIES) == {"G0_extreme_dry", "F1_intimate"}

    def test_default_points_at_a_real_preset(self):
        assert DEFAULT_STRATEGY in STRATEGIES

    def test_presets_are_validated_clean(self):
        assert validate_strategies() == []

    def test_validate_flags_out_of_range_value(self, monkeypatch):
        bad = ReverbStrategy(name="bad", early_gain_db=-500.0)
        monkeypatch.setitem(STRATEGIES, "bad", bad)
        problems = validate_strategies()
        assert any("early_gain_db" in p for p in problems)

    def test_g0_disables_early_reflections(self):
        assert STRATEGIES["G0_extreme_dry"].n_early == 0

    def test_f1_has_early_reflections(self):
        assert STRATEGIES["F1_intimate"].n_early > 0

    @pytest.mark.parametrize("name", sorted(STRATEGIES))
    def test_t60_factors_cover_every_band(self, name):
        assert len(STRATEGIES[name].t60_factors) == len(BAND_NAMES)

    @pytest.mark.parametrize("name", sorted(STRATEGIES))
    def test_t60_factors_are_non_increasing(self, name):
        """High frequencies must decay at least as fast as low ones."""
        factors = STRATEGIES[name].t60_factors
        assert all(b <= a + 1e-9 for a, b in itertools.pairwise(factors))


class TestClampParams:
    @pytest.mark.parametrize("key,bounds", sorted(PARAM_RANGES.items()))
    def test_below_range_is_raised(self, key, bounds):
        low, _ = bounds
        assert clamp_params({key: low - 1000})[key] == pytest.approx(low)

    @pytest.mark.parametrize("key,bounds", sorted(PARAM_RANGES.items()))
    def test_above_range_is_lowered(self, key, bounds):
        _, high = bounds
        assert clamp_params({key: high + 1000})[key] == pytest.approx(high)

    def test_integer_params_stay_integers(self):
        assert isinstance(clamp_params({"n_early": 7.4})["n_early"], int)

    def test_unknown_keys_pass_through(self):
        assert clamp_params({"not_a_param": 5})["not_a_param"] == 5

    def test_none_values_are_left_alone(self):
        assert clamp_params({"predelay_ms": None})["predelay_ms"] is None

    def test_empty_input(self):
        assert clamp_params({}) == {}


class TestMakeRir:
    @pytest.mark.parametrize("name", sorted(STRATEGIES))
    def test_shape_is_two_channel(self, name):
        rir = make_rir(STRATEGIES[name], seed=1)
        assert rir.ndim == 2
        assert rir.shape[1] == 2
        assert len(rir) > 0

    @pytest.mark.parametrize("name", sorted(STRATEGIES))
    def test_has_a_direct_sound_at_sample_zero(self, name):
        rir = make_rir(STRATEGIES[name], seed=1)
        head = np.abs(rir[0])
        assert np.all(head > 0)

    def test_deterministic_for_a_given_seed(self):
        a = make_rir(STRATEGIES["F1_intimate"], seed=42)
        b = make_rir(STRATEGIES["F1_intimate"], seed=42)
        assert np.allclose(a, b)

    def test_different_seeds_differ(self):
        a = make_rir(STRATEGIES["F1_intimate"], seed=1)
        b = make_rir(STRATEGIES["F1_intimate"], seed=2)
        assert not np.allclose(a, b)

    def test_channels_are_normalised_to_equal_energy(self):
        rir = make_rir(STRATEGIES["F1_intimate"], seed=3)
        energy = np.sum(rir**2, axis=0)
        assert energy[0] == pytest.approx(energy[1], rel=1e-6)

    def test_no_nans_or_infs(self):
        assert np.all(np.isfinite(make_rir(STRATEGIES["F1_intimate"], seed=4)))

    def test_longer_t60_gives_longer_rir(self):
        short = make_rir(STRATEGIES["G0_extreme_dry"], seed=5, base_t60_s=0.5)
        long = make_rir(STRATEGIES["G0_extreme_dry"], seed=5, base_t60_s=3.0)
        assert len(long) > len(short)

    def test_zero_early_reflections_removes_the_gap_fill(self):
        """With n_early=0 the RIR must be silent between direct sound and late onset."""
        cfg = ReverbStrategy(
            n_early=0, predelay_ms=5.0, late_onset_ms=120.0, late_build_ms=300.0,
            base_t60_s=1.0, t60_scale=1.0,
        )
        rir = make_rir(cfg, seed=6)
        gap = rir[int(0.02 * 44100) : int(0.11 * 44100), 0]
        assert np.max(np.abs(gap)) < 1e-6


class TestStereoWidth:
    def test_positive_side_mid_target_is_reached(self):
        cfg = STRATEGIES["F1_intimate"]
        rir = make_rir(cfg, seed=8, side_mid_db=3.0)
        achieved = sidemid_db(rir)
        assert achieved > 0.0

    def test_negative_side_mid_target_is_reached(self):
        cfg = STRATEGIES["F1_intimate"]
        rir = make_rir(cfg, seed=9, side_mid_db=-3.0)
        achieved = sidemid_db(rir)
        assert achieved < 0.0

    def test_more_side_energy_means_lower_correlation(self):
        cfg = STRATEGIES["F1_intimate"]
        wide = lr_corr(make_rir(cfg, seed=10, side_mid_db=3.0))
        narrow = lr_corr(make_rir(cfg, seed=10, side_mid_db=-6.0))
        assert wide < narrow

    def test_extreme_anti_phase_target_stays_bounded(self):
        """rho is clipped, so an absurd request must still produce finite output."""
        rir = make_rir(STRATEGIES["F1_intimate"], seed=11, side_mid_db=40.0)
        assert np.all(np.isfinite(rir))


class TestRirMetrics:
    def test_returns_all_keys(self):
        metrics = rir_metrics(make_rir(STRATEGIES["F1_intimate"], seed=12)[:, 0])
        assert set(metrics) == {
            "early_late_db",
            "reflection_onsets_0_100ms",
            "env_peak_ms",
            "energy_half_time_ms",
        }

    def test_late_heavy_rir_has_high_half_time(self):
        early = rir_metrics(make_rir(STRATEGIES["F1_intimate"], seed=13)[:, 0])
        late = rir_metrics(make_rir(STRATEGIES["G0_extreme_dry"], seed=13)[:, 0])
        assert late["energy_half_time_ms"] > early["energy_half_time_ms"]

    def test_handles_an_all_zero_input(self):
        metrics = rir_metrics(np.zeros(1024))
        assert metrics["energy_half_time_ms"] == 0.0
        assert np.isfinite(metrics["early_late_db"])
