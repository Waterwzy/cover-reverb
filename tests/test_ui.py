"""UI helpers: path defaults, folder scanning, strategy merging and the Gradio app."""

from __future__ import annotations

import os

import numpy as np
import pytest
import soundfile as sf

from srv.pipeline import STRATEGIES
from srv.ui import (
    build_strategy,
    build_ui,
    default_out_dir,
    on_scan,
    on_song_change,
    on_strategy_change,
    scan_roots,
    scan_songs,
)


@pytest.fixture
def stem_tree(tmp_path):
    """A folder layout matching the documented naming convention."""
    sr = 44100
    tone = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
    stereo = np.stack([tone, tone], axis=1)
    song = tmp_path / "MySong"
    song.mkdir()
    for suffix in ("_firefly", "_vocals_reverb", "_vocals_noreverb", "_other"):
        sf.write(str(song / f"MySong{suffix}.wav"), stereo, sr, subtype="FLOAT")

    # A folder without the required material must be ignored.
    incomplete = tmp_path / "Incomplete"
    incomplete.mkdir()
    sf.write(str(incomplete / "Incomplete_firefly.wav"), stereo, sr, subtype="FLOAT")
    return tmp_path


class TestDefaults:
    def test_output_dir_honours_the_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("COVER_REVERB_OUT", str(tmp_path / "custom"))
        assert default_out_dir() == os.path.abspath(tmp_path / "custom")

    def test_output_dir_falls_back_to_the_cwd(self, monkeypatch):
        monkeypatch.delenv("COVER_REVERB_OUT", raising=False)
        assert default_out_dir().endswith("cover_reverb_out")

    def test_scan_roots_honours_the_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("COVER_REVERB_SCAN", str(tmp_path))
        assert scan_roots() == [str(tmp_path)]

    def test_scan_roots_ignores_missing_paths(self, monkeypatch, tmp_path):
        monkeypatch.setenv("COVER_REVERB_SCAN", str(tmp_path / "nope"))
        assert scan_roots() == []

    def test_scan_roots_defaults_to_existing_directories(self, monkeypatch):
        monkeypatch.delenv("COVER_REVERB_SCAN", raising=False)
        for root in scan_roots():
            assert os.path.isdir(root)


class TestScanSongs:
    def test_finds_a_complete_folder(self, stem_tree):
        found = scan_songs([str(stem_tree)])
        assert [entry["song"] for entry in found] == ["MySong"]

    def test_ignores_incomplete_folders(self, stem_tree):
        found = scan_songs([str(stem_tree)])
        assert all(entry["song"] != "Incomplete" for entry in found)

    def test_maps_every_expected_stem(self, stem_tree):
        entry = scan_songs([str(stem_tree)])[0]
        assert entry["cover_dry"].endswith("MySong_firefly.wav")
        assert entry["ref_rev"].endswith("MySong_vocals_reverb.wav")
        assert entry["ref_dry"].endswith("MySong_vocals_noreverb.wav")
        assert entry["instr"].endswith("MySong_other.wav")

    def test_leaves_ref_wet_empty_on_purpose(self, stem_tree):
        """A wet reference must not be auto-filled; it silently degrades calibration."""
        assert scan_songs([str(stem_tree)])[0]["ref_wet"] is None

    def test_missing_root_yields_nothing(self, tmp_path):
        assert scan_songs([str(tmp_path / "nope")]) == []

    def test_empty_root_yields_nothing(self, tmp_path):
        assert scan_songs([str(tmp_path)]) == []


class TestWidgetCallbacks:
    def test_strategy_change_updates_are_clamped(self):
        updates = on_strategy_change("G0_extreme_dry")
        values = [u["value"] for u in updates]
        assert values[0] == pytest.approx(5.0)  # predelay
        assert values[1] == 0  # n_early
        assert -60.0 <= values[2] <= 0.0  # early gain inside the slider range

    def test_strategy_change_covers_every_bound_widget(self):
        assert len(on_strategy_change("F1_intimate")) == 13

    def test_unknown_strategy_falls_back_to_the_default(self):
        assert on_strategy_change("does_not_exist") == on_strategy_change("F1_intimate")

    def test_on_scan_returns_a_dropdown_update_and_state(self, stem_tree, monkeypatch):
        monkeypatch.setenv("COVER_REVERB_SCAN", str(stem_tree))
        dropdown, state, label = on_scan("")
        assert dropdown["choices"] == ["MySong"]
        assert "MySong" in state
        assert "1" in label

    def test_on_scan_accepts_explicit_roots(self, stem_tree):
        _dropdown, state, _label = on_scan(str(stem_tree))
        assert "MySong" in state

    def test_on_scan_reports_when_nothing_is_found(self, tmp_path):
        _dropdown, state, label = on_scan(str(tmp_path))
        assert state == {}
        assert "no stem folders" in label

    def test_song_change_fills_the_audio_widgets(self, stem_tree):
        state = {entry["song"]: entry for entry in scan_songs([str(stem_tree)])}
        values = on_song_change("MySong", state)
        assert values[0].endswith("_firefly.wav")
        assert values[1].endswith("_vocals_reverb.wav")
        assert values[2].endswith("_vocals_noreverb.wav")
        assert values[3] is None
        assert values[4].endswith("_other.wav")
        assert values[5] == "MySong"

    def test_song_change_with_unknown_name_clears_everything(self):
        assert on_song_change("ghost", {}) == (None, None, None, None, None, "")


class TestBuildStrategy:
    def _call(self, **overrides):
        args = {
            "strategy_name": "F1_intimate",
            "predelay": 12.0,
            "n_early": 6,
            "early_gain": -8.0,
            "early_end": 90.0,
            "late_onset": 70.0,
            "late_build": 500.0,
            "t60_scale": 1.1,
            "base_t60": 1.9,
            "wet_trim": -2.0,
            "target_mode": "active",
            "match_spec": True,
            "match_stereo": False,
            "inst_match": True,
        }
        args.update(overrides)
        return build_strategy(**args)

    def test_widget_values_win_over_the_preset(self):
        cfg = self._call()
        assert cfg.predelay_ms == pytest.approx(12.0)
        assert cfg.n_early == 6
        assert cfg.t60_scale == pytest.approx(1.1)

    def test_out_of_range_widget_values_are_clamped(self):
        cfg = self._call(predelay=9999.0, early_gain=-9999.0)
        assert cfg.predelay_ms == pytest.approx(120.0)
        assert cfg.early_gain_db == pytest.approx(-60.0)

    def test_flags_are_coerced_to_bool(self):
        cfg = self._call(match_stereo=0, match_spec=1)
        assert cfg.match_stereo is False
        assert cfg.match_spectrum is True

    def test_unknown_strategy_uses_the_default_preset(self):
        cfg = self._call(strategy_name="nope")
        assert cfg.name == STRATEGIES["F1_intimate"].name

    def test_returns_a_new_object(self):
        assert self._call() is not STRATEGIES["F1_intimate"]


class TestBuildUi:
    def test_blocks_app_can_be_constructed(self):
        app = build_ui()
        assert app is not None

    def test_expected_labels_are_present(self):
        app = build_ui()
        labels = set()

        def walk(block):
            label = getattr(block, "label", None)
            if label:
                labels.add(label)
            for child in getattr(block, "children", []):
                walk(child)

        for block in app.children:
            walk(block)

        assert any("Dry cover vocal" in str(label) for label in labels)
        assert any("reverb component" in str(label).lower() for label in labels)
