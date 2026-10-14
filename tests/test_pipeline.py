"""End-to-end behaviour of :func:`srv.pipeline.process`."""

from __future__ import annotations

import os

import numpy as np
import pytest
import soundfile as sf

from srv.pipeline import DEFAULT_STRATEGY, STRATEGIES, PipelineResult, process, to_json


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    """Run the pipeline once for the whole module and reuse the result."""
    from .conftest import SR, convolve, synth_dry, synth_rir

    work = tmp_path_factory.mktemp("pipeline")
    dry = synth_dry(10.0, seed=101)
    rir = synth_rir(0.7, seed=102)
    dry_stereo = np.stack([dry, dry], axis=1)
    # Realistic reverb level: this is what makes the active-region ratio negative,
    # which is the range the calibration is built around.
    reverb = convolve(dry, rir, scale=0.75)
    wet = dry_stereo + reverb
    cover = np.stack([dry[::-1] * 0.8, dry[::-1] * 0.7], axis=1)
    instrumental = np.stack([np.roll(dry, 4000) * 0.3, np.roll(dry, 8000) * 0.3], axis=1)

    paths = {}
    for key, data in (
        ("dry", dry_stereo),
        ("reverb", reverb),
        ("wet", wet),
        ("cover", cover),
        ("instrumental", instrumental),
    ):
        path = work / f"{key}.wav"
        sf.write(str(path), data, SR, subtype="FLOAT")
        paths[key] = str(path)

    result = process(
        cover_dry_path=paths["cover"],
        out_dir=str(work / "out"),
        strategy=STRATEGIES[DEFAULT_STRATEGY],
        reference_dry_path=paths["dry"],
        reference_reverb_path=paths["reverb"],
        instrumental_path=paths["instrumental"],
        song_name="unit",
    )
    return result, paths, work


class TestProcessHappyPath:
    def test_reports_success(self, outputs):
        result, _, _ = outputs
        assert result.ok, result.message

    def test_returns_a_pipeline_result(self, outputs):
        result, _, _ = outputs
        assert isinstance(result, PipelineResult)

    def test_writes_every_declared_output(self, outputs):
        result, _, _ = outputs
        for path in (result.wet_vocal, result.remix, result.reverb_only, result.rir):
            assert path and os.path.isfile(path), path

    def test_outputs_are_readable_stereo_audio(self, outputs):
        result, _, _ = outputs
        data, sr = sf.read(result.wet_vocal, always_2d=True)
        assert sr == 44100
        assert data.shape[1] == 2
        assert len(data) > 0
        assert np.all(np.isfinite(data))

    def test_remix_is_written_when_an_instrumental_is_given(self, outputs):
        result, _, _ = outputs
        assert result.remix is not None

    def test_report_is_json_serialisable(self, outputs):
        result, _, _ = outputs
        text = to_json(result.report)
        assert '"targets"' in text
        assert '"achieved"' in text

    def test_report_has_the_expected_sections(self, outputs):
        result, _, _ = outputs
        for key in ("song", "strategy", "targets", "rir", "achieved", "outputs", "warnings"):
            assert key in result.report


class TestCalibration:
    def test_active_ratio_lands_on_target(self, outputs):
        """The headline invariant: achieved level must match the measured target."""
        result, _, _ = outputs
        target = result.report["targets"]["active_ratio_db"]
        achieved = result.report["achieved"]["active_ratio_db"]
        assert achieved == pytest.approx(target, abs=0.5)

    def test_target_is_negative_for_a_normal_reference(self, outputs):
        result, _, _ = outputs
        assert result.report["targets"]["active_ratio_db"] < -1.0

    def test_low_confidence_is_false_when_the_dry_reference_is_given(self, outputs):
        result, _, _ = outputs
        assert result.report["targets"]["low_confidence"] is False

    def test_no_warnings_on_a_complete_input_set(self, outputs):
        result, _, _ = outputs
        assert result.report["warnings"] == []

    def test_spectrum_match_is_within_tolerance(self, outputs):
        result, _, _ = outputs
        errors = result.report["achieved"]["octave_spectrum_err_db"]
        assert all(abs(v) < 3.0 for v in errors.values()), errors

    def test_side_mid_is_in_the_right_ballpark(self, outputs):
        result, _, _ = outputs
        target = result.report["targets"]["side_mid_db"]
        achieved = result.report["achieved"]["side_mid_db"]
        assert achieved == pytest.approx(target, abs=2.5)


class TestReferenceModes:
    def test_wet_plus_dry_derives_the_reverb(self, outputs, tmp_path):
        _, paths, _ = outputs
        result = process(
            cover_dry_path=paths["cover"],
            out_dir=str(tmp_path / "o1"),
            strategy=STRATEGIES[DEFAULT_STRATEGY],
            reference_dry_path=paths["dry"],
            reference_wet_path=paths["wet"],
            song_name="wetmode",
        )
        assert result.ok
        assert "dry+wet" in result.report["targets"]["reference_mode"]

    def test_reverb_only_lowers_confidence(self, outputs, tmp_path):
        _, paths, _ = outputs
        result = process(
            cover_dry_path=paths["cover"],
            out_dir=str(tmp_path / "o2"),
            strategy=STRATEGIES[DEFAULT_STRATEGY],
            reference_reverb_path=paths["reverb"],
            song_name="revonly",
        )
        assert result.ok
        assert result.report["targets"]["low_confidence"] is True
        assert result.report["warnings"]

    def test_wet_only_lowers_confidence(self, outputs, tmp_path):
        _, paths, _ = outputs
        result = process(
            cover_dry_path=paths["cover"],
            out_dir=str(tmp_path / "o3"),
            strategy=STRATEGIES[DEFAULT_STRATEGY],
            reference_wet_path=paths["wet"],
            song_name="wetonely",
        )
        assert result.ok
        assert result.report["targets"]["low_confidence"] is True

    def test_implausible_active_ratio_falls_back_to_a_safe_target(self, tmp_path):
        """A wet-only reference yields a ratio near 0 dB, which must be rejected."""
        from .conftest import SR, convolve, synth_dry, synth_rir

        dry = synth_dry(10.0, seed=201)
        wet = np.stack([dry, dry], axis=1) + convolve(dry, synth_rir(0.6, seed=202), scale=0.75)
        cover = np.stack([dry[::-1]] * 2, axis=1)
        cover_path = tmp_path / "cover.wav"
        wet_path = tmp_path / "wet.wav"
        sf.write(str(cover_path), cover, SR, subtype="FLOAT")
        sf.write(str(wet_path), wet, SR, subtype="FLOAT")

        result = process(
            cover_dry_path=str(cover_path),
            out_dir=str(tmp_path / "out"),
            strategy=STRATEGIES[DEFAULT_STRATEGY],
            reference_wet_path=str(wet_path),
            song_name="fallback",
        )
        assert result.ok
        assert result.report["targets"]["active_ratio_db"] == -8.0
        assert any("implausibly high" in w for w in result.report["warnings"])


class TestFailureModes:
    def test_missing_reference_is_reported_not_raised(self, outputs, tmp_path):
        _, paths, _ = outputs
        result = process(
            cover_dry_path=paths["cover"],
            out_dir=str(tmp_path / "o4"),
            strategy=STRATEGIES[DEFAULT_STRATEGY],
        )
        assert result.ok is False
        assert "reference" in result.message.lower()

    def test_nonexistent_cover_path_is_reported(self, outputs, tmp_path):
        _, paths, _ = outputs
        result = process(
            cover_dry_path=str(tmp_path / "missing.wav"),
            out_dir=str(tmp_path / "o5"),
            strategy=STRATEGIES[DEFAULT_STRATEGY],
            reference_dry_path=paths["dry"],
            reference_reverb_path=paths["reverb"],
        )
        assert result.ok is False
        assert result.message

    def test_no_instrumental_means_no_remix(self, outputs, tmp_path):
        _, paths, _ = outputs
        result = process(
            cover_dry_path=paths["cover"],
            out_dir=str(tmp_path / "o6"),
            strategy=STRATEGIES[DEFAULT_STRATEGY],
            reference_dry_path=paths["dry"],
            reference_reverb_path=paths["reverb"],
            song_name="noinst",
        )
        assert result.ok
        assert result.remix is None


class TestOptions:
    @pytest.mark.parametrize("name", sorted(STRATEGIES))
    def test_every_strategy_runs(self, outputs, tmp_path, name):
        _, paths, _ = outputs
        result = process(
            cover_dry_path=paths["cover"],
            out_dir=str(tmp_path / f"strat_{name}"),
            strategy=STRATEGIES[name],
            reference_dry_path=paths["dry"],
            reference_reverb_path=paths["reverb"],
            song_name=name,
        )
        assert result.ok, result.message

    def test_wet_trim_reduces_the_reverb_level(self, outputs, tmp_path):
        _, paths, _ = outputs
        loud = process(
            cover_dry_path=paths["cover"],
            out_dir=str(tmp_path / "trim_up"),
            strategy=STRATEGIES[DEFAULT_STRATEGY],
            reference_dry_path=paths["dry"],
            reference_reverb_path=paths["reverb"],
            wet_trim_db=6.0,
            song_name="up",
        )
        quiet = process(
            cover_dry_path=paths["cover"],
            out_dir=str(tmp_path / "trim_dn"),
            strategy=STRATEGIES[DEFAULT_STRATEGY],
            reference_dry_path=paths["dry"],
            reference_reverb_path=paths["reverb"],
            wet_trim_db=-6.0,
            song_name="dn",
        )
        assert loud.ok and quiet.ok
        loud_ratio = loud.report["achieved"]["active_ratio_db"]
        quiet_ratio = quiet.report["achieved"]["active_ratio_db"]
        assert loud_ratio > quiet_ratio + 8.0

    def test_base_t60_override_reaches_the_report(self, outputs, tmp_path):
        _, paths, _ = outputs
        result = process(
            cover_dry_path=paths["cover"],
            out_dir=str(tmp_path / "t60"),
            strategy=STRATEGIES[DEFAULT_STRATEGY],
            reference_dry_path=paths["dry"],
            reference_reverb_path=paths["reverb"],
            base_t60_s=2.5,
            song_name="t60",
        )
        assert result.ok
        assert result.report["rir"]["t60_base_s"] == pytest.approx(2.5)

    def test_song_name_controls_the_output_prefix(self, outputs, tmp_path):
        _, paths, _ = outputs
        result = process(
            cover_dry_path=paths["cover"],
            out_dir=str(tmp_path / "named"),
            strategy=STRATEGIES[DEFAULT_STRATEGY],
            reference_dry_path=paths["dry"],
            reference_reverb_path=paths["reverb"],
            song_name="my_song",
        )
        assert result.ok
        assert os.path.basename(result.wet_vocal) == "my_song_wet_vocal.wav"

    def test_progress_callback_is_invoked(self, outputs, tmp_path):
        _, paths, _ = outputs
        seen: list[float] = []

        def on_progress(fraction, _desc):
            seen.append(fraction)

        result = process(
            cover_dry_path=paths["cover"],
            out_dir=str(tmp_path / "prog"),
            strategy=STRATEGIES[DEFAULT_STRATEGY],
            reference_dry_path=paths["dry"],
            reference_reverb_path=paths["reverb"],
            song_name="prog",
            progress=on_progress,
        )
        assert result.ok
        assert seen
        assert seen[-1] == pytest.approx(1.0)

    def test_output_directory_is_created(self, tmp_path, outputs):
        _, paths, _ = outputs
        target = tmp_path / "a" / "b" / "c"
        result = process(
            cover_dry_path=paths["cover"],
            out_dir=str(target),
            strategy=STRATEGIES[DEFAULT_STRATEGY],
            reference_dry_path=paths["dry"],
            reference_reverb_path=paths["reverb"],
            song_name="nested",
        )
        assert result.ok
        assert target.is_dir()

    def test_results_are_deterministic_for_the_same_song_name(self, outputs, tmp_path):
        _, paths, _ = outputs
        kwargs = {
            "cover_dry_path": paths["cover"],
            "strategy": STRATEGIES[DEFAULT_STRATEGY],
            "reference_dry_path": paths["dry"],
            "reference_reverb_path": paths["reverb"],
            "song_name": "same",
        }
        first = process(out_dir=str(tmp_path / "d1"), **kwargs)
        second = process(out_dir=str(tmp_path / "d2"), **kwargs)
        assert first.ok and second.ok
        a, _ = sf.read(first.rir, always_2d=True)
        b, _ = sf.read(second.rir, always_2d=True)
        assert np.allclose(a, b)


class TestEncodeMp3:
    def test_returns_none_when_ffmpeg_is_absent(self, monkeypatch, tmp_path):
        import srv.pipeline as pipeline

        def boom(*_args, **_kwargs):
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr(pipeline.subprocess, "run", boom)
        wav = tmp_path / "x.wav"
        sf.write(str(wav), np.zeros(1000), 44100)
        assert pipeline.encode_mp3(str(wav)) is None

    def test_returns_none_when_ffmpeg_fails(self, monkeypatch, tmp_path):
        import srv.pipeline as pipeline

        class Result:
            returncode = 1
            stderr = "boom"

        monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **k: Result())
        wav = tmp_path / "y.wav"
        sf.write(str(wav), np.zeros(1000), 44100)
        assert pipeline.encode_mp3(str(wav)) is None
