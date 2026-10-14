# Cover Reverb Matching

Transfer the reverb character of a **reference vocal** onto a **dry cover vocal**.

You have a dry vocal take and a reference track whose reverb you like. This tool measures
that reverb and synthesises a matching one on your vocal, then (optionally) drops it back
onto the instrumental.

Everything runs locally. There is no model download, no GPU requirement, and no network
access.

<p align="center">
  <em>dry cover vocal + reference reverb → wet cover vocal + full remix</em>
</p>

---

## Why not just deconvolve the reference?

The obvious approach is: take the reference's dry and wet vocals, estimate the impulse
response, convolve it with your vocal. **It does not work on stem-separated material**, and
that was measured rather than assumed:

- The reference "reverb stem" produced by a separator is a **time-varying masked
  decomposition**, not a convolution of the dry stem.
- Coherence between the dry stem and the reverb stem measures **0.06-0.26**. A genuinely
  convolutive pair — the positive control, built from a real impulse response — measures
  **0.42-0.56**. No static impulse response can reproduce the relationship.
- The reverb stem's envelope tracks the dry vocal with only a 20-40 ms lag and vanishes
  (−100 dB) the instant the vocal stops; its decay rate is a plateau of −1 to +1 dB/s
  instead of the constant −60/T60 a real room would give.

So instead of estimating *a signal*, this tool matches three *measurable, stable* targets
and synthesises a reverb that satisfies them.

---

## Install

Requires **Python 3.10+**. No compiler, no CUDA, no model weights.

### Option 1 — one-click scripts

```bash
# Windows
scripts\launch.bat

# Linux / macOS
chmod +x scripts/launch.sh && ./scripts/launch.sh
```

The launcher finds an interpreter, bootstraps an environment if there is none, and opens
the web UI.

### Option 2 — set up the environment yourself

```bash
python bootstrap_env.py              # conda if available, otherwise a .venv
python bootstrap_env.py --manager venv
python bootstrap_env.py --with-dev   # also install pytest and ruff
```

`bootstrap_env.py` prefers conda (creating an env named `cover-reverb`) and falls back to
a plain `venv` when conda is not installed. Every runtime dependency is pure pip, so both
paths work on Windows, Linux and macOS.

### Option 3 — plain pip

```bash
pip install -e .          # runtime only
pip install -e ".[dev]"   # plus pytest, pytest-cov, ruff
```

ffmpeg is **optional** and only used for the MP3 export. All WAV output works without it.

---

## Usage

### Web UI

```bash
cover-reverb                          # or: python -m srv.runner
cover-reverb --port 8080
cover-reverb --host 0.0.0.0
```

Opening the app:

1. **Dry cover vocal** — your vocal take. Required.
2. **Reference material** — where the reverb comes from. Either the original **reverb
   component**, or the original **dry + wet** vocals (the difference gives the reverb).
   Supplying the original dry vocal too gives the most accurate level calibration.
3. **Instrumental** — optional. Adds a remix output and aligns the reverb-to-instrument
   balance with the reference.
4. Pick a strategy, optionally edit any parameter, press **Generate**.

The metrics table reports target vs achieved for every matched quantity. If 7860 is taken
the launcher picks the next free port and says so.

### Command line

```bash
cover-reverb --selftest                                  # end-to-end run on synthetic audio
cover-reverb --check                                     # environment check
cover-reverb --batch --roots "D:/songs" --strategy F1_intimate
cover-reverb --batch --all                               # every folder found by the scanner
```

### Python

```python
from srv import STRATEGIES, process

result = process(
    cover_dry_path="cover_vocal.wav",
    out_dir="out",
    strategy=STRATEGIES["F1_intimate"],
    reference_dry_path="ref_vocals_dry.wav",
    reference_reverb_path="ref_vocals_reverb.wav",
    instrumental_path="instrumental.wav",
    song_name="my_song",
)
print(result.ok, result.report["achieved"]["active_ratio_db"])
```

---

## Strategies

Two presets ship with the tool. Both are starting points — every parameter is editable in
the UI.

| Strategy         | Character                                                     | Predelay | Early reflections | Late onset / build | T60 × |
| ---------------- | ------------------------------------------------------------- | -------- | ----------------- | ------------------ | ----- |
| `F1_intimate`    | Close and clear. The vocal stays upfront.                     | 8 ms     | 12 (−3 dB)        | 40 / 420 ms        | 0.80  |
| `G0_extreme_dry` | Very dry. Only a thin tail, reverb barely occupies the vocal. | 5 ms     | 0                 | 55 / 700 ms        | 0.70  |

T60 is **base T60 × T60 scale**. The base T60 defaults to 1.6 s and is the one number that
cannot be measured from the reference (its decay is not exponential), so treat it as a
listening choice.

### Parameters

| Parameter            | Range           | Effect                                                    |
| -------------------- | --------------- | --------------------------------------------------------- |
| `predelay_ms`        | 0–120           | Longer keeps the vocal clearer and pushes the reverb back |
| `n_early`            | 0–24            | Number of discrete early reflections; 0 disables them     |
| `early_gain_db`      | −60–0           | Early reflection level                                    |
| `early_end_ms`       | 20–250          | How late the early reflections extend                     |
| `late_onset_ms`      | 0–300           | When the diffuse field starts                             |
| `late_build_ms`      | 50–1500         | How slowly it fills in; larger is sparser and cleaner     |
| `t60_scale`          | 0.3–2.5         | Multiplier on the base T60                                |
| `base_t60_s`         | 0.3–5.0         | Reference reverberation time                              |
| `wet_trim_db`        | −12–+12         | Global reverb level trim; go negative if it is too much   |
| `target_mode`        | active / global | Level calibration source; `active` is correct             |
| `match_spectrum`     | on/off          | Match the octave-band envelope                            |
| `match_stereo`       | on/off          | Match the stereo width                                    |
| `instrumental_match` | on/off          | Align the reverb/instrument balance                       |

---

## Outputs

| File                     | Contents                                                           |
| ------------------------ | ------------------------------------------------------------------ |
| `<name>_wet_vocal.wav`   | The reverb-treated cover vocal                                     |
| `<name>_remix.wav`       | Instrumental + wet vocal (only when an instrumental is given)      |
| `<name>_reverb_only.wav` | The reverb component alone — **listen to this to judge the level** |
| `<name>_rir.wav`         | The stereo impulse response that was used                          |

### Environment variables

| Variable              | Effect                                                           |
| --------------------- | ---------------------------------------------------------------- |
| `COVER_REVERB_OUT`    | Output directory (default: `./cover_reverb_out`)                 |
| `COVER_REVERB_SCAN`   | Folders the optional stem scanner searches (`;`-separated)       |
| `COVER_REVERB_ENV`    | Name of the conda environment to reuse (default: `cover-reverb`) |
| `COVER_REVERB_PYTHON` | Use this interpreter directly, skipping all detection            |
| `PIP_MIRROR`          | pip index URL for `bootstrap_env.py`, e.g. a domestic mirror     |

---

## How it works

### Matched targets

| Target                                     | What it controls | Why it matters                                                            |
| ------------------------------------------ | ---------------- | ------------------------------------------------------------------------- |
| **Active-region reverb/dry energy ratio**  | How much reverb  | See the calibration note below                                            |
| **Side/mid energy ratio, L/R correlation** | Stereo width     | Reference reverb is strongly decorrelated: side sits 2–3.7 dB *above* mid |
| **Octave-band spectral envelope**          | Reverb timbre    | 250–1000 Hz is strongest; 2 kHz and up rolls off 10–16 dB                 |

### Three findings worth knowing

**1. Calibrate on active regions, never on global energy.**
An earlier iteration matched the *total* reverb-to-dry energy ratio. The result sounded
like karaoke: while singing, the reverb measured **+0.24 to +0.78 dB** — louder than the
vocal — against a reference of **−5.6 to −10.0 dB**. A 7–10 dB error. The cause is that
the reference reverb's tails carry energy into the quiet blocks, so a global ratio reads
*lower* than the level actually present during singing. Matching it therefore
over-reverberates exactly where it matters.

**2. A mono source cannot produce `side > mid` without anti-phase channels.**
Convolving one mono vocal with two impulse responses gives `side ≤ mid`, always. The
reference measures `side > mid`, so the stereo RIR is built from a shared/independent
decomposition with an explicit sign flip:

```
l =  sqrt(rho)*common + sqrt(1-rho)*indep_l
r = -[sqrt(rho)*common + sqrt(1-rho)*indep_r]      ->  E_side/E_mid = (1+rho)/(1-rho)
```

**3. The late field must ramp in smoothly.**
An earlier version gated the late field with a hard step at its onset. That clicks, and it
inflates echo-density metrics. The current build ramps it continuously
(`1 - exp(-(t-onset)/build)`), which is also what a real room does.

### Order of operations

```
load cover + reference  ->  derive the reverb component
    ->  measure targets (active ratio, stereo, octave spectrum)
    ->  synthesise a stereo RIR
    ->  convolve the cover vocal
    ->  shape the octave spectrum
    ->  align the reverb-to-instrument balance      (changes the level)
    ->  calibrate the level on active regions       (must come last)
```

The last two steps cannot be swapped: the instrumental balance moves the reverb level, so
the active-region calibration has to run afterwards or it gets undone.

The `srv/pipeline.py` module docstring carries the same explanation next to the code.

---

## Development

```bash
pip install -e ".[dev]"

pytest -q                                   # 186 tests, ~26 s
pytest -q --cov=srv --cov-report=term-missing
ruff check .
```

Tests generate all of their audio with fixed seeds — **no binary test data is committed**.
A synthetic dry/reverb pair is built from a known impulse response, which makes every
assertion reproducible on any machine.

### Linting

`ruff check` is enforced in CI. `ruff format` is deliberately **not** used: at
`line-length = 110` it rewrites hand-wrapped argument lists into long single lines, which
is churn rather than improvement. A handful of overly opinionated rules (`RUF046`,
`RUF007`, `B905`, `B008`, `B017`, `SIM108`) are ignored in `pyproject.toml` with a
rationale next to each.

### CI

`.github/workflows/ci.yml` runs:

- **lint** — `ruff check` on Ubuntu / Python 3.12
- **test** — pytest across Linux (3.10–3.13), Windows and macOS, with ffmpeg installed on
  each platform, plus coverage
- **build** — sdist + wheel, then installs the wheel and runs `cover-reverb --check`

---

## Requirements and limitations

- Python 3.10+, numpy, scipy, soundfile, gradio.
- CPU only. Memory scales with track length; a 5 minute stereo track peaks in the low
  hundreds of MB.
- The reference reverb's **decay time cannot be measured** from stem-separated material,
  so `base_t60_s` is a listening decision, not a measurement.
- Supplying only the **wet** reference (no dry vocal) degrades both the spectral envelope
  and the level calibration. The tool warns when this happens and falls back to a
  conservative target.
- Matching is statistical, not sample-exact. The goal is that the result *sounds like* the
  same space, not that it nulls against the reference.

---

## Project layout

```
srv/
  pipeline.py       core: metrics, RIR synthesis, the process() pipeline
  ui.py             Gradio interface and folder scanning
  runner.py         environment check, CLI, self-test, batch mode
tests/              pytest suite; all fixtures synthesise their own audio
scripts/
  launch.bat        Windows launcher (ASCII-only: cmd parses .bat in the OEM codepage)
  launch.sh         Linux / macOS launcher
bootstrap_env.py    conda-first, venv-fallback environment setup
launch.py           shared launcher logic the scripts delegate to
```

---

## License

MIT
