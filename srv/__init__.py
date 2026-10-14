"""Cover reverb matching: transfer a reference vocal's reverb onto a dry cover vocal.

The public surface is small on purpose:

* :class:`ReverbStrategy` / :data:`STRATEGIES` -- the two built-in presets plus every
  tunable parameter.
* :func:`process` -- the whole pipeline, one call.
* :func:`build_ui` -- a Gradio interface around the pipeline.
"""

from .pipeline import (
    BAND_EDGES,
    BAND_NAMES,
    DEFAULT_STRATEGY,
    PARAM_RANGES,
    STRATEGIES,
    PipelineResult,
    ReverbStrategy,
    active_ratio_db,
    apply_octave_spectrum,
    clamp_params,
    encode_mp3,
    lr_corr,
    make_rir,
    octave_spectrum_db,
    process,
    rir_metrics,
    sidemid_db,
    validate_strategies,
)

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
    "clamp_params",
    "encode_mp3",
    "lr_corr",
    "make_rir",
    "octave_spectrum_db",
    "process",
    "rir_metrics",
    "sidemid_db",
    "validate_strategies",
]

__version__ = "0.1.0"
