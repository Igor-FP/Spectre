"""Persisted UI/camera settings (a small JSON file next to the app)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

#: Override with SPECTRE_SETTINGS to keep a run from touching the real file.
SETTINGS_PATH = os.environ.get(
    "SPECTRE_SETTINGS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json"),
)


@dataclass
class Settings:
    # camera
    last_camera: str = ""
    # -1 means "whatever the camera already has"; real values land here as soon
    # as the user touches a control, and are restored on the next run
    exposure_us: int = -1
    gain: int = -1
    offset: int = -1
    bandwidth: int = -1
    high_speed: bool = False

    # display stretch
    auto_stretch: bool = True
    black: float = 0.0
    white: float = 1.0
    midtone: float = 0.5
    lo_percentile: float = 0.2
    hi_percentile: float = 99.9

    # dark / bias correction
    #: With no dark for the current gain and exposure, a flat level is taken off
    #: instead, so that ratios between ADU values stay honest.
    use_bias: bool = False
    bias_level: int = 2000

    # saving frames
    save_fits: bool = True
    save_npy: bool = True
    save_full_frame: bool = False

    # crop (region of interest): every spectrum algorithm works inside it.
    # -1 means "not set yet" and is initialised to the full frame.
    crop_x0: int = -1
    crop_y0: int = -1
    crop_x1: int = -1
    crop_y1: int = -1
    show_full_frame: bool = True

    # band geometry search
    band_angle_range_deg: float = 10.0
    band_angle_step_deg: float = 0.01
    band_lo_percentile: float = 0.1
    band_hi_percentile: float = 99.9
    band_smooth_window: int = 5

    # last calibration result, so the overlay survives a restart
    band_valid: bool = False
    band_angle_deg: float = 0.0
    band_centre_y: float = 0.0
    band_edge_lo_y: float = 0.0
    band_edge_hi_y: float = 0.0
    band_fwhm_px: float = 0.0
    band_reference_x: float = 0.0
    band_x_from: float = 0.0
    band_x_to: float = 0.0

    # shear: direction of the spectral lines (the Y axis of the spectrum basis)
    shear_blur_scale: float = 1.0
    shear_valid: bool = False
    shear_line_tilt_deg: float = 0.0

    # extracted spectrum window
    show_spectrum: bool = False
    #: Spectra averaged together before anything is shown or exported. 1 = off.
    spectrum_average: int = 1
    spectrum_height: float = 300.0
    spectrum_strip_ratio: float = 20.0

    # wavelength calibration: the reference strip and the anchor points
    show_reference: bool = True
    reference_blur_nm: float = 2.0
    #: Range the reference is spread over until there are points to fit.
    reference_from_nm: float = 380.0
    reference_to_nm: float = 780.0
    wavelength_max_degree: int = 3
    #: [{"x_px": ..., "nm": ..., "label": ...}], X in full-frame columns.
    wavelength_anchors: list = field(default_factory=list)
    # The fitted formula itself: coefficients highest power first, evaluated in
    # t = (x - x_ref) / x_scale.
    wavelength_valid: bool = False
    wavelength_coefficients: list = field(default_factory=list)
    wavelength_x_ref: float = 0.0
    wavelength_x_scale: float = 1.0

    # view / window
    window_width: int = 1920
    window_height: int = 1040
    panel_width: float = 400.0
    right_panel_width: float = 400.0
    show_band_overlay: bool = True
    ui_scale: float = 1.0
    fit_to_window: bool = True
    zoom: float = 1.0
    show_help: bool = False

    @classmethod
    def load(cls, path: str = SETTINGS_PATH) -> "Settings":
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str = SETTINGS_PATH) -> None:
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(asdict(self), handle, indent=2)
        except OSError:
            pass  # a read-only checkout is not worth crashing over
