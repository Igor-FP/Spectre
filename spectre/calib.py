"""Geometry calibration of the spectrum band: its tilt against the X axis.

Implements exactly the specified method, and nothing else.

1. Scan the image top to bottom, one scan line per row (1 px interval), but the
   lines are not along X - they are tilted: the line numbered `Y` starts at
   (0, Y) at the left edge of the crop and has slope `dy/dx = tan(L)`.
2. For every angle `L` from -10 to +10 degrees in 0.01 degree steps, take the
   *arithmetic mean* of all pixels along each scan line and store it as
   `array[L][Y]`: for each angle, a 1-D projection of the 2-D image by
   summation.  The mean, not a median: the spectrum covers only part of the
   width, and the mean still shows it as a bump.
3. In each projection take the 0.1 % and 99.9 % percentiles and the level a
   quarter of the way from the low one to the high one.  Scanning the 1-D array
   from the top and from the bottom, find the first crossing of that level; the
   distance between them is the width of the bump.
4. Gaussian-filter the width-versus-angle array (5 samples wide) and take the
   minimum width.  Its angle is the direction of the spectrum, and the two
   crossings at that angle bound the width of the band.

Everything runs on the crop, so coordinates in `BandResult` are relative to the
crop; the caller shifts them into full-frame coordinates.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

Progress = Optional[Callable[[float], None]]
Cancelled = Optional[Callable[[], bool]]


@dataclass
class BandParams:
    """Search settings."""

    angle_range_deg: float = 10.0  # scan -range .. +range
    angle_step_deg: float = 0.01
    lo_percentile: float = 0.1
    hi_percentile: float = 99.9
    #: Where between the low and the high level the band width is measured.
    level_fraction: float = 0.25
    smooth_window: int = 5  # samples of the Gaussian filter on width(angle)


@dataclass
class BandResult:
    ok: bool = False
    message: str = ""
    warning: str = ""
    elapsed_s: float = 0.0

    # Geometry in crop pixels; the angle is positive when the band runs
    # downwards to the right (Y points down).  `reference_x` is where the
    # scan lines are anchored, i.e. the left edge of the crop.
    angle_deg: float = 0.0
    tan_angle: float = 0.0
    centre_y: float = 0.0
    edge_lo_y: float = 0.0
    edge_hi_y: float = 0.0
    fwhm_px: float = 0.0
    reference_x: float = 0.0
    x_from: float = 0.0  # horizontal span the measurement covers
    x_to: float = 0.0

    # diagnostics
    contrast: float = 0.0  # high minus low level of the projection
    snr: float = 0.0  # contrast / projection noise
    angles_evaluated: int = 0
    fwhm_min: float = 0.0  # range of the filtered FWHM curve over the scan:
    fwhm_max: float = 0.0  # how much the curve actually moves

    # curves for the UI
    angles_deg: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    fwhm_curve: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    fwhm_curve_raw: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    profile: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))

    def shifted(self, dx: float, dy: float) -> "BandResult":
        """A copy with the geometry moved into full-frame coordinates."""
        return replace(
            self,
            centre_y=self.centre_y + dy,
            edge_lo_y=self.edge_lo_y + dy,
            edge_hi_y=self.edge_hi_y + dy,
            reference_x=self.reference_x + dx,
            x_from=self.x_from + dx,
            x_to=self.x_to + dx,
        )

    def edge_points(self):
        """The two band edges as ((x0,y0),(x1,y1)) pairs over the measured span."""
        return [
            (
                (self.x_from, edge + (self.x_from - self.reference_x) * self.tan_angle),
                (self.x_to, edge + (self.x_to - self.reference_x) * self.tan_angle),
            )
            for edge in (self.edge_lo_y, self.edge_hi_y)
        ]

    def centre_points(self):
        return (
            (self.x_from, self.centre_y + (self.x_from - self.reference_x) * self.tan_angle),
            (self.x_to, self.centre_y + (self.x_to - self.reference_x) * self.tan_angle),
        )


# ---------------------------------------------------------------------------
# Step 1 and 2: tilted scan lines, arithmetic mean along each line
# ---------------------------------------------------------------------------


def column_cumsum(image: np.ndarray) -> np.ndarray:
    """Running sum along X with a leading zero column: C[:, k] = sum of x < k.

    float64 so that differences of large partial sums stay exact for 16-bit data.
    """
    height, width = image.shape
    cumulative = np.zeros((height, width + 1), dtype=np.float64)
    np.cumsum(image, axis=1, dtype=np.float64, out=cumulative[:, 1:])
    return cumulative


def scan_line_means(cumulative: np.ndarray, slope: float, reference_x: float = 0.0):
    """Arithmetic mean along each tilted scan line. Returns (means, counts).

    `means[y]` is the mean of the pixels on the line through (reference_x, y)
    with dy/dx = slope; `counts[y]` how many columns contributed (fewer for the
    lines that leave the frame).

    Columns that share the same rounded row offset are summed in one step
    through the running sum, so a whole projection costs O(runs x height)
    instead of O(width x height).
    """
    height = cumulative.shape[0]
    width = cumulative.shape[1] - 1
    offsets = np.rint((np.arange(width) - reference_x) * slope).astype(np.intp)

    changes = np.flatnonzero(offsets[1:] != offsets[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [width]))

    total = np.zeros(height, dtype=np.float64)
    count = np.zeros(height, dtype=np.float64)
    for start, end in zip(starts, ends):
        offset = int(offsets[start])
        if abs(offset) >= height:
            continue
        segment = cumulative[:, end] - cumulative[:, start]
        columns = float(end - start)
        if offset >= 0:
            total[: height - offset] += segment[offset:]
            count[: height - offset] += columns
        else:
            total[-offset:] += segment[: height + offset]
            count[-offset:] += columns

    means = total / np.maximum(count, 1.0)
    return means, count


# ---------------------------------------------------------------------------
# Step 3: width of the bump in one projection
# ---------------------------------------------------------------------------


#: The peak of the projection is looked for on a copy smoothed this wide, so
#: that a single hot row cannot win it.
PEAK_SMOOTH_SAMPLES = 9


def _crossing_outwards(profile: np.ndarray, peak: int, direction: int, level: float) -> float:
    """Where the profile drops below `level`, walking away from its peak.

    Searched outwards from the peak rather than inwards from the ends of the
    array: anything else that pokes above the level - stray light, the zero
    order, a hot row - would otherwise be measured instead of the band, and the
    result would depend on how much of it happens to be inside the crop.

    Sub-pixel: interpolates between the last sample above the level and the
    first one below it.
    """
    length = profile.size
    index = peak
    while 0 <= index + direction < length and profile[index + direction] >= level:
        index += direction
    outside = index + direction
    if not 0 <= outside < length:
        return float(index)  # the band runs into the end of the projection
    inner, outer = profile[index], profile[outside]
    span = inner - outer
    fraction = (inner - level) / span if span > 0 else 0.0
    return index + direction * fraction


@dataclass
class BumpMeasure:
    fwhm: float
    low_y: float
    high_y: float
    mid_level: float
    lo_level: float
    hi_level: float

    @property
    def centre(self) -> float:
        return 0.5 * (self.low_y + self.high_y)


def measure_bump(
    profile: np.ndarray,
    lo_percentile: float,
    hi_percentile: float,
    level_fraction: float = 0.25,
) -> BumpMeasure:
    """Bump width at `level_fraction` of the way from the low to the high level.

    The two crossings are found outwards from the peak of the projection, so the
    number is the width of the band itself and not the distance between whatever
    else in the crop rises above the level.
    """
    lo_level, hi_level = np.percentile(profile, [lo_percentile, hi_percentile])
    mid_level = lo_level + level_fraction * (hi_level - lo_level)
    peak = int(np.argmax(gaussian_smooth(profile, PEAK_SMOOTH_SAMPLES)))
    low_y = _crossing_outwards(profile, peak, -1, mid_level)
    high_y = _crossing_outwards(profile, peak, +1, mid_level)
    return BumpMeasure(
        fwhm=high_y - low_y,
        low_y=low_y,
        high_y=high_y,
        mid_level=float(mid_level),
        lo_level=float(lo_level),
        hi_level=float(hi_level),
    )


# ---------------------------------------------------------------------------
# Step 4: smooth width(angle) and take the minimum
# ---------------------------------------------------------------------------


def gaussian_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Gaussian filter with edge padding; `window` samples wide (odd)."""
    if window < 3 or values.size < 3:
        return values.astype(np.float64, copy=True)
    window = int(window) | 1
    radius = window // 2
    sigma = max(window / 5.0, 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values.astype(np.float64), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def find_band(
    image: np.ndarray,
    params: Optional[BandParams] = None,
    progress: Progress = None,
    cancel: Cancelled = None,
) -> BandResult:
    """Angle of the spectrum band in `image` (already cropped)."""
    params = params or BandParams()
    started = time.perf_counter()
    result = BandResult()

    if image is None or image.ndim != 2 or min(image.shape) < 16:
        result.message = "crop is too small to measure"
        return result

    width = image.shape[1]
    result.x_from, result.x_to = 0.0, float(width - 1)
    cumulative = column_cumsum(image)

    step = max(params.angle_step_deg, 1e-4)
    span = abs(params.angle_range_deg)
    angles = np.arange(-span, span + step * 0.5, step)
    widths = np.full(angles.size, np.nan, dtype=np.float64)

    for index, angle in enumerate(angles):
        if cancel is not None and cancel():
            result.message = "cancelled"
            return result
        profile, _ = scan_line_means(cumulative, math.tan(math.radians(float(angle))))
        bump = measure_bump(
            profile, params.lo_percentile, params.hi_percentile, params.level_fraction
        )
        if np.isfinite(bump.fwhm) and bump.fwhm > 0:
            widths[index] = bump.fwhm
        if progress is not None and (index % 16 == 0 or index == angles.size - 1):
            progress(0.95 * (index + 1) / angles.size)

    measured = np.isfinite(widths)
    if not measured.any():
        result.elapsed_s = time.perf_counter() - started
        result.message = "no bump in any projection: is the band inside the crop?"
        return result

    # Angles where the bump could not be measured must not win the minimum.
    filled = np.where(measured, widths, np.max(widths[measured]))
    smoothed = gaussian_smooth(filled, params.smooth_window)
    best_index = int(np.argmin(smoothed))
    best_angle = float(angles[best_index])

    result.angles_evaluated = int(angles.size)
    result.angles_deg = angles.astype(np.float32)
    result.fwhm_curve = smoothed.astype(np.float32)
    result.fwhm_curve_raw = filled.astype(np.float32)
    result.fwhm_min = float(smoothed.min())
    result.fwhm_max = float(smoothed.max())

    # The band edges are the two crossings at the chosen angle.
    profile, _ = scan_line_means(cumulative, math.tan(math.radians(best_angle)))
    bump = measure_bump(
        profile, params.lo_percentile, params.hi_percentile, params.level_fraction
    )
    result.profile = profile.astype(np.float32)
    result.contrast = bump.hi_level - bump.lo_level
    noise = float(np.median(np.abs(np.diff(profile)))) * 1.4826 / math.sqrt(2.0)
    result.snr = result.contrast / noise if noise > 0 else 0.0
    result.elapsed_s = time.perf_counter() - started

    if result.contrast <= 0.0:
        result.message = "no band: the projection is flat"
        return result
    if not np.isfinite(bump.fwhm) or bump.fwhm <= 1.0:
        result.message = "no band: the projection never crosses its mid level"
        return result
    if bump.fwhm >= 0.9 * profile.size:
        result.message = "no band: the bump fills the whole crop height"
        return result

    result.ok = True
    result.angle_deg = best_angle
    result.tan_angle = math.tan(math.radians(best_angle))
    result.centre_y = bump.centre
    result.edge_lo_y = bump.low_y
    result.edge_hi_y = bump.high_y
    result.fwhm_px = bump.fwhm
    result.message = "ok"
    if result.snr < 30.0:
        result.warning = f"low contrast (contrast/noise {result.snr:.0f})"
    return result


# ---------------------------------------------------------------------------
# Shear: the direction of the spectral lines, i.e. the Y axis of the spectrum
# ---------------------------------------------------------------------------
#
# The goal of the whole geometry calibration is the basis of the spectrum's
# coordinate system: X along the wavelength axis (that is the band direction,
# from `find_band`) and Y along the spectral lines.  The two are not
# perpendicular to each other - the slit and the grating are not exactly
# aligned - and neither is perpendicular to the frame edges, because the camera
# is not screwed on perfectly.
#
# Same idea as the band angle, turned by 90 degrees: try a range of line
# directions, and for each one sum the pixels along that direction across the
# band width, producing a 1-D projection along the wavelength axis.  At the true
# direction the spectral lines add up in phase and the projection is sharpest;
# off it they smear.  Sharpness is measured as the RMS of the difference of two
# Gaussian blurs of the projection (scales 1 and 4 times a multiplier) - a band
# pass that ignores the continuum and the uneven slit illumination, and needs no
# line detection.
#
# Everything is float: coordinates are computed in floating point and each pixel
# lands in the nearest bin of the projection.  No integer pre-shifts of the
# image, no interpolation - it averages out over the thousands of pixels that
# fall into each bin.


@dataclass
class ShearParams:
    """Search settings. Only `blur_scale` is meant to be user-visible."""

    angle_range_deg: float = 10.0  # scan -range .. +range around the band normal
    angle_step_deg: float = 0.05
    blur_scale: float = 1.0  # sharpness uses blur(1 x scale) - blur(4 x scale)


@dataclass
class ShearResult:
    ok: bool = False
    message: str = ""
    elapsed_s: float = 0.0

    #: Tilt of the spectral lines against the frame Y axis, degrees; positive
    #: means the lines lean towards +X as Y increases.
    line_tilt_deg: float = 0.0
    #: How far the spectrum's X and Y axes are from perpendicular, degrees.
    shear_deg: float = 0.0
    #: du/dv: how far the wavelength coordinate slides per pixel across the band.
    shear_tan: float = 0.0

    # The basis itself, unit vectors in frame coordinates (y points down).
    axis_x: tuple = (1.0, 0.0)  # along the wavelength axis
    axis_y: tuple = (0.0, 1.0)  # along the spectral lines
    band_angle_deg: float = 0.0  # what the X axis came from

    # diagnostics
    columns_used: int = 0
    rows_used: int = 0
    sharpness_min: float = 0.0
    sharpness_max: float = 0.0
    angles_evaluated: int = 0

    # curves for the UI
    angles_deg: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    sharpness: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    profile: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))


def gaussian_blur_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a 1-D profile, with edge padding."""
    if sigma <= 0.0:
        return values.astype(np.float64, copy=True)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values.astype(np.float64), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def profile_sharpness(profile: np.ndarray, scale: float) -> float:
    """RMS difference between a fine and a coarse blur of the profile."""
    fine = gaussian_blur_1d(profile, 1.0 * scale)
    coarse = gaussian_blur_1d(profile, 4.0 * scale)
    difference = fine - coarse
    return float(math.sqrt(float(np.dot(difference, difference)) / difference.size))


class BandStrip:
    """The pixels between the band edges, ready to be projected along the lines.

    Shared by the shear search and by the spectrum extraction so that both use
    exactly the same geometry.  `across` is every pixel's exact (float) distance
    from the band centre line, `along` its column.
    """

    def __init__(self, image: np.ndarray, band: BandResult, span_deg: float):
        height, width = image.shape
        self.slope = band.tan_angle
        columns = np.arange(width, dtype=np.float64)
        centre = band.centre_y + (columns - band.reference_x) * self.slope
        self.half = int(band.fwhm_px * 0.5)
        if self.half < 4:
            raise ValueError("band is too narrow to work with")
        anchor = np.rint(centre).astype(np.intp)
        inside = (anchor - self.half >= 0) & (anchor + self.half <= height - 1)
        used = np.flatnonzero(inside)
        if used.size < 64:
            raise ValueError("the band leaves the crop: not enough columns to work with")

        rows = np.arange(-self.half, self.half + 1, dtype=np.intp)[:, None] + anchor[used][None, :]
        self.values = image[rows, used[None, :]].astype(np.float64)
        self.across = rows - centre[used][None, :]
        self.along = np.repeat(used.astype(np.float64)[None, :], rows.shape[0], axis=0)
        self.rows = int(rows.shape[0])
        self.columns = int(used.size)

        # Trim the ends, where a tilted line only crosses part of the band. The
        # same interval is used at every angle, so results stay comparable.
        reach = abs(self.shear_of(span_deg)) * self.half
        margin = int(math.ceil(max(reach, abs(self.shear_of(-span_deg)) * self.half))) + 1
        self.pad = margin + 2
        self.length = width + 2 * self.pad
        self.first = int(used[0]) + self.pad + margin
        self.last = int(used[-1]) + self.pad - margin
        if self.last - self.first < 32:
            raise ValueError("not enough columns left after trimming the ends")

    def shear_of(self, angle_deg: float) -> float:
        """du/dv for lines tilted `angle_deg` from the frame Y axis."""
        tangent = math.tan(math.radians(float(angle_deg)))
        return tangent / (1.0 - self.slope * tangent)

    def project(self, shear_tan: float):
        """Mean along the tilted lines. Returns the profile over the kept interval."""
        coordinate = self.along - self.across * shear_tan
        bins = np.rint(coordinate).astype(np.intp) + self.pad
        np.clip(bins, 0, self.length - 1, out=bins)
        flat = bins.ravel()
        flux = np.bincount(flat, weights=self.values.ravel(), minlength=self.length)
        count = np.bincount(flat, minlength=self.length)
        window = slice(self.first, self.last)
        return flux[window] / np.maximum(count[window], 1.0)

    @property
    def first_column(self) -> float:
        """Column of the frame the first sample of a profile corresponds to."""
        return float(self.first - self.pad)


def find_shear(
    image: np.ndarray,
    band: BandResult,
    params: Optional[ShearParams] = None,
    progress: Progress = None,
    cancel: Cancelled = None,
) -> ShearResult:
    """Direction of the spectral lines in `image` (the crop), given the band.

    `band` must be in the same coordinates as `image`.
    """
    params = params or ShearParams()
    started = time.perf_counter()
    result = ShearResult()
    if image is None or image.ndim != 2:
        result.message = "no image"
        return result
    if band is None or not band.ok or band.fwhm_px < 8.0:
        result.message = "measure the band angle first"
        return result

    span = abs(params.angle_range_deg)
    try:
        strip = BandStrip(image, band, span)
    except ValueError as exc:
        result.message = str(exc)
        return result

    result.band_angle_deg = band.angle_deg
    result.rows_used = strip.rows
    result.columns_used = strip.columns

    step = max(params.angle_step_deg, 1e-3)
    angles = np.arange(-span, span + step * 0.5, step)

    sharpness = np.zeros(angles.size, dtype=np.float64)
    for index, angle in enumerate(angles):
        if cancel is not None and cancel():
            result.message = "cancelled"
            return result
        profile = strip.project(strip.shear_of(angle))
        sharpness[index] = profile_sharpness(profile, params.blur_scale)
        if progress is not None and (index % 8 == 0 or index == angles.size - 1):
            progress(0.97 * (index + 1) / angles.size)

    best_index = int(np.argmax(sharpness))
    best_angle = float(angles[best_index])
    best_profile = strip.project(strip.shear_of(best_angle))

    band_radians = math.radians(band.angle_deg)
    line_radians = math.radians(best_angle)
    result.ok = True
    result.message = "ok"
    result.line_tilt_deg = best_angle
    result.shear_deg = band.angle_deg + best_angle  # departure from perpendicular
    result.shear_tan = strip.shear_of(best_angle)
    result.axis_x = (math.cos(band_radians), math.sin(band_radians))
    result.axis_y = (math.sin(line_radians), math.cos(line_radians))
    result.angles_evaluated = int(angles.size)
    result.angles_deg = angles.astype(np.float32)
    result.sharpness = sharpness.astype(np.float32)
    result.sharpness_min = float(sharpness.min())
    result.sharpness_max = float(sharpness.max())
    result.profile = best_profile.astype(np.float32)
    result.elapsed_s = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------
# Extracting the spectrum
# ---------------------------------------------------------------------------


@dataclass
class Spectrum:
    """One-dimensional spectrum: the band averaged along the spectral lines."""

    ok: bool = False
    message: str = ""
    values: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    first_column: float = 0.0  # frame column the first sample belongs to
    rows_averaged: int = 0
    line_tilt_deg: float = 0.0
    band_angle_deg: float = 0.0
    frame_index: int = 0
    elapsed_s: float = 0.0

    @property
    def length(self) -> int:
        return int(self.values.size)


def extract_spectrum(
    image: np.ndarray, band: BandResult, shear: ShearResult, span_deg: float = 10.0
) -> Spectrum:
    """Average the pixels between the band edges along the spectral-line vector.

    Walks the wavelength axis and, for every position, averages along the
    direction of the spectral lines - the band normal tilted by the measured
    shear - between the two band edges.  `band` and `shear` must refer to the
    same coordinates as `image`.
    """
    started = time.perf_counter()
    if image is None or image.ndim != 2:
        return Spectrum(message="no image")
    if band is None or not band.ok:
        return Spectrum(message="measure the band angle first")
    if shear is None or not shear.ok:
        return Spectrum(message="measure the shear first")
    try:
        strip = BandStrip(image, band, span_deg)
    except ValueError as exc:
        return Spectrum(message=str(exc))

    values = strip.project(strip.shear_of(shear.line_tilt_deg))
    return Spectrum(
        ok=True,
        message="ok",
        values=values.astype(np.float32),
        first_column=strip.first_column,
        rows_averaged=strip.rows,
        line_tilt_deg=shear.line_tilt_deg,
        band_angle_deg=band.angle_deg,
        elapsed_s=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


class BandFinder:
    """Runs `find_band` off the UI thread, with progress and cancellation."""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._progress = 0.0
        self._cancel = False
        self._result: Optional[BandResult] = None

    def start(self, image: np.ndarray, params: BandParams) -> bool:
        if self.running:
            return False
        with self._lock:
            self._progress = 0.0
            self._cancel = False
            self._result = None
        frame = np.array(image, copy=True)  # the grabber may reuse its buffer
        self._thread = threading.Thread(
            target=self._run, args=(frame, params), name="band-finder", daemon=True
        )
        self._thread.start()
        return True

    def _run(self, image: np.ndarray, params: BandParams) -> None:
        try:
            result = find_band(image, params, progress=self._set_progress, cancel=self._cancelled)
        except Exception as exc:  # never take the UI down with us
            result = BandResult(message=f"{type(exc).__name__}: {exc}")
        with self._lock:
            self._result = result
            self._progress = 1.0

    def _set_progress(self, value: float) -> None:
        with self._lock:
            self._progress = value

    def _cancelled(self) -> bool:
        with self._lock:
            return self._cancel

    def cancel(self) -> None:
        with self._lock:
            self._cancel = True

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def progress(self) -> float:
        with self._lock:
            return self._progress

    def take_result(self) -> Optional[BandResult]:
        """Hand over the finished result (once)."""
        with self._lock:
            result, self._result = self._result, None
        if result is not None:
            self._thread = None
        return result


class ShearFinder:
    """Runs `find_shear` off the UI thread, with progress and cancellation."""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._progress = 0.0
        self._cancel = False
        self._result: Optional[ShearResult] = None

    def start(self, image: np.ndarray, band: BandResult, params: ShearParams) -> bool:
        if self.running:
            return False
        with self._lock:
            self._progress = 0.0
            self._cancel = False
            self._result = None
        frame = np.array(image, copy=True)  # the grabber may reuse its buffer
        self._thread = threading.Thread(
            target=self._run, args=(frame, band, params), name="shear-finder", daemon=True
        )
        self._thread.start()
        return True

    def _run(self, image: np.ndarray, band: BandResult, params: ShearParams) -> None:
        try:
            result = find_shear(
                image, band, params, progress=self._set_progress, cancel=self._cancelled
            )
        except Exception as exc:  # never take the UI down with us
            result = ShearResult(message=f"{type(exc).__name__}: {exc}")
        with self._lock:
            self._result = result
            self._progress = 1.0

    def _set_progress(self, value: float) -> None:
        with self._lock:
            self._progress = value

    def _cancelled(self) -> bool:
        with self._lock:
            return self._cancel

    def cancel(self) -> None:
        with self._lock:
            self._cancel = True

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def progress(self) -> float:
        with self._lock:
            return self._progress

    def take_result(self) -> Optional[ShearResult]:
        """Hand over the finished result (once)."""
        with self._lock:
            result, self._result = self._result, None
        if result is not None:
            self._thread = None
        return result
