"""The reference solar spectrum, and resampling it into our pixel coordinates.

The atlas is a wavelength-domain curve at a resolution far finer than anything
this instrument can reach, so it is convolved down to the instrument's
resolution before use.  The blur is applied in wavelength, in nanometres, which
is where it physically belongs.

The direction of the resampling matters and is deliberate: the reference is
evaluated at the wavelength of each of *our* pixels, so it ends up on our X axis
rather than the other way round.  Our own spectrum is never resampled - it is
the measurement, it stays as it was measured.

Fetch or refresh the file with `python tools/fetch_reference.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .calib import gaussian_blur_1d

#: Where `load_default` looks.  Not inside the package: it is data, not code.
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "solar_reference.csv"
)

#: Gaussian FWHM in sigmas.
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass
class ReferenceSpectrum:
    """Wavelength (nm) against continuum-normalised intensity, plus line labels."""

    wavelength_nm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    intensity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    labels: List[Tuple[float, str]] = field(default_factory=list)
    path: str = ""
    step_nm: float = 0.1

    _blur_nm: float = -1.0
    _blurred: Optional[np.ndarray] = None

    @property
    def ok(self) -> bool:
        return self.wavelength_nm.size > 1

    @property
    def first_nm(self) -> float:
        return float(self.wavelength_nm[0]) if self.ok else 0.0

    @property
    def last_nm(self) -> float:
        return float(self.wavelength_nm[-1]) if self.ok else 0.0

    def blurred(self, fwhm_nm: float) -> np.ndarray:
        """The atlas convolved to an instrument resolution of `fwhm_nm`."""
        if not self.ok:
            return self.intensity
        if self._blurred is not None and abs(self._blur_nm - fwhm_nm) < 1e-9:
            return self._blurred
        sigma_samples = (fwhm_nm * FWHM_TO_SIGMA) / self.step_nm
        self._blurred = gaussian_blur_1d(self.intensity, sigma_samples)
        self._blur_nm = fwhm_nm
        return self._blurred

    def sample(self, wavelengths_nm: np.ndarray, fwhm_nm: float) -> np.ndarray:
        """Blurred intensity at each of the given wavelengths; NaN outside the atlas.

        The atlas is on a 0.1 nm grid, an order finer than one pixel of this
        spectrograph, so linear interpolation between its samples costs nothing
        once the curve has been blurred to the instrument resolution.
        """
        wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
        if not self.ok:
            return np.full(wavelengths_nm.shape, np.nan)
        values = np.interp(wavelengths_nm, self.wavelength_nm, self.blurred(fwhm_nm))
        outside = (wavelengths_nm < self.first_nm) | (wavelengths_nm > self.last_nm)
        return np.where(outside, np.nan, values)

    def nearest_label(self, wavelength_nm: float, tolerance_nm: float = 1.0) -> str:
        """Catalogue label within `tolerance_nm` of this wavelength, if there is one."""
        best, best_distance = "", tolerance_nm
        for centre, text in self.labels:
            distance = abs(centre - wavelength_nm)
            if distance <= best_distance:
                best, best_distance = text, distance
        return best


def load(path: str = DEFAULT_PATH) -> ReferenceSpectrum:
    """Read the CSV written by `tools/fetch_reference.py`. Empty result if missing."""
    wavelengths, intensities, labels = [], [], []
    try:
        with open(path, "r", encoding="ascii") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("wavelength"):
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                try:
                    wavelength, intensity = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                wavelengths.append(wavelength)
                intensities.append(intensity)
                if len(parts) > 2 and parts[2]:
                    labels.append((wavelength, parts[2]))
    except OSError:
        return ReferenceSpectrum(path=path)

    if len(wavelengths) < 2:
        return ReferenceSpectrum(path=path)

    wavelength_nm = np.asarray(wavelengths, dtype=np.float64)
    order = np.argsort(wavelength_nm)
    wavelength_nm = wavelength_nm[order]
    intensity = np.asarray(intensities, dtype=np.float64)[order]
    step = float(np.median(np.diff(wavelength_nm)))
    return ReferenceSpectrum(
        wavelength_nm=wavelength_nm,
        intensity=intensity,
        labels=sorted(labels),
        path=path,
        step_nm=step if step > 0 else 0.1,
    )
