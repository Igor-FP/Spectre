"""X -> wavelength: the anchor points and the polynomial fitted through them.

The model is a polynomial of X, of a degree that grows with the number of anchor
points and stops at `max_degree`.  Which model this ought to be in the end is an
open question - see `docs/TZ_Wavelength.md`; nothing here assumes the answer
beyond the polynomial being easy to replace.

X is a **full-frame column**, not an index into the spectrum array, so the
solution survives moving the crop or re-extracting the spectrum.

The polynomial is evaluated in a normalised variable t = (x - x_ref) / x_scale,
with t running over about -1..+1 across the spectrum.  Fitting a cubic in raw
column numbers is badly conditioned; this costs one subtraction and one divide
and is the reason the two constants are saved along with the coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

#: Highest polynomial degree offered.  Above this the curve starts following the
#: click noise rather than the dispersion.
MAX_DEGREE = 3

#: A calibration is either complete or it does not exist - there is no half of
#: one.  Below this many points nothing is stored, nothing is restored, and
#: nothing downstream is allowed to call itself calibrated.  Fewer points still
#: give a mapping to draw the reference with while it is being made; that is a
#: working sketch, not a calibration.
POINTS_FOR_CALIBRATION = 3


@dataclass
class Anchor:
    """One identified feature: where it sits on our X axis, and its wavelength."""

    x_px: float  # full-frame column
    wavelength_nm: float
    label: str = ""
    #: Order the point was made in.  The list is kept sorted by column for the
    #: table, so this is the only way back to "the one added last" for undo, and
    #: it is saved so that undo still means that after a restart.
    added: int = 0

    def as_dict(self) -> dict:
        return {
            "x_px": float(self.x_px),
            "nm": float(self.wavelength_nm),
            "label": self.label,
            "added": int(self.added),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Anchor":
        return cls(
            x_px=float(data.get("x_px", 0.0)),
            wavelength_nm=float(data.get("nm", 0.0)),
            label=str(data.get("label", "")),
            added=int(data.get("added", 0)),
        )


@dataclass
class Solution:
    """A usable X -> wavelength mapping, however it was arrived at."""

    ok: bool = False
    message: str = ""
    #: "manual"  - no anchors yet, the range set by hand across the spectrum;
    #: "shifted" - one anchor: the manual dispersion, slid onto that point;
    #: "fit"     - two or more anchors, least squares.
    kind: str = "manual"
    degree: int = 1
    coefficients: np.ndarray = field(default_factory=lambda: np.zeros(0))
    x_ref: float = 0.0
    x_scale: float = 1.0
    points_used: int = 0
    residuals_nm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rms_nm: float = 0.0

    def lambda_of_x(self, x) -> np.ndarray:
        """Wavelength in nm at one column or an array of them."""
        t = (np.asarray(x, dtype=np.float64) - self.x_ref) / self.x_scale
        return np.polyval(self.coefficients, t)

    def nm_per_px_at(self, x: float) -> float:
        """Local dispersion, for the readout."""
        derivative = np.polyder(self.coefficients)
        if derivative.size == 0:
            return 0.0
        t = (float(x) - self.x_ref) / self.x_scale
        return float(np.polyval(derivative, t)) / self.x_scale

    def describe(self) -> str:
        """The formula as text, for the panel and the clipboard."""
        terms = []
        order = self.coefficients.size - 1
        for power, value in enumerate(self.coefficients):
            exponent = order - power
            if exponent == 0:
                terms.append(f"{value:+.6g}")
            elif exponent == 1:
                terms.append(f"{value:+.6g}*t")
            else:
                terms.append(f"{value:+.6g}*t^{exponent}")
        return (
            "lambda = " + " ".join(terms)
            + f",  t = (x - {self.x_ref:.2f}) / {self.x_scale:.2f}"
        )


def solve(
    anchors: Sequence[Anchor],
    x_from: float,
    x_to: float,
    manual_from_nm: float,
    manual_to_nm: float,
    max_degree: int = MAX_DEGREE,
) -> Solution:
    """Best mapping the current anchors allow, over the span x_from..x_to.

    With no anchors the manual range is spread linearly over the span, which is
    what makes the reference visible in the first place - there is nothing to
    click on otherwise.  One anchor keeps that dispersion and slides it onto the
    point; from two anchors on the dispersion itself is fitted, and the degree
    follows the number of points up to `max_degree`.
    """
    x_ref = 0.5 * (float(x_from) + float(x_to))
    x_scale = max(1.0, 0.5 * (float(x_to) - float(x_from)))
    slope = 0.5 * (float(manual_to_nm) - float(manual_from_nm))
    centre = 0.5 * (float(manual_to_nm) + float(manual_from_nm))

    points = list(anchors)
    if not points:
        return Solution(
            ok=True,
            message="no points yet: showing the range set by hand",
            kind="manual",
            degree=1,
            coefficients=np.array([slope, centre], dtype=np.float64),
            x_ref=x_ref,
            x_scale=x_scale,
        )

    xs = np.array([point.x_px for point in points], dtype=np.float64)
    lambdas = np.array([point.wavelength_nm for point in points], dtype=np.float64)
    t = (xs - x_ref) / x_scale

    if len(points) == 1:
        coefficients = np.array([slope, float(lambdas[0]) - slope * float(t[0])])
        return Solution(
            ok=True,
            message="one point: the range by hand, slid onto it",
            kind="shifted",
            degree=1,
            coefficients=coefficients,
            x_ref=x_ref,
            x_scale=x_scale,
            points_used=1,
            residuals_nm=np.zeros(1),
        )

    # Two identical columns cannot both be right, and would make the fit blow up.
    if np.ptp(t) <= 0.0:
        return Solution(message="all the points sit on the same column")

    degree = int(min(len(points) - 1, max(1, min(max_degree, MAX_DEGREE))))
    coefficients = np.polyfit(t, lambdas, degree)
    residuals = np.polyval(coefficients, t) - lambdas
    return Solution(
        ok=True,
        message="ok",
        kind="fit",
        degree=degree,
        coefficients=np.asarray(coefficients, dtype=np.float64),
        x_ref=x_ref,
        x_scale=x_scale,
        points_used=len(points),
        residuals_nm=residuals,
        rms_nm=float(np.sqrt(np.mean(residuals ** 2))),
    )


def from_settings(
    coefficients: Sequence[float], x_ref: float, x_scale: float
) -> Optional[Solution]:
    """Rebuild a saved solution, so a restart does not lose the calibration."""
    array = np.asarray(list(coefficients), dtype=np.float64)
    if array.size < 2 or not np.all(np.isfinite(array)) or x_scale == 0.0:
        return None
    return Solution(
        ok=True,
        message="restored from settings.json",
        kind="fit",
        degree=int(array.size - 1),
        coefficients=array,
        x_ref=float(x_ref),
        x_scale=float(x_scale),
    )


def anchors_from_settings(saved: Sequence[dict]) -> List[Anchor]:
    points = []
    for entry in saved:
        if isinstance(entry, dict):
            points.append(Anchor.from_dict(entry))
    points.sort(key=lambda point: point.x_px)
    return points
