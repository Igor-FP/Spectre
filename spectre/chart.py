"""Drawing the spectrum as a finished picture: PNG for pasting, SVG for print.

Nothing here runs while the program is idle.  matplotlib takes about a second to
import and wants a backend of its own, so it is pulled in inside the one function
that needs it and forced onto Agg first - it must never try to open a window of
its own next to the live OpenGL one.

The colours of the strip under the axis come from `display.wavelength_rgba`, the
same function the strip on screen is built from, so the picture and the screen
cannot drift apart.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from . import display

#: The wavelength range the picture covers, fixed rather than following the data:
#: past this the camera barely responds and the curve there is not worth drawing.
CHART_FROM_NM = 300.0
CHART_TO_NM = 1050.0

#: Lines marked on every chart - the ones that matter for astronomy, not the
#: ones that happened to be clicked for the calibration.  Doublets are drawn at
#: their strong member; the pair is closer together than the chart can resolve.
MARKED_LINES = (
    (486.13, "H-beta"),
    (500.68, "O III"),
    (656.28, "H-alpha"),
    (671.64, "S II"),
    (760.50, "O2 A"),
)

#: 16 x 9 inches at 240 dpi is exactly 3840 x 2160.  Sizes in matplotlib are in
#: points, so fonts and line widths scale with the dpi on their own.
FIGSIZE = (16.0, 9.0)
PNG_DPI = 240

#: Samples across the colour strip.  It is a smooth ramp; more is pointless.
STRIP_SAMPLES = 2048

CURVE_COLOUR = "#1f4e79"
GRID_COLOUR = "#c8ccd4"
OUTSIDE_COLOUR = "#00000010"  # the dimming beyond the visible range
ANCHOR_COLOUR = "#b03030"


def export(
    path_stem: str,
    axis: np.ndarray,
    values: np.ndarray,
    unit: str,
    calibrated: bool,
    title: str = "",
    notes: Sequence[str] = (),
    relative: bool = False,
    formats: Sequence[str] = ("png", "svg"),
) -> list:
    """Write the curve as a picture. Returns the paths written.

    `axis` is wavelength in nm when `calibrated`, otherwise the frame column;
    without a wavelength there can be no colour strip and no visible-range
    marks, so the picture falls back to a plain graph.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)  # never a window of its own
    import matplotlib.pyplot as plt

    axis = np.asarray(axis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    # In nanometres the picture always covers the same range, so two charts can
    # be laid side by side; in frame columns there is nothing to fix it to.
    if calibrated:
        low, high = CHART_FROM_NM, CHART_TO_NM
    else:
        low, high = float(axis.min()), float(axis.max())

    figure = plt.figure(figsize=FIGSIZE)
    if calibrated:
        grid = figure.add_gridspec(2, 1, height_ratios=[22, 1], hspace=0.03)
        plot = figure.add_subplot(grid[0])
        strip = figure.add_subplot(grid[1], sharex=plot)
        plt.setp(plot.get_xticklabels(), visible=False)
    else:
        plot = figure.add_subplot(1, 1, 1)
        strip = None

    # Outside the visible range first, so the curve and the grid sit on top.
    if calibrated:
        if low < display.VISIBLE_FROM_NM:
            plot.axvspan(low, display.VISIBLE_FROM_NM, color=OUTSIDE_COLOUR, lw=0)
        if high > display.VISIBLE_TO_NM:
            plot.axvspan(display.VISIBLE_TO_NM, high, color=OUTSIDE_COLOUR, lw=0)
        for edge in (display.VISIBLE_FROM_NM, display.VISIBLE_TO_NM):
            if low < edge < high:
                plot.axvline(edge, color="#606060", lw=1.0, ls=(0, (6, 4)))

    plot.grid(True, which="major", color=GRID_COLOUR, lw=0.8)
    plot.grid(True, which="minor", color=GRID_COLOUR, lw=0.4, alpha=0.6)
    plot.minorticks_on()
    plot.plot(axis, values, color=CURVE_COLOUR, lw=1.2)
    plot.set_xlim(low, high)
    if relative:
        # Same rule as the graph on screen: above the comparison spectrum is
        # noise, and letting it set the top would squash what is being measured.
        plot.set_ylim(0.0, 100.0)
    plot.set_ylabel(unit)
    if title:
        plot.set_title(title)

    if calibrated:
        _mark_lines(plot, low, high)

    if strip is not None:
        wavelengths = np.linspace(low, high, STRIP_SAMPLES)
        strip.imshow(
            display.wavelength_image(wavelengths),
            extent=(low, high, 0.0, 1.0), aspect="auto", interpolation="bilinear",
        )
        strip.set_yticks([])
        strip.set_xlim(low, high)
        strip.set_xlabel("Wavelength, nm")
    else:
        plot.set_xlabel("Frame column, px")

    if notes:
        # Aligned with the left edge of the plot, not with the figure: saving
        # crops to the ink, so a note starting further left than the axis labels
        # would leave a band of white down the whole left side of the picture.
        figure.text(
            plot.get_position().x0, 0.005, "   ".join(notes),
            fontsize=7, color="#606060",
        )

    written = []
    for suffix in formats:
        path = f"{path_stem}.{suffix}"
        figure.savefig(
            path, dpi=PNG_DPI if suffix == "png" else None,
            facecolor="white", bbox_inches="tight", pad_inches=0.2,
        )
        written.append(path)
    plt.close(figure)
    return written


def _mark_lines(plot, low: float, high: float) -> None:
    """A dashed line and a label at every line of interest that is in range."""
    for position, name in MARKED_LINES:
        if not low <= position <= high:
            continue
        plot.axvline(position, color=ANCHOR_COLOUR, lw=0.9, ls=(0, (2, 3)), alpha=0.8)
        text = f"{name}\n{position:.1f} nm"
        plot.annotate(
            text,
            xy=(position, 1.0), xycoords=("data", "axes fraction"),
            xytext=(3, -4), textcoords="offset points",
            rotation=90, ha="left", va="top",
            fontsize=7, color=ANCHOR_COLOUR,
            # The curve often runs close to the top; without this the label
            # would be read through it.
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.75),
        )


def stem_for(path: str) -> str:
    """`captures/spectrum_...csv` -> the same without its suffix."""
    return os.path.splitext(path)[0]
