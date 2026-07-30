"""Turning linear 16-bit frames into something visible on screen.

The camera data stays untouched: the screen mapping is a lookup table
(black point / white point / midtone transfer function) applied to the raw
values, and the result is uploaded as an 8-bit single-channel GL texture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import OpenGL.GL as gl
from imgui_bundle import imgui

#: Frames are subsampled to roughly this many pixels for statistics and
#: histogram, which keeps them cheap at full frame rate.
STATS_SAMPLES = 120_000


def mtf(x: np.ndarray, m: float) -> np.ndarray:
    """Midtones transfer function: (1-m)*x / (m + x*(1-2m)).

    x in [0..1], m in (0..1). m = 0.5 is the identity (linear), smaller
    brightens. Same function the AstroBatch tools use for display stretch.
    """
    denom = m + x * (1.0 - 2.0 * m)
    return (1.0 - m) * x / np.maximum(denom, 1e-10)


@dataclass
class Stretch:
    """Screen mapping parameters, all in normalised 0..1 of full scale."""

    auto: bool = True
    black: float = 0.0
    white: float = 1.0
    midtone: float = 0.5  # 0.5 = linear
    lo_percentile: float = 0.2
    hi_percentile: float = 99.9

    _lut: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _lut_key: tuple = field(default=(), repr=False, compare=False)

    def lut(self, levels: int) -> np.ndarray:
        """uint8 lookup table with `levels` entries (65536 for RAW16)."""
        black, white = self.black, max(self.white, self.black + 1e-6)
        key = (levels, black, white, self.midtone)
        if self._lut_key != key:
            x = np.linspace(0.0, 1.0, levels, dtype=np.float64)
            t = np.clip((x - black) / (white - black), 0.0, 1.0)
            t = mtf(t, min(max(self.midtone, 1e-3), 0.999))
            self._lut = (t * 255.0 + 0.5).astype(np.uint8)
            self._lut_key = key
        return self._lut

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Map raw frame data to a uint8 image of the same shape."""
        levels = 256 if data.dtype == np.uint8 else 65536
        return self.lut(levels)[data]

    def autoscale(self, stats: "ImageStats") -> None:
        """Set black/white from the current frame's percentiles."""
        lo, hi = stats.lo, stats.hi
        if hi <= lo:
            hi = min(1.0, lo + 1.0 / 4096.0)
        self.black, self.white = float(lo), float(hi)


@dataclass
class ImageStats:
    """Cheap per-frame statistics, computed on a subsampled view."""

    min_adu: int = 0
    max_adu: int = 0
    mean_adu: float = 0.0
    saturated_fraction: float = 0.0
    lo: float = 0.0  # normalised low percentile
    hi: float = 1.0  # normalised high percentile
    hist: np.ndarray = field(default_factory=lambda: np.zeros(256, np.float32))
    full_scale: int = 65535
    sample_step: int = 1


def frame_stats(
    data: np.ndarray,
    full_scale: int,
    lo_percentile: float = 0.2,
    hi_percentile: float = 99.9,
) -> ImageStats:
    step = max(1, int(np.sqrt(data.size / STATS_SAMPLES)))
    sample = data[::step, ::step]
    flat = sample.ravel()
    lo_val, hi_val = np.percentile(flat, [lo_percentile, hi_percentile])
    hist, _ = np.histogram(flat, bins=256, range=(0, full_scale))
    saturation_level = full_scale * 0.995
    return ImageStats(
        min_adu=int(flat.min()),
        max_adu=int(flat.max()),
        mean_adu=float(flat.mean()),
        saturated_fraction=float(np.count_nonzero(sample >= saturation_level) / sample.size),
        lo=float(lo_val) / full_scale,
        hi=float(hi_val) / full_scale,
        hist=hist.astype(np.float32),
        full_scale=full_scale,
        sample_step=step,
    )


class ImageTexture:
    """A single-channel 8-bit GL texture, shown as grey through a swizzle."""

    def __init__(self):
        self._tex_id = 0
        self.width = 0
        self.height = 0

    def _create(self, width: int, height: int) -> None:
        self.release()
        self._tex_id = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        # One byte per pixel expanded to opaque grey.
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_SWIZZLE_R, gl.GL_RED)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_SWIZZLE_G, gl.GL_RED)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_SWIZZLE_B, gl.GL_RED)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_SWIZZLE_A, gl.GL_ONE)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_R8, width, height, 0, gl.GL_RED, gl.GL_UNSIGNED_BYTE, None
        )
        self.width, self.height = width, height

    def update(self, gray: np.ndarray) -> None:
        """Upload a (H, W) uint8 array."""
        if gray.dtype != np.uint8 or gray.ndim != 2:
            raise ValueError("expected a 2-D uint8 array")
        gray = np.ascontiguousarray(gray)
        height, width = gray.shape
        last = gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D)
        if self._tex_id == 0 or (width, height) != (self.width, self.height):
            self._create(width, height)
        else:
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex_id)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, 0)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D, 0, 0, 0, width, height, gl.GL_RED, gl.GL_UNSIGNED_BYTE, gray
        )
        # Mipmaps: without them, thin spectral lines vanish in a zoomed-out view.
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, last)

    @property
    def valid(self) -> bool:
        return self._tex_id != 0

    @property
    def ref(self) -> "imgui.ImTextureRef":
        return imgui.ImTextureRef(self._tex_id)

    def release(self) -> None:
        if self._tex_id:
            gl.glDeleteTextures([self._tex_id])
            self._tex_id = 0
            self.width = self.height = 0
