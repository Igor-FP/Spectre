"""Saving and loading single frames.

Frames are written exactly as the camera delivered them - linear, 16-bit, no
stretch - so a saved file can be fed back into the app and into the calibration
and give the same numbers as the live frame did.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np

CAPTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "captures"
)
FITS_SUFFIXES = (".fits", ".fit", ".fts")
LOADABLE_SUFFIXES = FITS_SUFFIXES + (".npy",)


@dataclass
class FrameMeta:
    """What we know about a frame, for the FITS header and for replay."""

    camera: str = ""
    exposure_us: int = 0
    gain: int = 0
    offset: Optional[int] = None
    binning: int = 1
    bit_depth: int = 16
    pixel_size_um: float = 0.0
    temperature_c: Optional[float] = None
    crop: Optional[Tuple[int, int, int, int]] = None
    band: Optional[Dict[str, float]] = None
    timestamp: Optional[datetime] = None


def timestamped_name(prefix: str = "spectre", when: Optional[datetime] = None) -> str:
    when = when or datetime.now()
    return f"{prefix}_{when.strftime('%Y%m%d_%H%M%S')}"


def save_frame(
    data: np.ndarray,
    meta: FrameMeta,
    directory: str = CAPTURE_DIR,
    basename: Optional[str] = None,
    write_fits: bool = True,
    write_npy: bool = True,
) -> list:
    """Write the frame; returns the list of paths written."""
    if data is None or data.ndim != 2:
        raise ValueError("expected a 2-D frame")
    when = meta.timestamp or datetime.now()
    os.makedirs(directory, exist_ok=True)
    stem = os.path.join(directory, basename or timestamped_name(when=when))

    written = []
    if write_npy:
        path = stem + ".npy"
        np.save(path, data)
        written.append(path)
    if write_fits:
        path = stem + ".fits"
        _write_fits(path, data, meta, when)
        written.append(path)
    if not written:
        raise ValueError("no output format selected")
    return written


def _write_fits(path: str, data: np.ndarray, meta: FrameMeta, when: datetime) -> None:
    from astropy.io import fits  # imported lazily: only needed when saving

    hdu = fits.PrimaryHDU(np.ascontiguousarray(data))
    header = hdu.header
    header["IMAGETYP"] = ("LIGHT", "frame type")
    header["DATE-OBS"] = (when.isoformat(timespec="milliseconds"), "local time of capture")
    header["EXPTIME"] = (meta.exposure_us / 1e6, "[s] exposure time")
    header["EXPOSURE"] = (meta.exposure_us / 1e6, "[s] exposure time")
    header["EXPUS"] = (int(meta.exposure_us), "[us] exposure time as set on the camera")
    header["GAIN"] = (int(meta.gain), "camera gain units")
    if meta.offset is not None:
        header["OFFSET"] = (int(meta.offset), "camera offset units")
    header["XBINNING"] = (int(meta.binning), "binning along X")
    header["YBINNING"] = (int(meta.binning), "binning along Y")
    header["BITDEPTH"] = (int(meta.bit_depth), "sensor ADC bits")
    if meta.camera:
        header["INSTRUME"] = (meta.camera[:68], "camera")
    if meta.pixel_size_um:
        header["XPIXSZ"] = (meta.pixel_size_um, "[um] pixel size")
        header["YPIXSZ"] = (meta.pixel_size_um, "[um] pixel size")
    if meta.temperature_c is not None:
        header["CCD-TEMP"] = (meta.temperature_c, "[C] sensor temperature")
    if meta.crop is not None:
        x0, y0, x1, y1 = meta.crop
        header["CROPX0"] = (int(x0), "crop left, full-frame pixels")
        header["CROPY0"] = (int(y0), "crop top, full-frame pixels")
        header["CROPX1"] = (int(x1), "crop right, exclusive")
        header["CROPY1"] = (int(y1), "crop bottom, exclusive")
    if meta.band:
        for key, comment in (
            ("BANDANG", "[deg] band angle against X"),
            ("BANDCY", "band centre y at BANDREFX"),
            ("BANDW", "[px] band width (FWHM of the projection)"),
            ("BANDLO", "band upper edge y"),
            ("BANDHI", "band lower edge y"),
            ("BANDREFX", "x the band geometry refers to"),
        ):
            if key in meta.band:
                header[key] = (float(meta.band[key]), comment)
    header["SWCREATE"] = ("Spectre", "software")
    hdu.writeto(path, overwrite=True)


def save_spectrum_csv(
    columns,
    values,
    wavelengths=None,
    unit: str = "%",
    notes=(),
    directory: str = CAPTURE_DIR,
) -> str:
    """Write the 1-D curve as CSV: wavelength (or column) against per cent.

    Two data columns, the same pair the graph draws.  Without a wavelength
    calibration the first one is the frame column instead, named as such rather
    than left blank.
    """
    import numpy as np

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, timestamped_name("spectrum") + ".csv")
    calibrated = wavelengths is not None
    axis = np.asarray(wavelengths if calibrated else columns, dtype=float)
    values = np.asarray(values, dtype=float)
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("# Spectre 1-D spectrum\n")
        for note in notes:
            handle.write(f"# {note}\n")
        if not calibrated:
            handle.write("# no wavelength calibration: the axis is the frame column\n")
        handle.write(
            "wavelength_nm,percent\n" if calibrated else "x_px,percent\n"
        )
        for position, value in zip(axis, values):
            handle.write(f"{position:.4f},{value:.4f}\n")
    return path


def list_captures(directory: str = CAPTURE_DIR) -> list:
    """Saved frames, newest first."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    paths = [
        os.path.join(directory, name)
        for name in names
        if name.lower().endswith(LOADABLE_SUFFIXES)
    ]
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths


def load_frame(path: str) -> Tuple[np.ndarray, FrameMeta]:
    """Load a saved frame, with whatever metadata the file carries."""
    lower = path.lower()
    meta = FrameMeta(camera=os.path.basename(path))
    if lower.endswith(".npy"):
        data = np.load(path)
    elif lower.endswith(FITS_SUFFIXES):
        from astropy.io import fits

        with fits.open(path, memmap=False) as hdul:
            hdu = next((h for h in hdul if getattr(h, "data", None) is not None), None)
            if hdu is None:
                raise ValueError(f"no image data in {path}")
            data = np.array(hdu.data)
            meta = _meta_from_header(hdu.header, meta)
    else:
        raise ValueError(f"unsupported file type: {path}")

    data = np.asarray(data)
    while data.ndim > 2:  # take the first plane of a cube
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {data.shape}")
    if data.dtype == np.uint8:
        pass
    elif data.dtype != np.uint16:
        data = np.clip(data, 0, 65535).astype(np.uint16)
    return np.ascontiguousarray(data), meta


def _meta_from_header(header, meta: FrameMeta) -> FrameMeta:
    def number(key, default=None):
        value = header.get(key)
        return value if isinstance(value, (int, float)) else default

    exposure_s = number("EXPTIME", number("EXPOSURE"))
    exposure_us = number("EXPUS")
    if exposure_us is None and exposure_s is not None:
        exposure_us = exposure_s * 1e6
    meta.exposure_us = int(exposure_us or 0)
    meta.gain = int(number("GAIN", 0) or 0)
    offset = number("OFFSET")
    meta.offset = int(offset) if offset is not None else None
    meta.binning = int(number("XBINNING", 1) or 1)
    meta.bit_depth = int(number("BITDEPTH", 16) or 16)
    meta.pixel_size_um = float(number("XPIXSZ", 0.0) or 0.0)
    temperature = number("CCD-TEMP")
    meta.temperature_c = float(temperature) if temperature is not None else None
    instrument = header.get("INSTRUME")
    if isinstance(instrument, str) and instrument.strip():
        meta.camera = instrument.strip()
    crop = [number(key) for key in ("CROPX0", "CROPY0", "CROPX1", "CROPY1")]
    if all(value is not None for value in crop):
        meta.crop = tuple(int(value) for value in crop)
    return meta
