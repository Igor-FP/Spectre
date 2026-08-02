"""Dark frames: making a master dark and keeping it ready to subtract.

A master dark is the honest median of `DARK_FRAME_COUNT` frames taken with the
lens covered.  The median rather than the mean because the point is to throw
away cosmic rays, hot pixels that fire only sometimes and the odd glitched
readout, not to win signal to noise.

A dark is only valid for the gain and exposure it was taken at, so that is what
names the file: `dark_<gain>_<exposure key>.fit`, where the exposure key is the
position on the exposure scale in whole per cent (`camera.exposure_key`).  Whole
frames only - the crop can move, the dark cannot follow it.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Callable, List, Optional

import numpy as np

from . import camera as cam

#: Frames combined into one master dark.  Odd on purpose: the median of an even
#: count averages the two middle samples, which would break the 16 ADU step of a
#: 12-bit sensor read as RAW16.
DARK_FRAME_COUNT = 7

DARK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "darks"
)

#: How long to wait for one frame beyond the exposure itself before giving up.
FRAME_TIMEOUT_S = 30.0

_NAME = re.compile(r"^dark_(\d+)_(\d+)\.fit$", re.IGNORECASE)


def dark_key(gain: int, exposure_us: int):
    """What a dark is filed under: the gain, and the exposure as whole per cent."""
    return int(gain), cam.exposure_key(exposure_us)


def dark_name(gain: int, exposure_us: int) -> str:
    return "dark_{0}_{1}.fit".format(*dark_key(gain, exposure_us))


def dark_path(gain: int, exposure_us: int, directory: str = DARK_DIR) -> str:
    return os.path.join(directory, dark_name(gain, exposure_us))


def find_dark(gain: int, exposure_us: int, directory: str = DARK_DIR) -> Optional[str]:
    """The master dark for this gain and exposure, if one has been made."""
    path = dark_path(gain, exposure_us, directory)
    return path if os.path.isfile(path) else None


def list_darks(directory: str = DARK_DIR) -> List[tuple]:
    """[(gain, exposure key, path)] for every dark on disk."""
    found = []
    try:
        names = os.listdir(directory)
    except OSError:
        return found
    for name in sorted(names):
        match = _NAME.match(name)
        if match:
            found.append((int(match.group(1)), int(match.group(2)),
                          os.path.join(directory, name)))
    return found


def combine(frames: List[np.ndarray]) -> np.ndarray:
    """Honest median of the stack, in the frames' own dtype."""
    stack = np.stack(frames)
    middle = stack.shape[0] // 2
    if stack.shape[0] % 2:
        return np.partition(stack, middle, axis=0)[middle]
    pair = np.partition(stack, [middle - 1, middle], axis=0)
    total = pair[middle - 1].astype(np.uint32) + pair[middle].astype(np.uint32)
    return (total // 2).astype(stack.dtype)


def save_dark(path: str, data: np.ndarray, gain: int, exposure_us: int, frames: int,
              camera_name: str = "", temperature_c: Optional[float] = None) -> None:
    from astropy.io import fits  # imported lazily: only needed when saving

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    hdu = fits.PrimaryHDU(np.ascontiguousarray(data))
    header = hdu.header
    header["IMAGETYP"] = ("DARK", "frame type")
    header["EXPTIME"] = (exposure_us / 1e6, "[s] exposure time")
    header["EXPUS"] = (int(exposure_us), "[us] exposure time as set on the camera")
    header["GAIN"] = (int(gain), "camera gain units")
    header["EXPKEY"] = (cam.exposure_key(exposure_us), "exposure scale position, per cent")
    header["NCOMBINE"] = (int(frames), "frames combined")
    header["STACKING"] = ("median", "how they were combined")
    if camera_name:
        header["INSTRUME"] = (camera_name[:68], "camera")
    if temperature_c is not None:
        header["CCD-TEMP"] = (temperature_c, "[C] sensor temperature")
    header["SWCREATE"] = ("Spectre", "software")
    hdu.writeto(path, overwrite=True)


def subtract(data: np.ndarray, dark) -> np.ndarray:
    """Frame minus dark, floored at zero, in the frame's own dtype.

    `dark` is a master dark of the same shape, or a plain number when there is
    no dark and a flat bias level is all there is to take off.
    """
    difference = data.astype(np.int32) - np.asarray(dark, dtype=np.int32)
    np.maximum(difference, 0, out=difference)
    return difference.astype(data.dtype)


# ---------------------------------------------------------------------------
# Collecting the frames
# ---------------------------------------------------------------------------


class DarkMaker:
    """Pulls N distinct frames off the running camera and medians them.

    Runs on its own thread so a ten-minute exposure does not freeze the window.
    Every frame has to come from the same control settings; if the exposure or
    the gain moves while it is collecting, the run is abandoned rather than
    quietly producing a dark that belongs to nothing.
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._collected = 0
        self._wanted = 0
        self._cancel = False
        self._result = None  # (path, gain, exposure_us) or ("", "", message)
        self._error = ""

    def start(self, camera, count: int = DARK_FRAME_COUNT,
              directory: str = DARK_DIR) -> bool:
        if self.running or camera is None:
            return False
        with self._lock:
            self._collected = 0
            self._wanted = int(count)
            self._cancel = False
            self._result = None
            self._error = ""
        self._thread = threading.Thread(
            target=self._run, args=(camera, int(count), directory),
            name="dark-maker", daemon=True,
        )
        self._thread.start()
        return True

    def _run(self, camera, count: int, directory: str) -> None:
        try:
            self._collect_and_save(camera, count, directory)
        except Exception as exc:  # never take the UI down with us
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"

    def _collect_and_save(self, camera, count: int, directory: str) -> None:
        frames: List[np.ndarray] = []
        gain = exposure_us = settings_gen = None
        last_index = -1
        patience = FRAME_TIMEOUT_S + camera.control_value(cam.EXPOSURE, 0) / 1e6
        deadline = time.monotonic() + patience

        while len(frames) < count:
            if self._cancelled():
                with self._lock:
                    self._error = "cancelled"
                return
            frame = camera.latest_frame()
            if frame is None or frame.index == last_index:
                if time.monotonic() > deadline:
                    with self._lock:
                        self._error = "no frames from the camera"
                    return
                time.sleep(0.005)
                continue

            last_index = frame.index
            if gain is None:
                gain, exposure_us = frame.gain, frame.exposure_us
                settings_gen = frame.settings_gen
            elif frame.settings_gen != settings_gen:
                with self._lock:
                    self._error = "exposure or gain changed during the run"
                return
            # The grabber reuses its buffer, so this has to be our own copy.
            frames.append(np.array(frame.data, copy=True))
            deadline = time.monotonic() + patience
            with self._lock:
                self._collected = len(frames)

        master = combine(frames)
        path = dark_path(gain, exposure_us, directory)
        stats = camera.stats()
        save_dark(
            path, master, gain, exposure_us, len(frames),
            camera_name=getattr(camera, "name", ""),
            temperature_c=stats.temperature_c if stats else None,
        )
        with self._lock:
            self._result = (path, gain, exposure_us)

    # -- state -------------------------------------------------------------

    def cancel(self) -> None:
        with self._lock:
            self._cancel = True

    def _cancelled(self) -> bool:
        with self._lock:
            return self._cancel

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def progress(self):
        """(collected, wanted) so far."""
        with self._lock:
            return self._collected, self._wanted

    def take_result(self):
        """(path, gain, exposure_us) once, or None; check `error` after it."""
        with self._lock:
            result, self._result = self._result, None
        if result is not None:
            self._thread = None
        return result

    def take_error(self) -> str:
        with self._lock:
            error, self._error = self._error, ""
        if error:
            self._thread = None
        return error
