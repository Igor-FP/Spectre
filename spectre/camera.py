"""Camera acquisition: a background grabber thread with a latest-frame slot.

The UI thread never touches the SDK directly.  It queues control changes with
`set_control()` and picks up images with `latest_frame()`; everything else
happens on the acquisition thread.

Two implementations share `BaseCamera`:
  * `AsiCamera`       - a real ZWO ASI camera through `asi_sdk`
  * `SimulatedCamera` - a synthetic spectrum, for working without hardware
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from . import asi_sdk
from .asi_sdk import ControlType, ExposureStatus, ImgType

# Control names used across the app; mapped to SDK control types by AsiCamera.
EXPOSURE = "exposure"  # microseconds
GAIN = "gain"  # camera units (ASI290MM: 0..600, 0.1 dB each)
OFFSET = "offset"
BANDWIDTH = "bandwidth"
HIGH_SPEED = "high_speed"

_SDK_CONTROL = {
    EXPOSURE: ControlType.EXPOSURE,
    GAIN: ControlType.GAIN,
    OFFSET: ControlType.OFFSET,
    BANDWIDTH: ControlType.BANDWIDTHOVERLOAD,
    HIGH_SPEED: ControlType.HIGH_SPEED_MODE,
}

#: Above this exposure the grabber switches from video to single-shot (snap)
#: mode, which is what ZWO recommends for long exposures and lets us abort a
#: running exposure when the user moves a slider.
SNAP_THRESHOLD_US = 1_000_000


# ---------------------------------------------------------------------------
# The exposure scale
# ---------------------------------------------------------------------------
#
# 1 ms at one end, 1 s exactly in the middle, 10 minutes at the other. Each half
# is logarithmic on its own, because a single logarithmic scale always puts
# sqrt(min*max) in the middle - 775 ms here - and no choice of base changes
# that, the base cancels. The two halves span 3.00 and 2.78 decades, so the kink
# in the middle is not noticeable.
#
# This lives here rather than in the UI because it is not only a slider: a dark
# frame is filed under its position on this scale, and that name must not depend
# on a widget.

EXPOSURE_SCALE_MIN_MS = 1.0
EXPOSURE_SCALE_MID_MS = 1000.0
EXPOSURE_SCALE_MAX_MS = 600_000.0


def exposure_position(exposure_ms: float) -> float:
    """Exposure in milliseconds -> position on the scale, 0..1."""
    exposure_ms = min(max(float(exposure_ms), EXPOSURE_SCALE_MIN_MS), EXPOSURE_SCALE_MAX_MS)
    if exposure_ms <= EXPOSURE_SCALE_MID_MS:
        span = math.log(EXPOSURE_SCALE_MID_MS / EXPOSURE_SCALE_MIN_MS)
        return 0.5 * math.log(exposure_ms / EXPOSURE_SCALE_MIN_MS) / span
    span = math.log(EXPOSURE_SCALE_MAX_MS / EXPOSURE_SCALE_MID_MS)
    return 0.5 + 0.5 * math.log(exposure_ms / EXPOSURE_SCALE_MID_MS) / span


def exposure_from_position(position: float) -> float:
    """Position on the scale, 0..1 -> exposure in milliseconds."""
    position = min(max(float(position), 0.0), 1.0)
    if position <= 0.5:
        ratio = EXPOSURE_SCALE_MID_MS / EXPOSURE_SCALE_MIN_MS
        return EXPOSURE_SCALE_MIN_MS * ratio ** (position / 0.5)
    ratio = EXPOSURE_SCALE_MAX_MS / EXPOSURE_SCALE_MID_MS
    return EXPOSURE_SCALE_MID_MS * ratio ** ((position - 0.5) / 0.5)


def exposure_key(exposure_us: int) -> int:
    """Position on the scale as a whole per cent, 0..100.

    What a dark frame is filed under, together with the gain.  Rounding to a
    per cent puts exposures within about 1.4 % of each other in the same bucket,
    which is far finer than the dark current cares about.
    """
    return int(round(100.0 * exposure_position(float(exposure_us) / 1000.0)))


@dataclass(frozen=True)
class ControlRange:
    name: str
    min_value: int
    max_value: int
    default_value: int
    writable: bool = True
    auto_supported: bool = False


@dataclass
class Frame:
    """One acquired image plus the settings that produced it."""

    data: np.ndarray  # 2-D, uint16 (RAW16) or uint8 (RAW8)
    index: int
    timestamp: float  # time.monotonic() at arrival
    exposure_us: int
    gain: int
    settings_gen: int  # bumped on every applied control change
    full_scale: int  # highest value the dtype can hold (65535 / 255)
    mode: str  # "video" or "snap"

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]


@dataclass
class Stats:
    frames: int = 0
    fps: float = 0.0
    timeouts: int = 0
    errors: int = 0
    dropped: int = 0
    temperature_c: Optional[float] = None
    mode: str = "-"
    exposing_since: Optional[float] = None  # monotonic, for the progress bar
    last_error: str = ""


class BaseCamera:
    """Thread + latest-frame slot + queued control changes."""

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        bit_depth: int,
        img_type: ImgType,
        controls: Dict[str, ControlRange],
        values: Dict[str, int],
        pixel_size_um: float = 0.0,
        is_simulated: bool = False,
    ):
        self.name = name
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.img_type = img_type
        self.controls = controls
        self.pixel_size_um = pixel_size_um
        self.is_simulated = is_simulated

        self._values = dict(values)
        self._pending: Dict[str, int] = {}
        self._lock = threading.Lock()  # guards _values / _pending / _frame / _stats
        self._settings_gen = 0
        self._frame: Optional[Frame] = None
        self._frame_index = 0
        self._stats = Stats()
        self._times: deque = deque(maxlen=30)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._paused = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="grabber", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout)

    def close(self) -> None:
        self.stop()
        self._teardown()

    def set_paused(self, paused: bool) -> None:
        """Pause acquisition without releasing the camera."""
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    # -- controls ----------------------------------------------------------

    def set_control(self, name: str, value: int) -> None:
        """Queue a control change; applied by the grabber before the next frame."""
        rng = self.controls.get(name)
        if rng is None:
            return
        value = int(max(rng.min_value, min(rng.max_value, value)))
        with self._lock:
            if self._values.get(name) == value and name not in self._pending:
                return
            self._pending[name] = value

    def control_value(self, name: str, default: int = 0) -> int:
        with self._lock:
            if name in self._pending:
                return self._pending[name]
            return self._values.get(name, default)

    @property
    def exposure_us(self) -> int:
        return self.control_value(EXPOSURE, 10_000)

    @property
    def gain(self) -> int:
        return self.control_value(GAIN, 0)

    def _drain_pending(self) -> Dict[str, int]:
        with self._lock:
            pending, self._pending = self._pending, {}
            return pending

    def _has_pending(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def _apply_pending(self) -> None:
        pending = self._drain_pending()
        if not pending:
            return
        for name, value in pending.items():
            try:
                self._apply_control(name, value)
            except Exception as exc:  # keep grabbing; surface in the UI
                self._note_error(f"set {name}={value}: {exc}")
                continue
            with self._lock:
                self._values[name] = value
        with self._lock:
            self._settings_gen += 1

    # -- frames ------------------------------------------------------------

    def latest_frame(self) -> Optional[Frame]:
        """Take the newest frame, or None if nothing new arrived."""
        with self._lock:
            frame, self._frame = self._frame, None
            return frame

    def _publish(self, data: np.ndarray, mode: str) -> None:
        now = time.monotonic()
        self._times.append(now)
        with self._lock:
            self._frame_index += 1
            self._frame = Frame(
                data=data,
                index=self._frame_index,
                timestamp=now,
                exposure_us=self._values.get(EXPOSURE, 0),
                gain=self._values.get(GAIN, 0),
                settings_gen=self._settings_gen,
                full_scale=255 if data.dtype == np.uint8 else 65535,
                mode=mode,
            )
            self._stats.frames = self._frame_index
            self._stats.mode = mode
            if len(self._times) >= 2:
                span = self._times[-1] - self._times[0]
                self._stats.fps = (len(self._times) - 1) / span if span > 0 else 0.0

    def stats(self) -> Stats:
        with self._lock:
            s = self._stats
            return Stats(
                frames=s.frames,
                fps=s.fps,
                timeouts=s.timeouts,
                errors=s.errors,
                dropped=s.dropped,
                temperature_c=s.temperature_c,
                mode=s.mode,
                exposing_since=s.exposing_since,
                last_error=s.last_error,
            )

    def _note_error(self, message: str) -> None:
        with self._lock:
            self._stats.errors += 1
            self._stats.last_error = message

    def _note_timeout(self) -> None:
        with self._lock:
            self._stats.timeouts += 1

    # -- thread body -------------------------------------------------------

    def _run(self) -> None:
        was_paused = False
        try:
            while not self._stop.is_set():
                self._apply_pending()
                paused = self._paused.is_set()
                if paused != was_paused:
                    was_paused = paused
                    try:
                        self._on_pause(paused)
                    except Exception as exc:
                        self._note_error(str(exc))
                if paused:
                    time.sleep(0.05)
                    continue
                try:
                    self._grab()
                except Exception as exc:
                    self._note_error(str(exc))
                    time.sleep(0.25)
        finally:
            with self._lock:
                self._stats.exposing_since = None
            try:
                self._idle()
            except Exception:
                pass

    # -- to implement ------------------------------------------------------

    def _grab(self) -> None:
        """Acquire one frame and call `_publish`; may return without a frame."""
        raise NotImplementedError

    def _apply_control(self, name: str, value: int) -> None:
        raise NotImplementedError

    def _idle(self) -> None:
        """Called on the grabber thread when it exits (stop capture, etc.)."""

    def _on_pause(self, paused: bool) -> None:
        """Called on the grabber thread when pause state changes."""

    def _teardown(self) -> None:
        """Release the device (called after the thread has joined)."""


# ---------------------------------------------------------------------------
# Real hardware
# ---------------------------------------------------------------------------


def list_cameras() -> list:
    """Enumerate connected ASI cameras (loads the SDK on first call)."""
    asi_sdk.load()
    return asi_sdk.list_cameras()


class AsiCamera(BaseCamera):
    """A ZWO ASI camera in video or snap mode, full frame, RAW16 by default."""

    def __init__(
        self,
        info: asi_sdk.CameraInfo,
        img_type: ImgType = ImgType.RAW16,
        binning: int = 1,
        snap_threshold_us: int = SNAP_THRESHOLD_US,
    ):
        self.info = info
        self.camera_id = info.camera_id
        self.snap_threshold_us = snap_threshold_us
        self._capture_mode: Optional[str] = None  # None | "video" | "snap"
        self._buffer_shape = None
        self._last_temp_read = 0.0

        asi_sdk.open_camera(self.camera_id)
        try:
            asi_sdk.init_camera(self.camera_id)
            caps = asi_sdk.get_control_caps(self.camera_id)
            controls, values = {}, {}
            by_type = {c.control_type: c for c in caps}
            for name, ctrl in _SDK_CONTROL.items():
                cap = by_type.get(int(ctrl))
                if cap is None:
                    continue
                controls[name] = ControlRange(
                    name=name,
                    min_value=cap.min_value,
                    max_value=cap.max_value,
                    default_value=cap.default_value,
                    writable=cap.is_writable,
                    auto_supported=cap.is_auto_supported,
                )
                values[name] = asi_sdk.get_control_value(self.camera_id, ctrl)[0]

            self._has_temperature = int(ControlType.TEMPERATURE) in by_type
            width, height = info.max_width // binning, info.max_height // binning
            # ROI width must be a multiple of 8 and height of 2 (ZWO requirement).
            width -= width % 8
            height -= height % 2
            asi_sdk.set_roi_format(self.camera_id, width, height, binning, int(img_type))
            asi_sdk.set_start_pos(self.camera_id, 0, 0)
            try:
                asi_sdk.disable_dark_subtract(self.camera_id)
            except asi_sdk.ASIError:
                pass  # not supported on every model

            super().__init__(
                name=info.name,
                width=width,
                height=height,
                bit_depth=info.bit_depth,
                img_type=img_type,
                controls=controls,
                values=values,
                pixel_size_um=info.pixel_size_um,
            )
            self.binning = binning
            self.serial = asi_sdk.get_serial_number(self.camera_id)
        except Exception:
            try:
                asi_sdk.close_camera(self.camera_id)
            except Exception:
                pass
            raise

    # -- BaseCamera hooks --------------------------------------------------

    def _apply_control(self, name: str, value: int) -> None:
        asi_sdk.set_control_value(self.camera_id, _SDK_CONTROL[name], value)

    def _grab(self) -> None:
        self._refresh_temperature()
        want = "snap" if self.exposure_us >= self.snap_threshold_us else "video"
        if want != self._capture_mode:
            self._set_capture_mode(want)
        if want == "video":
            self._grab_video()
        else:
            self._grab_snap()

    def _idle(self) -> None:
        self._set_capture_mode(None)

    def _on_pause(self, paused: bool) -> None:
        # Stop the stream while paused, otherwise resuming would hand us a
        # stale frame out of the SDK's queue.
        if paused:
            self._set_capture_mode(None)

    def _teardown(self) -> None:
        try:
            asi_sdk.close_camera(self.camera_id)
        except Exception:
            pass

    # -- internals ---------------------------------------------------------

    def _new_buffer(self) -> np.ndarray:
        dtype = np.uint16 if self.img_type == ImgType.RAW16 else np.uint8
        return np.empty((self.height, self.width), dtype=dtype)

    def _set_capture_mode(self, mode: Optional[str]) -> None:
        if self._capture_mode == "video":
            asi_sdk.stop_video_capture(self.camera_id)
        elif self._capture_mode == "snap":
            try:
                asi_sdk.stop_exposure(self.camera_id)
            except asi_sdk.ASIError:
                pass
        self._capture_mode = mode
        if mode == "video":
            asi_sdk.start_video_capture(self.camera_id)
        with self._lock:
            self._stats.mode = mode or "-"

    def _grab_video(self) -> None:
        buf = self._new_buffer()
        exposure_ms = self.exposure_us / 1000.0
        timeout_ms = int(exposure_ms * 2 + 500)
        if not asi_sdk.get_video_data(self.camera_id, buf, timeout_ms):
            self._note_timeout()
            return
        self._publish(buf, "video")
        with self._lock:
            try:
                self._stats.dropped = asi_sdk.get_dropped_frames(self.camera_id)
            except asi_sdk.ASIError:
                pass

    def _grab_snap(self) -> None:
        exposure_s = self.exposure_us / 1e6
        asi_sdk.start_exposure(self.camera_id, False)
        start = time.monotonic()
        with self._lock:
            self._stats.exposing_since = start
        deadline = start + exposure_s * 2 + 10.0
        try:
            while True:
                status = asi_sdk.get_exp_status(self.camera_id)
                if status == ExposureStatus.SUCCESS:
                    break
                if status == ExposureStatus.FAILED:
                    self._note_error("exposure failed")
                    return
                # Abort so a slider change or a stop request takes effect at once.
                if self._stop.is_set() or self._has_pending() or self._paused.is_set():
                    asi_sdk.stop_exposure(self.camera_id)
                    return
                if time.monotonic() > deadline:
                    asi_sdk.stop_exposure(self.camera_id)
                    self._note_error("exposure timed out")
                    return
                time.sleep(0.02 if exposure_s < 2 else 0.05)
        finally:
            with self._lock:
                self._stats.exposing_since = None

        buf = self._new_buffer()
        asi_sdk.get_data_after_exp(self.camera_id, buf)
        self._publish(buf, "snap")

    def _refresh_temperature(self) -> None:
        if not self._has_temperature:
            return
        now = time.monotonic()
        if now - self._last_temp_read < 2.0:
            return
        self._last_temp_read = now
        try:
            raw, _ = asi_sdk.get_control_value(self.camera_id, ControlType.TEMPERATURE)
        except asi_sdk.ASIError:
            return
        with self._lock:
            self._stats.temperature_c = raw / 10.0


# ---------------------------------------------------------------------------
# Replay of a saved frame
# ---------------------------------------------------------------------------


class FileCamera(BaseCamera):
    """Serves a frame saved earlier, so the UI and the calibration can be run
    on real data with no camera attached.

    It has no controls: exposure and gain come from the file and are shown as
    they were when the frame was taken.
    """

    def __init__(self, path: str, interval_s: float = 0.3):
        from . import frameio

        data, meta = frameio.load_frame(path)
        self.path = path
        self.meta = meta
        self._data = data
        self._interval = interval_s
        super().__init__(
            name=os.path.basename(path),
            width=data.shape[1],
            height=data.shape[0],
            bit_depth=meta.bit_depth,
            img_type=ImgType.RAW8 if data.dtype == np.uint8 else ImgType.RAW16,
            controls={},  # nothing to change in a file
            values={EXPOSURE: meta.exposure_us, GAIN: meta.gain},
            pixel_size_um=meta.pixel_size_um,
        )
        with self._lock:
            self._stats.temperature_c = meta.temperature_c

    def _apply_control(self, name: str, value: int) -> None:
        pass

    def _grab(self) -> None:
        self._publish(self._data.copy(), "file")
        time.sleep(self._interval)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


#: Geometry baked into the simulated spectrum - the ground truth the geometry
#: calibration is expected to recover.
SIM_BAND_ANGLE_DEG = 0.60  # band tilt against the X axis (positive: down to the right)
SIM_SHEAR_DEG = 3.00  # tilt of the spectral lines against the band normal
SIM_BAND_WIDTH_PX = 90.0  # FWHM of the slit image across the band
SIM_EDGE_SOFTNESS_PX = 1.5  # how far the band edges are smeared by the optics


class SimulatedCamera(BaseCamera):
    """Synthetic 16-bit spectrum: a tilted band with emission lines.

    Reacts to exposure and gain the way a real sensor does (linear signal, shot
    noise, saturation), so the UI, the geometry calibration and the future
    auto-exposure can be exercised without hardware.
    """

    def __init__(
        self,
        width: int = 1936,
        height: int = 1096,
        snap_threshold_us: int = SNAP_THRESHOLD_US,
        band_angle_deg: float = SIM_BAND_ANGLE_DEG,
        shear_deg: float = SIM_SHEAR_DEG,
        band_width_px: float = SIM_BAND_WIDTH_PX,
        edge_softness_px: float = SIM_EDGE_SOFTNESS_PX,
    ):
        controls = {
            EXPOSURE: ControlRange(EXPOSURE, 32, 2_000_000_000, 10_000),
            GAIN: ControlRange(GAIN, 0, 600, 200),
            OFFSET: ControlRange(OFFSET, 0, 240, 10),
        }
        super().__init__(
            name="Simulated spectrum",
            width=width,
            height=height,
            bit_depth=12,
            img_type=ImgType.RAW16,
            controls=controls,
            values={EXPOSURE: 20_000, GAIN: 200, OFFSET: 10},
            pixel_size_um=2.9,
            is_simulated=True,
        )
        self.snap_threshold_us = snap_threshold_us
        self.band_angle_deg = band_angle_deg
        self.shear_deg = shear_deg
        self.band_width_px = band_width_px
        self.edge_softness_px = edge_softness_px
        self._rng = np.random.default_rng(1234)
        self._pattern = self._build_pattern()

    def _build_pattern(self) -> np.ndarray:
        """Noiseless electrons-per-second image.

        The band is tilted by `band_angle_deg` against X, has a flat-topped
        (super-Gaussian) slit profile across it, and its spectral lines are
        tilted by `shear_deg` against the band normal.
        """
        width, height = self.width, self.height
        x = np.arange(width, dtype=np.float32)
        y = np.arange(height, dtype=np.float32)
        tan_band = np.tan(np.radians(self.band_angle_deg))
        tan_shear = np.tan(np.radians(self.shear_deg))

        # Signed distance from the band centre line, per pixel.
        band_centre = height * 0.5 + (x - width * 0.5) * tan_band
        across_distance = y[:, None] - band_centre[None, :]
        # Slit image: a rectangle with edges softened by the optics.  The half
        # level sits exactly at +-band_width/2, so the true FWHM is band_width.
        half = 0.5 * self.band_width_px
        edge = self.edge_softness_px
        across = 0.5 * (
            np.tanh((half + across_distance) / edge) + np.tanh((half - across_distance) / edge)
        )

        # Position along the spectrum; sheared lines lean with the distance
        # across the band.
        u = x[None, :] + across_distance * tan_shear
        spectral = 900.0 + 500.0 * np.exp(-0.5 * ((u - width * 0.45) / (width * 0.35)) ** 2)
        for centre, strength, sigma in (
            (0.14, 9000, 1.8),
            (0.21, 3500, 1.6),
            (0.33, 16000, 2.0),
            (0.38, 2200, 1.5),
            (0.52, 6000, 2.4),
            (0.58, 1200, 1.4),
            (0.71, 11000, 2.1),
            (0.79, 2600, 1.7),
            (0.88, 5200, 1.9),
        ):
            spectral += strength * np.exp(-0.5 * ((u - centre * width) / sigma) ** 2)

        # Scaled so that ~30 ms at gain 200 lands mid-range on a 16-bit scale.
        return ((spectral * across + 12.0) * 20.0).astype(np.float32)

    def _apply_control(self, name: str, value: int) -> None:
        pass  # values are read straight out of _values by _grab

    def _grab(self) -> None:
        exposure_s = self.exposure_us / 1e6
        gain_linear = 10.0 ** (self.gain / 200.0)  # 0.1 dB per unit
        # Cap the simulated wait so huge exposures stay interactive.
        time.sleep(min(exposure_s, 0.5) if exposure_s < 1.0 else min(exposure_s, 3.0))
        if self._stop.is_set():
            return

        adu_per_electron = gain_linear / 8.0
        read_noise_adu = 40.0
        electrons = self._pattern * exposure_s
        image = electrons * adu_per_electron + self.control_value(OFFSET, 10) * 16.0
        # Shot noise and read noise in one draw.
        sigma = np.sqrt(electrons * adu_per_electron**2 + read_noise_adu**2)
        image += sigma * self._rng.standard_normal(image.shape, dtype=np.float32)
        np.clip(image, 0, 65535, out=image)
        # 12-bit sensor read out as 16-bit: quantise to 4096 levels, then shift.
        data = (image.astype(np.uint16) >> 4) << 4
        mode = "snap" if self.exposure_us >= self.snap_threshold_us else "video"
        self._publish(data, mode)
