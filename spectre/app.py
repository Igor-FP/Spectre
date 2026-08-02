"""Application state: camera selection, connection, and the frame pipeline."""

from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np

from . import (
    asi_sdk, calib, camera as cam, chart, darks, display, frameio, reference, wavelength,
)
from .asi_sdk import ImgType
from .camera import BANDWIDTH, EXPOSURE, GAIN, HIGH_SPEED, OFFSET
from .settings import Settings

SIM_LABEL = "Simulator (no hardware)"

ZOOM_MIN, ZOOM_MAX = 0.05, 32.0
MIN_CROP = 16  # smallest crop the calibration can work with

DARK_CONFIRM_POPUP = "Make dark"
DARK_PROGRESS_POPUP = "Making dark"


@dataclass
class CameraEntry:
    label: str
    info: Optional[asi_sdk.CameraInfo] = None  # a real camera
    path: Optional[str] = None  # a saved frame to replay
    simulator: bool = False


def _band_from_settings(settings: Settings) -> Optional[calib.BandResult]:
    """Rebuild just enough of a result to keep drawing the saved band."""
    if not settings.band_valid:
        return None
    return calib.BandResult(
        ok=True,
        message="restored from settings.json",
        angle_deg=settings.band_angle_deg,
        tan_angle=math.tan(math.radians(settings.band_angle_deg)),
        centre_y=settings.band_centre_y,
        edge_lo_y=settings.band_edge_lo_y,
        edge_hi_y=settings.band_edge_hi_y,
        fwhm_px=settings.band_fwhm_px,
        reference_x=settings.band_reference_x,
        x_from=settings.band_x_from,
        x_to=settings.band_x_to,
    )


def _shear_from_settings(
    settings: Settings, band: Optional[calib.BandResult]
) -> Optional[calib.ShearResult]:
    """Rebuild the saved basis so it survives a restart."""
    if not settings.shear_valid:
        return None
    band_angle = band.angle_deg if band is not None else 0.0
    tilt = settings.shear_line_tilt_deg
    band_radians = math.radians(band_angle)
    line_radians = math.radians(tilt)
    tangent = math.tan(line_radians)
    slope = math.tan(band_radians)
    return calib.ShearResult(
        ok=True,
        message="restored from settings.json",
        line_tilt_deg=tilt,
        shear_deg=band_angle + tilt,
        shear_tan=tangent / (1.0 - slope * tangent),
        axis_x=(math.cos(band_radians), math.sin(band_radians)),
        axis_y=(math.sin(line_radians), math.cos(line_radians)),
        band_angle_deg=band_angle,
    )


class App:
    def __init__(
        self,
        settings: Settings,
        dll_path: Optional[str] = None,
        img_type: ImgType = ImgType.RAW16,
        binning: int = 1,
        keep_camera_settings: bool = False,
        extra_files: Optional[List[str]] = None,
    ):
        self.settings = settings
        self.dll_path = dll_path
        self.img_type = img_type
        self.binning = binning
        self.extra_files = list(extra_files or [])
        # When set, exposure/gain/... are read from the camera and left alone -
        # for when they were dialled in elsewhere and must not be overwritten.
        self.keep_camera_settings = keep_camera_settings

        self.running = True
        self.camera: Optional[cam.BaseCamera] = None
        self.entries: List[CameraEntry] = []
        self.selected = 0
        self.sdk_version = ""
        self.sdk_error = ""
        self.connect_error = ""

        self.frame: Optional[cam.Frame] = None
        self.stats = display.ImageStats()
        self.texture = display.ImageTexture()
        self.stretch = display.Stretch(
            auto=settings.auto_stretch,
            black=settings.black,
            white=settings.white,
            midtone=settings.midtone,
            lo_percentile=settings.lo_percentile,
            hi_percentile=settings.hi_percentile,
        )

        # crop (region of interest) in full-frame pixels: [x0, y0, x1, y1),
        # initialised to the whole frame when the first frame arrives
        self.crop = [settings.crop_x0, settings.crop_y0, settings.crop_x1, settings.crop_y1]
        self.show_full_frame = settings.show_full_frame
        self.crop_drag: Optional[str] = None  # "left" | "right" | "top" | "bottom"
        self._stats_crop = None  # crop the current statistics were measured in

        # band geometry calibration
        self.band_params = calib.BandParams(
            angle_range_deg=settings.band_angle_range_deg,
            angle_step_deg=settings.band_angle_step_deg,
            lo_percentile=settings.band_lo_percentile,
            hi_percentile=settings.band_hi_percentile,
            smooth_window=settings.band_smooth_window,
        )
        self.band_finder = calib.BandFinder()
        self.band_result: Optional[calib.BandResult] = _band_from_settings(settings)
        self.show_band_overlay = settings.show_band_overlay
        self.band_status = ""

        # shear: the direction of the spectral lines, the Y axis of the basis
        self.shear_params = calib.ShearParams(blur_scale=settings.shear_blur_scale)
        self.shear_finder = calib.ShearFinder()
        self.shear_result: Optional[calib.ShearResult] = _shear_from_settings(
            settings, self.band_result
        )
        self.shear_status = ""

        # extracted spectrum, shown in its own window under the preview
        self.show_spectrum = settings.show_spectrum
        self.spectrum: Optional[calib.Spectrum] = None
        self.spectrum_texture = display.ImageTexture()
        self.spectrum_height = settings.spectrum_height
        self.strip_ratio = settings.spectrum_strip_ratio
        self.spectrum_status = ""

        #: Running mean over the last N extracted spectra. Averaging the
        #: spectrum rather than the frames costs a few kilobytes instead of
        #: several megabytes each, and comes to the same thing: extraction is a
        #: mean over the band, and a mean of means is a mean.
        self.average_count = max(1, int(settings.spectrum_average))
        self._history = deque(maxlen=self.average_count)
        self._history_key = None  # span the history was collected over
        self._history_frame = -1  # so one frame is never counted twice

        #: Relative measurement: a spectrum kept aside, everything then read as
        #: a per cent of it.  What a filter curve is made of.
        self.baseline: Optional[np.ndarray] = None
        self.baseline_first_column = 0.0
        self.baseline_status = ""
        #: What the graph and the readouts show, and the name of its unit.
        self.spectrum_shown: Optional[np.ndarray] = None
        self.spectrum_unit = "ADU"
        self.export_status = ""

        # wavelength calibration: the reference strip under ours, and the points
        self.reference = reference.load()
        self.reference_texture = display.ImageTexture()
        self.show_reference = settings.show_reference
        self.reference_blur_nm = settings.reference_blur_nm
        self.reference_from_nm = settings.reference_from_nm
        self.reference_to_nm = settings.reference_to_nm
        self.max_degree = settings.wavelength_max_degree
        self.anchors = wavelength.anchors_from_settings(settings.wavelength_anchors)
        self._anchor_serial = 1 + max((point.added for point in self.anchors), default=0)
        #: Points numbered from here on were made in this session.  Undo only
        #: reaches those: for a point read back from settings.json there is no
        #: "last click" to take back.
        self._session_serial = self._anchor_serial
        # The points are what a calibration is; the formula in the file is only
        # a cache of them.  Without enough points behind it, it is not restored -
        # otherwise a leftover formula would come back looking calibrated.
        self.solution: Optional[wavelength.Solution] = (
            wavelength.from_settings(
                settings.wavelength_coefficients,
                settings.wavelength_x_ref,
                settings.wavelength_x_scale,
            )
            if len(self.anchors) >= wavelength.POINTS_FOR_CALIBRATION
            else None
        )
        #: The strips only take clicks while this is on, and the points made are
        #: held in memory until Finish - Cancel has to be able to put back
        #: exactly what was there before.
        self.calibrating = False
        self._snapshot = None
        #: First half of an identification: the reference has been clicked and we
        #: are waiting for the same line on our spectrum.
        self.pending_nm: Optional[float] = None
        self.pending_ref_x: Optional[float] = None  # where that wavelength was clicked
        self.colour_texture = display.ImageTexture()
        self.wavelength_status = ""
        self._solved_span = None  # spectrum span the current solution was made for

        # dark frames: one master dark per gain/exposure, subtracted live
        self.dark_maker = darks.DarkMaker()
        self.use_dark = False
        self.dark: Optional[np.ndarray] = None
        self.dark_key = None  # (gain, exposure key) the loaded dark belongs to
        self.dark_status = ""
        #: Popup the UI should open on its next pass; consumed once.
        self.dark_popup_request = ""
        #: Gain/exposure the disk was last searched for a dark at, so the search
        #: happens when they move and not on every frame.
        self._known_key = None
        #: Kept as the user set it even while a dark is doing the job instead;
        #: when the dark goes, this comes back rather than having to be re-ticked.
        self.use_bias = settings.use_bias
        self.bias_level = settings.bias_level

        # saving frames
        self.save_fits = settings.save_fits
        self.save_npy = settings.save_npy
        self.save_full_frame = settings.save_full_frame
        self.save_status = ""

        self.fit = settings.fit_to_window
        self.zoom = settings.zoom
        self.show_help = settings.show_help
        self.cursor_text = ""
        self.pan_request = [0.0, 0.0]  # filled by keyboard handling, consumed by the view
        self.displayed_frames = 0
        self.display_ms = 0.0  # cost of stretch + stats + upload, per frame

        self.refresh_cameras()

    # -- camera list -------------------------------------------------------

    def refresh_cameras(self) -> None:
        """Re-list cameras, saved frames and the simulator, keeping the selection."""
        previous = self.entries[self.selected].label if self.entries else self.settings.last_camera
        entries = []
        self.sdk_error = ""
        try:
            asi_sdk.load(self.dll_path)
            self.sdk_version = asi_sdk.get_sdk_version()
            for info in asi_sdk.list_cameras():
                entries.append(CameraEntry(f"{info.name} #{info.camera_id}", info=info))
        except (OSError, asi_sdk.ASIError) as exc:
            self.sdk_error = str(exc)

        for path in self.extra_files + frameio.list_captures():
            if any(entry.path == path for entry in entries):
                continue
            entries.append(CameraEntry(f"file: {os.path.basename(path)}", path=path))
        entries.append(CameraEntry(SIM_LABEL, simulator=True))

        self.entries = entries
        self.selected = next(
            (i for i, entry in enumerate(entries) if entry.label == previous), 0
        )

    @property
    def connected(self) -> bool:
        return self.camera is not None

    @property
    def selected_entry(self) -> Optional[CameraEntry]:
        return self.entries[self.selected] if self.entries else None

    # -- connect / disconnect ---------------------------------------------

    def connect(self) -> None:
        self.disconnect()
        entry = self.selected_entry
        if entry is None:
            return
        self.connect_error = ""
        try:
            if entry.path is not None:
                device: cam.BaseCamera = cam.FileCamera(entry.path)
            elif entry.simulator or entry.info is None:
                device = cam.SimulatedCamera()
            else:
                device = cam.AsiCamera(entry.info, img_type=self.img_type, binning=self.binning)
        except Exception as exc:
            self.connect_error = f"{type(exc).__name__}: {exc}"
            return

        self.camera = device
        self.settings.last_camera = entry.label
        self._push_saved_controls(device)
        device.start()

    def disconnect(self) -> None:
        device, self.camera = self.camera, None
        if device is not None:
            self._store_controls(device)
            device.close()
        self.frame = None
        self.stats = display.ImageStats()
        self.cursor_text = ""
        self.texture.release()  # GL context belongs to the calling (UI) thread

    def _push_saved_controls(self, device: cam.BaseCamera) -> None:
        """Apply the last-used exposure/gain/... to a freshly opened camera."""
        if self.keep_camera_settings:
            return
        s = self.settings
        wanted = {
            EXPOSURE: s.exposure_us,
            GAIN: s.gain,
            OFFSET: s.offset,
            BANDWIDTH: s.bandwidth,
            HIGH_SPEED: 1 if s.high_speed else 0,
        }
        for name, value in wanted.items():
            if name not in device.controls:
                continue
            if value is None or value < 0:  # keep whatever the camera reports
                continue
            device.set_control(name, value)

    def _store_controls(self, device: cam.BaseCamera) -> None:
        s = self.settings
        s.exposure_us = device.control_value(EXPOSURE, s.exposure_us)
        s.gain = device.control_value(GAIN, s.gain)
        if OFFSET in device.controls:
            s.offset = device.control_value(OFFSET, s.offset)
        if BANDWIDTH in device.controls:
            s.bandwidth = device.control_value(BANDWIDTH, s.bandwidth)
        if HIGH_SPEED in device.controls:
            s.high_speed = bool(device.control_value(HIGH_SPEED, 0))

    # -- controls ----------------------------------------------------------

    def set_control(self, name: str, value: int) -> None:
        if self.camera is not None:
            self.camera.set_control(name, int(value))

    def control_value(self, name: str, default: int = 0) -> int:
        return self.camera.control_value(name, default) if self.camera else default

    def control_range(self, name: str) -> Optional[cam.ControlRange]:
        return self.camera.controls.get(name) if self.camera else None

    def scale_exposure(self, factor: float) -> None:
        rng = self.control_range(EXPOSURE)
        if rng is None:
            return
        value = int(round(self.control_value(EXPOSURE, 10_000) * factor))
        self.set_control(EXPOSURE, max(rng.min_value, min(rng.max_value, max(value, 1))))

    def nudge_gain(self, delta: int) -> None:
        if self.control_range(GAIN) is not None:
            self.set_control(GAIN, self.control_value(GAIN, 0) + delta)

    def toggle_pause(self) -> None:
        if self.camera is not None:
            self.camera.set_paused(not self.camera.paused)

    # -- frame pipeline ----------------------------------------------------

    def poll_camera(self) -> bool:
        """Pick up the newest frame and refresh stats + texture. True if updated.

        Statistics, histogram and the auto stretch are all measured inside the
        crop: a spectrograph throws glare and the zero order across the rest of
        the frame, and letting that set the display levels (or the numbers you
        read off the panel) is worse than useless.  The texture still covers the
        whole frame so the crop can be placed while looking at it.
        """
        frame = self.camera.latest_frame() if self.camera is not None else None
        crop_changed = tuple(self.crop) != self._stats_crop
        if frame is None and not (crop_changed and self.frame is not None):
            return False

        started = time.perf_counter()
        self.follow_dark_settings()
        if frame is not None:
            self.frame = self.apply_dark(frame)
            self.ensure_crop(frame.width, frame.height)
        frame = self.frame
        region = self.cropped()
        if region is None or region.size == 0:
            return False
        self.stats = display.frame_stats(
            region,
            frame.full_scale,
            self.stretch.lo_percentile,
            self.stretch.hi_percentile,
        )
        if self.stretch.auto:
            self.stretch.autoscale(self.stats)
        self.texture.update(self.stretch.apply_rgba(frame.data, self.saturation_value))
        self._stats_crop = tuple(self.crop)
        self.update_spectrum()
        self.displayed_frames += 1
        self.display_ms = (time.perf_counter() - started) * 1000.0
        return True

    # -- dark frames -------------------------------------------------------

    def current_dark_key(self):
        """(gain, exposure key) the camera is set to right now, or None."""
        if self.camera is None:
            return None
        return darks.dark_key(
            self.control_value(GAIN, 0), self.control_value(EXPOSURE, 0)
        )

    def dark_on_disk(self) -> Optional[str]:
        """The master dark matching the current settings, if there is one."""
        if self.camera is None:
            return None
        return darks.find_dark(
            self.control_value(GAIN, 0), self.control_value(EXPOSURE, 0)
        )

    def set_use_dark(self, on: bool) -> None:
        """Turn subtraction on or off; loading the file is a one-off."""
        if not on:
            self.use_dark = False
            self.dark = None
            self.dark_key = None
            return
        path = self.dark_on_disk()
        if path is None:
            self.dark_status = "no dark for this gain and exposure"
            return
        try:
            data, _ = frameio.load_frame(path)
        except Exception as exc:
            self.dark_status = f"{type(exc).__name__}: {exc}"
            return
        self.dark = data
        self.dark_key = self.current_dark_key()
        self.use_dark = True
        self.dark_status = f"using {os.path.basename(path)}"

    @property
    def dark_applies(self) -> bool:
        return self.use_dark and self.dark is not None

    @property
    def bias_applies(self) -> bool:
        """The bias only stands in while no dark is doing the job."""
        return self.use_bias and not self.dark_applies

    def follow_dark_settings(self) -> None:
        """Keep the tick in step with what is on disk for the current settings.

        Run when the gain or the exposure has moved, and once at startup: if
        there is a master dark for where they are now, it goes on; if there is
        not, it goes off.  The disk is only looked at on a change, so unticking
        the box by hand is not undone on the next frame.
        """
        key = self.current_dark_key()
        if key is None or key == self._known_key:
            return
        self._known_key = key
        if self.dark_on_disk() is not None:
            self.set_use_dark(True)
        elif self.use_dark:
            self.set_use_dark(False)
            self.dark_status = "no dark for these settings"

    def apply_dark(self, frame: cam.Frame) -> cam.Frame:
        """Take the dark off, or the flat bias level if there is no dark.

        A dark only belongs to the gain and exposure it was taken at.  A frame
        still in flight from before a slider moved is left alone rather than
        corrected with the wrong picture; the tick itself is not touched here,
        `follow_dark_settings` owns that.
        """
        if self.dark_applies:
            if self.dark.shape != frame.data.shape:
                self.set_use_dark(False)
                self.dark_status = "dark does not match the frame size: switched off"
            elif darks.dark_key(frame.gain, frame.exposure_us) == self.dark_key:
                return replace(frame, data=darks.subtract(frame.data, self.dark))
            else:
                return frame  # exposed before the change: not ours to correct
        if self.bias_applies and self.bias_level > 0:
            return replace(frame, data=darks.subtract(frame.data, self.bias_level))
        return frame

    def start_dark_capture(self) -> None:
        if self.camera is None:
            self.dark_status = "connect a camera first"
            return
        self.dark_status = ""
        if self.dark_maker.start(self.camera):
            self.dark_popup_request = DARK_PROGRESS_POPUP

    def cancel_dark_capture(self) -> None:
        self.dark_maker.cancel()

    def poll_dark_capture(self) -> None:
        """Pick up a finished master dark and switch it on."""
        error = self.dark_maker.take_error()
        if error:
            self.dark_status = error
            return
        result = self.dark_maker.take_result()
        if result is None:
            return
        path, gain, exposure_us = result
        self.dark_status = f"made {os.path.basename(path)}"
        # It was made for what the camera is set to now, so use it.
        self.set_use_dark(True)

    @property
    def saturation_value(self) -> int:
        """Pixel value that means the sensor clipped, for this camera and format."""
        if self.frame is None:
            return 65535
        return display.saturation_value(self.frame.full_scale)

    def redraw_texture(self) -> None:
        """Re-apply the stretch to the current frame (after a stretch change)."""
        if self.frame is not None:
            self.texture.update(
                self.stretch.apply_rgba(self.frame.data, self.saturation_value)
            )

    # -- saving frames -----------------------------------------------------

    def save_frame(self) -> None:
        """Write the current frame to captures/ as FITS and/or .npy."""
        if self.frame is None:
            self.save_status = "no frame to save"
            return
        if not (self.save_fits or self.save_npy):
            self.save_status = "pick at least one format"
            return
        camera_stats = self.camera.stats() if self.camera is not None else None
        band = None
        if self.band_result is not None and self.band_result.ok:
            result = self.band_result
            band = {
                "BANDANG": result.angle_deg,
                "BANDCY": result.centre_y,
                "BANDW": result.fwhm_px,
                "BANDLO": result.edge_lo_y,
                "BANDHI": result.edge_hi_y,
                "BANDREFX": result.reference_x,
            }
        # The crop is the working image, so that is what gets saved; the header
        # records where it came from on the sensor.
        data = self.frame.data if self.save_full_frame else self.cropped()
        meta = frameio.FrameMeta(
            camera=self.camera.name if self.camera is not None else "",
            exposure_us=self.frame.exposure_us,
            gain=self.frame.gain,
            offset=self.control_value(OFFSET, -1) if self.camera else None,
            binning=getattr(self.camera, "binning", 1) if self.camera else 1,
            bit_depth=self.camera.bit_depth if self.camera is not None else 16,
            pixel_size_um=self.camera.pixel_size_um if self.camera is not None else 0.0,
            temperature_c=camera_stats.temperature_c if camera_stats else None,
            crop=tuple(int(v) for v in self.crop),
            band=band,
        )
        try:
            written = frameio.save_frame(
                data, meta, write_fits=self.save_fits, write_npy=self.save_npy
            )
        except Exception as exc:
            self.save_status = f"{type(exc).__name__}: {exc}"
            return
        self.save_status = "saved " + ", ".join(os.path.basename(p) for p in written)

    # -- crop --------------------------------------------------------------

    def ensure_crop(self, width: int, height: int) -> None:
        """Initialise the crop to the whole frame and keep it inside it."""
        x0, y0, x1, y1 = self.crop
        if min(self.crop) < 0 or x1 <= x0 or y1 <= y0:
            self.crop = [0, 0, int(width), int(height)]
            return
        x0 = int(np.clip(x0, 0, width - 2))
        y0 = int(np.clip(y0, 0, height - 2))
        x1 = int(np.clip(x1, x0 + MIN_CROP, width))
        y1 = int(np.clip(y1, y0 + MIN_CROP, height))
        self.crop = [x0, y0, x1, y1]

    def reset_crop(self) -> None:
        if self.frame is not None:
            self.crop = [0, 0, self.frame.width, self.frame.height]

    @property
    def crop_size(self):
        x0, y0, x1, y1 = self.crop
        return max(0, x1 - x0), max(0, y1 - y0)

    def cropped(self) -> Optional[np.ndarray]:
        """The current frame restricted to the crop."""
        if self.frame is None:
            return None
        x0, y0, x1, y1 = self.crop
        return self.frame.data[y0:y1, x0:x1]

    # -- band geometry calibration ----------------------------------------

    def start_band_search(self) -> None:
        """Freeze the preview and look for the band inside the crop."""
        if self.band_finder.running:
            return
        if self.frame is None:
            self.band_status = "no frame to measure - connect a camera first"
            return
        crop = self.cropped()
        if crop is None or min(crop.shape) < 16:
            self.band_status = "crop is too small to measure"
            return
        if self.camera is not None:
            self.camera.set_paused(True)  # calibrate on one steady frame
        self.band_status = ""
        self.band_finder.start(crop, self.band_params)

    def cancel_band_search(self) -> None:
        self.band_finder.cancel()

    def poll_band_search(self) -> None:
        result = self.band_finder.take_result()
        if result is None:
            return
        self.band_status = result.message
        if result.ok:
            # measured inside the crop, reported in full-frame coordinates
            self.band_result = result.shifted(self.crop[0], self.crop[1])
            self._store_band(self.band_result)
            # The shear was measured against the previous band, so it no longer
            # applies: it has to be searched again.
            self.clear_shear()
        # A failed search keeps the previous calibration: it is more likely that
        # this one frame was bad than that the old numbers went wrong.

    def clear_band(self) -> None:
        self.band_result = None
        self.band_status = ""
        self.settings.band_valid = False
        self.clear_shear()  # the shear is measured against the band, so it goes too

    # -- shear: the direction of the spectral lines ------------------------

    def band_in_crop(self) -> Optional[calib.BandResult]:
        """The band geometry expressed in crop coordinates."""
        if self.band_result is None or not self.band_result.ok:
            return None
        return self.band_result.shifted(-self.crop[0], -self.crop[1])

    def start_shear_search(self) -> None:
        """Freeze the preview and look for the direction of the spectral lines."""
        if self.shear_finder.running:
            return
        band = self.band_in_crop()
        if band is None:
            self.shear_status = "measure the band angle first"
            return
        crop = self.cropped()
        if crop is None or min(crop.shape) < 16:
            self.shear_status = "crop is too small to measure"
            return
        if self.camera is not None:
            self.camera.set_paused(True)
        self.shear_status = ""
        self.shear_finder.start(crop, band, self.shear_params)

    def cancel_shear_search(self) -> None:
        self.shear_finder.cancel()

    def poll_shear_search(self) -> None:
        result = self.shear_finder.take_result()
        if result is None:
            return
        self.shear_status = result.message
        if result.ok:
            self.shear_result = result
            self.settings.shear_valid = True
            self.settings.shear_line_tilt_deg = result.line_tilt_deg

    def clear_shear(self) -> None:
        self.shear_result = None
        self.shear_status = ""
        self.settings.shear_valid = False
        self.spectrum = None  # the spectrum is extracted along the spectral lines

    # -- the extracted spectrum --------------------------------------------

    @property
    def can_capture_spectrum(self) -> bool:
        return (
            self.frame is not None
            and self.band_result is not None
            and self.band_result.ok
            and self.shear_result is not None
            and self.shear_result.ok
        )

    def capture_spectrum(self) -> None:
        """Open the spectrum window and extract from the frame on screen."""
        if not self.can_capture_spectrum:
            self.spectrum_status = "calibrate the band angle and the shear first"
            return
        self.show_spectrum = True
        self.update_spectrum()

    def close_spectrum(self) -> None:
        self.show_spectrum = False

    def update_spectrum(self) -> None:
        """Re-extract the spectrum from the current frame and refresh its strip."""
        if not self.show_spectrum or not self.can_capture_spectrum:
            return
        crop = self.cropped()
        band = self.band_in_crop()
        if crop is None or band is None:
            return
        spectrum = calib.extract_spectrum(
            crop, band, self.shear_result, self.band_params.angle_range_deg
        )
        spectrum.frame_index = self.frame.index if self.frame is not None else 0
        self.spectrum_status = "" if spectrum.ok else spectrum.message
        if not spectrum.ok:
            self.spectrum = None
            return
        spectrum.values = self._averaged(spectrum)
        self.spectrum = spectrum
        # The strip is shown on its own scale: these are means over the band, not
        # raw pixel values, so the image stretch does not apply to them.
        values = spectrum.values.astype(np.float32)
        low, high = float(values.min()), float(values.max())
        span = high - low
        scaled = (values - low) * (255.0 / span) if span > 0 else np.zeros_like(values)
        self.spectrum_texture.update(
            np.ascontiguousarray(scaled.astype(np.uint8)).reshape(1, -1)
        )
        self.refresh_display()
        self.refresh_wavelength(force=False)

    # -- averaging over the last N spectra ---------------------------------

    def set_average_count(self, count: int) -> None:
        """How many spectra to average. 1 is off; the history keeps what it can."""
        count = max(1, min(int(count), 500))
        if count == self.average_count:
            return
        self.average_count = count
        kept = list(self._history)[-count:]
        self._history = deque(kept, maxlen=count)
        self.save_state()

    def _averaged(self, spectrum) -> np.ndarray:
        """The mean of the last N spectra, or this one when N is 1.

        The history is thrown away whenever the spectrum stops covering the same
        columns - a moved crop or a re-measured band - because averaging two
        spectra that are not on the same axis would smear the lines rather than
        the noise.
        """
        key = (spectrum.length, spectrum.first_column)
        if key != self._history_key:
            self._history.clear()
            self._history_key = key
        if spectrum.frame_index != self._history_frame:
            self._history.append(spectrum.values)
            self._history_frame = spectrum.frame_index
        if self.average_count <= 1 or len(self._history) <= 1:
            return spectrum.values
        return np.mean(np.stack(self._history), axis=0, dtype=np.float64).astype(np.float32)

    @property
    def averaged_over(self) -> int:
        """How many spectra the current curve is actually a mean of."""
        return max(1, len(self._history)) if self.average_count > 1 else 1

    # -- relative measurement ---------------------------------------------

    def refresh_display(self) -> None:
        """What the graph, the readouts and the export show.

        Always a per cent: of the comparison spectrum when one is set, of the
        brightest sample otherwise.  The curve has the same shape either way -
        both are a linear rescale - so only the unit changes.
        """
        spectrum = self.spectrum
        if spectrum is None or not spectrum.ok or spectrum.length == 0:
            self.spectrum_shown = None
            return
        values = spectrum.values.astype(np.float64)
        if self.baseline is None:
            peak = float(values.max())
            shown = 100.0 * values / peak if peak > 0 else np.zeros_like(values)
            self.spectrum_unit = "% of peak"
        else:
            shown = 100.0 * values / self._aligned_baseline(spectrum)
            self.spectrum_unit = "% of baseline"
        self.spectrum_shown = shown.astype(np.float32)

    def _aligned_baseline(self, spectrum) -> np.ndarray:
        """The comparison spectrum on the current spectrum's columns.

        Lined up by frame column rather than by index, so moving the crop
        between the two exposures does not silently shift one against the other.
        """
        first = self.x_of_index(0)
        if spectrum.length == self.baseline.size and abs(first - self.baseline_first_column) < 0.5:
            self.baseline_status = ""
            return self.baseline
        self.baseline_status = "baseline covers a different span: interpolated onto this one"
        columns = np.arange(spectrum.length, dtype=np.float64) + first
        base_columns = np.arange(self.baseline.size, dtype=np.float64) + self.baseline_first_column
        return np.interp(columns, base_columns, self.baseline)

    def set_baseline(self) -> None:
        """Keep the current spectrum aside as the one everything is divided by."""
        spectrum = self.spectrum
        if spectrum is None or not spectrum.ok or spectrum.length == 0:
            self.baseline_status = "no spectrum to take as a baseline"
            return
        values = spectrum.values.astype(np.float64)
        # Zero and below become one, so the division always has something to
        # divide by.  Interpolation between clamped samples stays >= 1 as well.
        self.baseline = np.where(values <= 0.0, 1.0, values)
        self.baseline_first_column = self.x_of_index(0)
        self.baseline_status = f"baseline taken from frame {spectrum.frame_index}"
        self.refresh_display()

    def clear_baseline(self) -> None:
        self.baseline = None
        self.baseline_status = ""
        self.refresh_display()

    def _export_pieces(self):
        """(columns, wavelengths or None, notes) shared by both exports."""
        spectrum = self.spectrum
        columns = np.arange(spectrum.length, dtype=np.float64) + self.x_of_index(0)
        wavelengths = self.solution.lambda_of_x(columns) if self.calibrated else None
        notes = [
            f"unit: {self.spectrum_unit}",
            f"band {spectrum.band_angle_deg:+.4f} deg, lines {spectrum.line_tilt_deg:+.4f} deg",
            f"averaged over {spectrum.rows_averaged} rows",
        ]
        if self.averaged_over > 1:
            notes.append(f"mean of {self.averaged_over} measurements")
        if self.calibrated:
            notes.append(self.solution.describe())
        if self.baseline is not None:
            notes.append("values are per cent of a comparison spectrum")
        return columns, wavelengths, notes

    def export_csv(self) -> None:
        """Write the curve as it is drawn: wavelength and per cent."""
        if self.spectrum is None or self.spectrum_shown is None:
            self.export_status = "no spectrum to export"
            return
        columns, wavelengths, notes = self._export_pieces()
        try:
            path = frameio.save_spectrum_csv(
                columns, self.spectrum_shown, wavelengths, self.spectrum_unit, notes
            )
        except Exception as exc:
            self.export_status = f"{type(exc).__name__}: {exc}"
            return
        self.export_status = f"exported {os.path.basename(path)}"

    def export_chart(self) -> None:
        """Write the curve as a picture: PNG at 4K and SVG beside it."""
        if self.spectrum is None or self.spectrum_shown is None:
            self.export_status = "no spectrum to export"
            return
        columns, wavelengths, notes = self._export_pieces()
        stem = os.path.join(
            frameio.CAPTURE_DIR, frameio.timestamped_name("spectrum")
        )
        os.makedirs(frameio.CAPTURE_DIR, exist_ok=True)
        title = "Spectre - " + os.path.basename(stem).replace("spectrum_", "")
        if self.averaged_over > 1:
            title += f", mean of {self.averaged_over} measurements"
        try:
            written = chart.export(
                stem,
                wavelengths if self.calibrated else columns,
                self.spectrum_shown,
                self.spectrum_unit,
                calibrated=self.calibrated,
                title=title,
                notes=notes,
                relative=self.baseline is not None,
            )
        except Exception as exc:
            self.export_status = f"{type(exc).__name__}: {exc}"
            return
        self.export_status = "exported " + ", ".join(
            os.path.basename(path) for path in written
        )

    # -- wavelength calibration --------------------------------------------

    def spectrum_span(self):
        """(first, last) full-frame column the spectrum covers, or None."""
        spectrum = self.spectrum
        if spectrum is None or not spectrum.ok or spectrum.length < 2:
            return None
        first = self.crop[0] + spectrum.first_column
        return first, first + spectrum.length - 1

    def x_of_index(self, index: float) -> float:
        """Full-frame column of a sample of the extracted spectrum."""
        span = self.spectrum_span()
        return (span[0] + float(index)) if span is not None else float(index)

    def index_of_x(self, x_px: float) -> float:
        span = self.spectrum_span()
        return (float(x_px) - span[0]) if span is not None else float(x_px)

    def refresh_wavelength(self, force: bool = True) -> None:
        """Re-fit the mapping and redraw the reference strip.

        Called with `force` when something the user touched changed, and without
        it on every new frame, where only a changed spectrum span matters.
        """
        span = self.spectrum_span()
        if span is None:
            return
        if force or span != self._solved_span:
            self.solution = wavelength.solve(
                self.anchors,
                span[0],
                span[1],
                self.reference_from_nm,
                self.reference_to_nm,
                self.max_degree,
            )
            self._solved_span = span
            self.wavelength_status = "" if self.solution.ok else self.solution.message
            self._store_solution()
        self.update_reference_strip()

    def update_reference_strip(self) -> None:
        """Resample the reference into our pixels and upload it as a strip.

        Our spectrum is never touched: the reference is evaluated at the
        wavelength each of our columns maps to, so both strips share one X axis
        and the lines line up vertically when the calibration is right.  The
        illustrative colour bar comes off the same wavelengths.
        """
        spectrum = self.spectrum
        solution = self.solution
        if spectrum is None or solution is None or not solution.ok:
            return
        wavelengths = solution.lambda_of_x(
            np.arange(spectrum.length, dtype=np.float64) + self.x_of_index(0)
        )
        self.colour_texture.update(
            np.ascontiguousarray(display.wavelength_rgba(wavelengths)).reshape(1, -1)
        )
        if not self.reference.ok:
            return
        values = self.reference.sample(wavelengths, self.reference_blur_nm)
        inside = np.isfinite(values)
        scaled = np.zeros(values.shape, dtype=np.float32)  # off the atlas: black
        if inside.any():
            low = float(values[inside].min())
            high = float(values[inside].max())
            span = high - low
            if span > 0:
                scaled[inside] = (values[inside] - low) * (255.0 / span)
        self.reference_texture.update(
            np.ascontiguousarray(scaled.astype(np.uint8)).reshape(1, -1)
        )

    def wavelength_at(self, x_px: float) -> Optional[float]:
        if self.solution is None or not self.solution.ok:
            return None
        return float(self.solution.lambda_of_x(float(x_px)))

    # -- the two clicks that make one point -------------------------------
    #
    # Always the reference first, then our spectrum.  The order is fixed on
    # purpose: the yellow line appears on the reference and nowhere else, so
    # there is only ever one place to click next.

    def pick_reference(self, x_px: float) -> None:
        """First click: the wavelength the reference strip shows at this column.

        The wavelength comes from the same mapping the strip was drawn with, so
        it is exactly the feature under the cursor whatever the mapping is
        currently worth.
        """
        if not self.calibrating or self.pending_nm is not None:
            return
        value = self.wavelength_at(x_px)
        if value is None:
            return
        self.pending_nm = value
        self.pending_ref_x = float(x_px)

    def pick_our_spectrum(self, x_px: float) -> None:
        """Second click: the column of our spectrum that same line sits at."""
        if not self.calibrating or self.pending_nm is None:
            return
        self.add_anchor(float(x_px), self.pending_nm)
        self.cancel_pending()

    # -- entering and leaving the calibration ------------------------------

    #: Points needed before the calibration exists at all.
    POINTS_TO_FINISH = wavelength.POINTS_FOR_CALIBRATION

    def start_calibration(self) -> None:
        """Turn on the mode, remembering what to go back to if it is cancelled."""
        if self.calibrating:
            return
        s = self.settings
        self._snapshot = (
            [replace(point) for point in self.anchors],
            list(s.wavelength_coefficients),
            s.wavelength_x_ref,
            s.wavelength_x_scale,
            s.wavelength_valid,
        )
        self.calibrating = True
        self.show_reference = True  # there is nothing to click on without it
        self.cancel_pending()

    @property
    def can_finish_calibration(self) -> bool:
        return self.calibrating and len(self.anchors) >= self.POINTS_TO_FINISH

    def finish_calibration(self) -> None:
        """Keep the points made and write them out."""
        self.calibrating = False
        self._snapshot = None
        self.cancel_pending()
        self.refresh_wavelength()
        self.save_state()

    def cancel_calibration(self) -> None:
        """Leave the mode and put back what was there before it started."""
        if self._snapshot is not None:
            anchors, coefficients, x_ref, x_scale, valid = self._snapshot
            self.anchors = anchors
            self._store_snapshot_settings(coefficients, x_ref, x_scale, valid)
        self._snapshot = None
        self.calibrating = False
        self.cancel_pending()
        self.refresh_wavelength()
        # Nothing was written while the mode was on, so the file already holds
        # what we have just gone back to.

    def _store_snapshot_settings(self, coefficients, x_ref, x_scale, valid) -> None:
        s = self.settings
        s.wavelength_coefficients = list(coefficients)
        s.wavelength_x_ref = float(x_ref)
        s.wavelength_x_scale = float(x_scale)
        s.wavelength_valid = bool(valid)

    def add_anchor(self, x_px: float, wavelength_nm: float) -> None:
        label = self.reference.nearest_label(wavelength_nm) if self.reference.ok else ""
        self.anchors.append(
            wavelength.Anchor(float(x_px), float(wavelength_nm), label, self._anchor_serial)
        )
        self._anchor_serial += 1
        self.anchors.sort(key=lambda point: point.x_px)
        self.refresh_wavelength()
        self._save_unless_calibrating()

    def remove_anchor(self, index: int) -> None:
        if 0 <= index < len(self.anchors):
            del self.anchors[index]
            self.refresh_wavelength()
            self._save_unless_calibrating()

    def _save_unless_calibrating(self) -> None:
        """While the mode is on the points are a draft; Finish is what writes."""
        if not self.calibrating:
            self.save_state()

    @property
    def pending_pick(self) -> bool:
        return self.pending_nm is not None

    def _session_anchors(self) -> List[int]:
        """Indices of the points made since the program started."""
        return [
            index
            for index, point in enumerate(self.anchors)
            if point.added >= self._session_serial
        ]

    @property
    def can_undo_wavelength(self) -> bool:
        """True only while there is a click of this session left to take back."""
        return self.pending_pick or bool(self._session_anchors())

    @property
    def can_reset_wavelength(self) -> bool:
        """Reset does reach a calibration read back from settings.json."""
        return bool(self.anchors) or self.pending_pick or bool(
            self.settings.wavelength_coefficients
        )

    def undo_wavelength(self) -> None:
        """Take back the last click: the half-made pair first, then the last point."""
        if self.pending_pick:
            self.cancel_pending()
            return
        made_here = self._session_anchors()
        if made_here:
            self.remove_anchor(max(made_here, key=lambda i: self.anchors[i].added))

    def reset_wavelength(self) -> None:
        """Drop the wavelength calibration only, and leave the mode if it is on.

        The band angle, the shear and the extracted spectrum are left alone: they
        are a different calibration and re-measuring them is expensive.
        """
        self.anchors = []
        self.calibrating = False
        self._snapshot = None
        self.cancel_pending()
        self._store_snapshot_settings([], 0.0, 1.0, False)
        self.refresh_wavelength()
        self.save_state()

    def cancel_pending(self) -> None:
        self.pending_nm = self.pending_ref_x = None

    @property
    def calibrated(self) -> bool:
        """Either the calibration is complete, or there is none. No middle."""
        solution = self.solution
        return (
            len(self.anchors) >= wavelength.POINTS_FOR_CALIBRATION
            and solution is not None
            and solution.ok
            and solution.kind == "fit"
        )

    def _store_solution(self) -> None:
        s = self.settings
        solution = self.solution
        s.wavelength_anchors = [point.as_dict() for point in self.anchors]
        s.wavelength_valid = self.calibrated
        if self.calibrated:
            s.wavelength_coefficients = [float(c) for c in solution.coefficients]
            s.wavelength_x_ref = float(solution.x_ref)
            s.wavelength_x_scale = float(solution.x_scale)
        else:
            # A formula with too few points behind it must not survive in the
            # file: it would come back next launch looking like a finished job.
            s.wavelength_coefficients = []
            s.wavelength_x_ref = 0.0
            s.wavelength_x_scale = 1.0

    def _store_band(self, result: calib.BandResult) -> None:
        s = self.settings
        s.band_valid = True
        s.band_angle_deg = result.angle_deg
        s.band_centre_y = result.centre_y
        s.band_edge_lo_y = result.edge_lo_y
        s.band_edge_hi_y = result.edge_hi_y
        s.band_fwhm_px = result.fwhm_px
        s.band_reference_x = result.reference_x
        s.band_x_from = result.x_from
        s.band_x_to = result.x_to

    def pixel_value(self, x: int, y: int) -> Optional[int]:
        frame = self.frame
        if frame is None or not (0 <= x < frame.width and 0 <= y < frame.height):
            return None
        return int(frame.data[y, x])

    # -- shutdown ----------------------------------------------------------

    def save_state(self) -> None:
        """Write settings.json now, without touching the camera.

        Anchor points are made by hand, one line at a time; losing a session of
        them to a crash or a power cut would be worse than the cost of writing a
        two-kilobyte file on every click.
        """
        self._store_ui_state()
        self.settings.save()

    def save_settings(self) -> None:
        if self.camera is not None:
            self._store_controls(self.camera)
        self._store_ui_state()
        self.settings.save()

    def _store_ui_state(self) -> None:
        """Copy everything the UI owns into the settings object."""
        s = self.settings
        s.auto_stretch = self.stretch.auto
        s.black = self.stretch.black
        s.white = self.stretch.white
        s.midtone = self.stretch.midtone
        s.lo_percentile = self.stretch.lo_percentile
        s.hi_percentile = self.stretch.hi_percentile
        s.fit_to_window = self.fit
        s.zoom = self.zoom
        s.show_help = self.show_help
        s.show_band_overlay = self.show_band_overlay
        s.show_full_frame = self.show_full_frame
        s.save_fits = self.save_fits
        s.save_npy = self.save_npy
        s.save_full_frame = self.save_full_frame
        s.use_bias = self.use_bias
        s.bias_level = self.bias_level
        s.show_spectrum = self.show_spectrum
        s.spectrum_height = self.spectrum_height
        s.spectrum_strip_ratio = self.strip_ratio
        s.spectrum_average = self.average_count
        s.show_reference = self.show_reference
        s.reference_blur_nm = self.reference_blur_nm
        s.reference_from_nm = self.reference_from_nm
        s.reference_to_nm = self.reference_to_nm
        s.wavelength_max_degree = self.max_degree
        self._store_solution()
        s.crop_x0, s.crop_y0, s.crop_x1, s.crop_y1 = [int(v) for v in self.crop]
        p = self.band_params
        s.band_angle_range_deg = p.angle_range_deg
        s.band_angle_step_deg = p.angle_step_deg
        s.band_lo_percentile = p.lo_percentile
        s.band_hi_percentile = p.hi_percentile
        s.band_smooth_window = p.smooth_window
        s.shear_blur_scale = self.shear_params.blur_scale

    def shutdown(self) -> None:
        self.band_finder.cancel()
        self.shear_finder.cancel()
        self.dark_maker.cancel()
        self.save_settings()
        self.disconnect()
        self.texture.release()
        self.spectrum_texture.release()
        self.reference_texture.release()
        self.colour_texture.release()
