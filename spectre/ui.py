"""ImGui layout: camera controls on the left, preview in the middle, spectrum
work on the right."""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
from imgui_bundle import imgui

from .app import App, MIN_CROP, ZOOM_MAX, ZOOM_MIN
from .camera import BANDWIDTH, EXPOSURE, GAIN, HIGH_SPEED, OFFSET

ImVec2 = imgui.ImVec2
ImVec4 = imgui.ImVec4

RED = ImVec4(1.0, 0.45, 0.4, 1.0)
YELLOW = ImVec4(1.0, 0.85, 0.35, 1.0)
GREEN = ImVec4(0.5, 0.9, 0.55, 1.0)
GREY = ImVec4(0.65, 0.65, 0.68, 1.0)
CYAN = ImVec4(0.45, 0.85, 1.0, 1.0)


def _rgba(red: int, green: int, blue: int, alpha: int = 255) -> int:
    """Pack a colour the way ImDrawList wants it (0xAABBGGRR)."""
    return (alpha << 24) | (blue << 16) | (green << 8) | red


BAND_EDGE_COLOUR = _rgba(90, 255, 120)
BAND_CENTRE_COLOUR = _rgba(90, 255, 120, 110)
SHEAR_LINE_COLOUR = _rgba(90, 255, 120, 150)
SHEAR_LINE_COUNT = 9  # dashed lines across the band, showing the line direction
GRIP_COLOUR = _rgba(140, 170, 210, 200)
CROP_COLOUR = _rgba(255, 70, 70, 220)
CROP_ACTIVE_COLOUR = _rgba(255, 160, 90)
CROP_DIM_COLOUR = _rgba(0, 0, 0, 110)

_FIXED_WINDOW = (
    imgui.WindowFlags_.no_title_bar
    | imgui.WindowFlags_.no_resize
    | imgui.WindowFlags_.no_move
    | imgui.WindowFlags_.no_collapse
    | imgui.WindowFlags_.no_saved_settings
    | imgui.WindowFlags_.no_bring_to_front_on_focus
)
_OPEN = imgui.TreeNodeFlags_.default_open

ZOOM_STEP = 1.25
EXPOSURE_STEP = 1.25
GAIN_STEP = 5


# ---------------------------------------------------------------------------
# Frame entry point
# ---------------------------------------------------------------------------


MIN_PANEL_W = 300.0
MIN_PREVIEW_W = 240.0


def _panel_widths(app: App, total: float):
    """Left and right panel widths, shrunk together if the window is narrow."""
    left = max(MIN_PANEL_W, app.settings.panel_width)
    right = max(MIN_PANEL_W, app.settings.right_panel_width)
    spare = total - MIN_PREVIEW_W
    if left + right > spare:
        scale = max(0.2, spare / (left + right))
        left, right = left * scale, right * scale
    return left, right


def draw(app: App) -> None:
    io = imgui.get_io()
    width, height = io.display_size.x, io.display_size.y
    left_w, right_w = _panel_widths(app, width)

    imgui.set_next_window_pos(ImVec2(0, 0))
    imgui.set_next_window_size(ImVec2(left_w, height))
    imgui.begin("##panel", None, _FIXED_WINDOW)
    _control_panel(app)
    imgui.end()

    middle_w = max(80.0, width - left_w - right_w)
    spectrum_h = 0.0
    if app.show_spectrum:
        spectrum_h = float(np.clip(app.spectrum_height, 140.0, max(140.0, height - 240.0)))
        app.spectrum_height = spectrum_h

    imgui.set_next_window_pos(ImVec2(left_w, 0))
    imgui.set_next_window_size(ImVec2(middle_w, height - spectrum_h))
    imgui.begin("##preview", None, _FIXED_WINDOW)
    _preview(app)
    imgui.end()

    if app.show_spectrum:
        imgui.set_next_window_pos(ImVec2(left_w, height - spectrum_h))
        imgui.set_next_window_size(ImVec2(middle_w, spectrum_h))
        imgui.begin("##spectrum_window", None, _FIXED_WINDOW)
        _spectrum_window(app)
        imgui.end()

    imgui.set_next_window_pos(ImVec2(width - right_w, 0))
    imgui.set_next_window_size(ImVec2(right_w, height))
    imgui.begin("##spectrum", None, _FIXED_WINDOW)
    _spectrum_panel(app)
    imgui.end()

    if app.show_help:
        _help_window(app)


# ---------------------------------------------------------------------------
# Left panel
# ---------------------------------------------------------------------------


def _control_panel(app: App) -> None:
    _camera_section(app)
    _acquisition_section(app)
    _display_section(app)
    _statistics_section(app)
    _save_section(app)
    _view_section(app)


def _save_section(app: App) -> None:
    if not imgui.collapsing_header("Save frame", _OPEN):
        return
    imgui.begin_disabled(app.frame is None)
    if imgui.button("Save current frame  (S)", ImVec2(-1, 0)):
        app.save_frame()
    imgui.end_disabled()
    _, app.save_fits = imgui.checkbox("FITS", app.save_fits)
    imgui.same_line()
    _, app.save_npy = imgui.checkbox("NumPy .npy", app.save_npy)
    _, app.save_full_frame = imgui.checkbox("Whole frame instead of the crop", app.save_full_frame)
    width, height = app.crop_size
    imgui.push_text_wrap_pos(0.0)
    imgui.text_colored(
        GREY,
        f"Raw linear pixels as the camera delivered them, into captures/ next to "
        f"the app: {'whole frame' if app.save_full_frame else f'the crop, {width} x {height} px'}. "
        "Exposure, gain, the crop position on the sensor and the band calibration "
        "go into the FITS header.",
    )
    if app.save_status:
        imgui.text_colored(GREEN if app.save_status.startswith("saved") else RED, app.save_status)
    imgui.pop_text_wrap_pos()
    imgui.spacing()


def _camera_section(app: App) -> None:
    if not imgui.collapsing_header("Camera", _OPEN):
        return

    imgui.begin_disabled(app.connected)
    imgui.set_next_item_width(-1)
    changed, app.selected = imgui.combo(
        "##camera", app.selected, [entry.label for entry in app.entries]
    )
    imgui.end_disabled()

    half = (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5
    if app.connected:
        if imgui.button("Disconnect", ImVec2(half, 0)):
            app.disconnect()
    else:
        if imgui.button("Connect", ImVec2(half, 0)):
            app.connect()
    imgui.same_line()
    imgui.begin_disabled(app.connected)
    if imgui.button("Refresh list", ImVec2(half, 0)):
        app.refresh_cameras()
    imgui.end_disabled()

    if app.sdk_error:
        imgui.text_colored(RED, "ASI SDK not available")
        if imgui.is_item_hovered():
            imgui.set_tooltip(app.sdk_error)
    elif app.sdk_version:
        imgui.text_colored(GREY, f"ASI SDK {app.sdk_version}")
    if app.connect_error:
        imgui.text_colored(RED, app.connect_error)

    device = app.camera
    if device is not None:
        imgui.text_colored(GREEN, f"Connected: {device.name}")
        imgui.text_colored(
            GREY,
            f"{device.width}x{device.height}  {device.bit_depth}-bit  "
            f"{device.img_type.name}  {device.pixel_size_um:.2f} um",
        )
        serial = getattr(device, "serial", None)
        if serial:
            imgui.text_colored(GREY, f"S/N {serial}")
    else:
        imgui.text_colored(GREY, "Not connected")
    imgui.spacing()


def _acquisition_section(app: App) -> None:
    if not imgui.collapsing_header("Exposure / Gain", _OPEN):
        return
    if not app.connected:
        imgui.text_colored(GREY, "Connect a camera to enable controls.")
        imgui.spacing()
        return

    device = app.camera

    # -- exposure ---------------------------------------------------------
    rng = app.control_range(EXPOSURE)
    if rng is not None:
        exposure_us = app.control_value(EXPOSURE, 20_000)
        lo_ms = max(rng.min_value / 1000.0, 0.001)
        hi_ms = min(rng.max_value / 1000.0, 60_000.0)  # 60 s on the slider
        ms = exposure_us / 1000.0
        imgui.text(f"Exposure   ({exposure_us} us)")
        imgui.set_next_item_width(-1)
        changed, ms = imgui.slider_float(
            "##exposure",
            float(np.clip(ms, lo_ms, hi_ms)),
            lo_ms,
            hi_ms,
            _exposure_format(ms),
            imgui.SliderFlags_.logarithmic,
        )
        if changed:
            app.set_control(EXPOSURE, max(rng.min_value, int(round(ms * 1000.0))))

        spacing = imgui.get_style().item_spacing.x
        button_w = (imgui.get_content_region_avail().x - 4 * spacing) / 5.0
        for label, action in (
            ("/2", lambda: app.scale_exposure(0.5)),
            ("x2", lambda: app.scale_exposure(2.0)),
            ("10ms", lambda: app.set_control(EXPOSURE, 10_000)),
            ("100ms", lambda: app.set_control(EXPOSURE, 100_000)),
            ("1s", lambda: app.set_control(EXPOSURE, 1_000_000)),
        ):
            if imgui.button(label, ImVec2(button_w, 0)):
                action()
            if label != "1s":
                imgui.same_line()

        imgui.set_next_item_width(-1)
        changed, exact = imgui.input_double(
            "##exposure_us", float(app.control_value(EXPOSURE, 20_000)), 0.0, 0.0, "%.0f us"
        )
        if changed:
            app.set_control(EXPOSURE, int(round(exact)))

    # -- gain -------------------------------------------------------------
    rng = app.control_range(GAIN)
    if rng is not None:
        gain = app.control_value(GAIN, 0)
        imgui.text(f"Gain   ({gain / 10.0:.1f} dB)")
        imgui.set_next_item_width(-1)
        changed, gain = imgui.slider_int("##gain", gain, rng.min_value, rng.max_value)
        if changed:
            app.set_control(GAIN, gain)
        spacing = imgui.get_style().item_spacing.x
        button_w = (imgui.get_content_region_avail().x - 3 * spacing) / 4.0
        for label, value in (("min", rng.min_value), ("default", rng.default_value),
                             ("-5", None), ("+5", None)):
            if imgui.button(label, ImVec2(button_w, 0)):
                if value is None:
                    app.nudge_gain(GAIN_STEP if label == "+5" else -GAIN_STEP)
                else:
                    app.set_control(GAIN, value)
            if label != "+5":
                imgui.same_line()

    imgui.spacing()
    paused = device.paused
    if imgui.button("Resume  (Space)" if paused else "Pause  (Space)", ImVec2(-1, 0)):
        app.toggle_pause()

    stats = device.stats()
    if stats.exposing_since is not None:
        total = max(app.control_value(EXPOSURE, 1) / 1e6, 1e-6)
        elapsed = time.monotonic() - stats.exposing_since
        imgui.progress_bar(
            float(np.clip(elapsed / total, 0.0, 1.0)),
            ImVec2(-1, 0),
            f"exposing {elapsed:.1f} / {total:.1f} s",
        )

    if imgui.tree_node_ex("Advanced", 0):
        for name, label in ((OFFSET, "Offset"), (BANDWIDTH, "USB bandwidth")):
            rng = app.control_range(name)
            if rng is None:
                continue
            value = app.control_value(name, rng.default_value)
            imgui.set_next_item_width(-1)
            changed, value = imgui.slider_int(
                f"##{name}", value, rng.min_value, rng.max_value, f"{label}: %d"
            )
            if changed:
                app.set_control(name, value)
        if app.control_range(HIGH_SPEED) is not None:
            changed, high = imgui.checkbox(
                "High speed mode", bool(app.control_value(HIGH_SPEED, 0))
            )
            if changed:
                app.set_control(HIGH_SPEED, 1 if high else 0)
        threshold = getattr(device, "snap_threshold_us", 0) / 1e6
        imgui.text_colored(
            GREY, f"video mode below {threshold:g} s exposure, single-shot above"
        )
        imgui.tree_pop()
    imgui.spacing()


def _exposure_format(ms: float) -> str:
    if ms < 1.0:
        return "%.3f ms"
    if ms < 100.0:
        return "%.2f ms"
    return "%.0f ms"


def _display_section(app: App) -> None:
    if not imgui.collapsing_header("Display stretch", _OPEN):
        return

    stretch = app.stretch
    full_scale = float(app.stats.full_scale or 65535)
    dirty = False

    changed, stretch.auto = imgui.checkbox("Auto stretch  (A)", stretch.auto)
    dirty |= changed

    imgui.begin_disabled(not stretch.auto)
    imgui.set_next_item_width(-1)
    changed, stretch.lo_percentile = imgui.slider_float(
        "##lo_pct", stretch.lo_percentile, 0.0, 10.0, "low percentile %.2f %%"
    )
    dirty |= changed
    imgui.set_next_item_width(-1)
    changed, stretch.hi_percentile = imgui.slider_float(
        "##hi_pct", stretch.hi_percentile, 90.0, 100.0, "high percentile %.3f %%"
    )
    dirty |= changed
    imgui.end_disabled()

    imgui.begin_disabled(stretch.auto)
    imgui.set_next_item_width(-1)
    changed, black = imgui.slider_float(
        "##black", stretch.black * full_scale, 0.0, full_scale, "black %.0f ADU"
    )
    if changed:
        stretch.black = black / full_scale
        dirty = True
    imgui.set_next_item_width(-1)
    changed, white = imgui.slider_float(
        "##white", stretch.white * full_scale, 0.0, full_scale, "white %.0f ADU"
    )
    if changed:
        stretch.white = white / full_scale
        dirty = True
    imgui.end_disabled()

    imgui.set_next_item_width(-1)
    changed, stretch.midtone = imgui.slider_float(
        "##midtone", stretch.midtone, 0.005, 0.995, "midtone %.3f", imgui.SliderFlags_.logarithmic
    )
    dirty |= changed
    if imgui.button("Linear (0.5)", ImVec2(-1, 0)):
        stretch.midtone = 0.5
        dirty = True

    _histogram(app)
    if dirty:
        app.redraw_texture()
    imgui.spacing()


def _histogram(app: App) -> None:
    stats = app.stats
    hist = stats.hist
    if hist is None or not len(hist):
        return
    values = np.log10(hist + 1.0).astype(np.float32)
    imgui.plot_lines(
        "##hist",
        values,
        overlay_text=f"histogram of the crop (log), {stats.max_adu} ADU peak",
        scale_min=0.0,
        scale_max=float(values.max()) if values.size else 1.0,
        graph_size=ImVec2(-1, 70),
    )
    # Mark where the stretch endpoints sit on the histogram.
    draw_list = imgui.get_window_draw_list()
    x0, y0 = imgui.get_item_rect_min().x, imgui.get_item_rect_min().y
    x1, y1 = imgui.get_item_rect_max().x, imgui.get_item_rect_max().y
    for position, colour in ((app.stretch.black, 0xFF4488FF), (app.stretch.white, 0xFF66FF88)):
        x = x0 + float(np.clip(position, 0.0, 1.0)) * (x1 - x0)
        draw_list.add_line(ImVec2(x, y0), ImVec2(x, y1), colour, 1.0)


def _statistics_section(app: App) -> None:
    if not imgui.collapsing_header("Statistics", _OPEN):
        return

    stats = app.stats
    crop_w, crop_h = app.crop_size
    imgui.text_colored(GREY, f"measured in the crop, {crop_w} x {crop_h} px")
    imgui.text(f"min {stats.min_adu}   max {stats.max_adu}   mean {stats.mean_adu:.1f} ADU")
    saturated = stats.saturated_fraction * 100.0
    colour = RED if saturated > 0.05 else GREY
    imgui.text_colored(colour, f"saturated {saturated:.3f} %   (sampled 1/{stats.sample_step})")

    if app.camera is not None:
        cam_stats = app.camera.stats()
        imgui.text(f"camera {cam_stats.fps:5.2f} fps   mode {cam_stats.mode}")
        imgui.text(
            f"frames {cam_stats.frames}   dropped {cam_stats.dropped}   "
            f"timeouts {cam_stats.timeouts}"
        )
        if cam_stats.temperature_c is not None:
            imgui.text(f"sensor {cam_stats.temperature_c:.1f} C")
        if cam_stats.errors:
            imgui.text_colored(YELLOW, f"{cam_stats.errors} error(s): {cam_stats.last_error}")
    imgui.text_colored(
        GREY, f"UI {imgui.get_io().framerate:5.1f} fps   frame prep {app.display_ms:.1f} ms"
    )
    imgui.spacing()


def _view_section(app: App) -> None:
    if not imgui.collapsing_header("View", 0):
        return
    changed, app.fit = imgui.checkbox("Fit to window  (F)", app.fit)
    imgui.set_next_item_width(-1)
    changed, zoom = imgui.slider_float(
        "##zoom", app.zoom, ZOOM_MIN, ZOOM_MAX, "zoom %.2fx", imgui.SliderFlags_.logarithmic
    )
    if changed:
        app.zoom, app.fit = zoom, False
    imgui.set_next_item_width(-1)
    _, app.settings.panel_width = imgui.slider_float(
        "##panel", app.settings.panel_width, 300.0, 700.0, "left panel %.0f px"
    )
    imgui.set_next_item_width(-1)
    _, app.settings.right_panel_width = imgui.slider_float(
        "##rpanel", app.settings.right_panel_width, 300.0, 700.0, "right panel %.0f px"
    )
    imgui.set_next_item_width(-1)
    changed, scale = imgui.slider_float(
        "##uiscale", app.settings.ui_scale, 0.8, 2.0, "UI scale %.2f"
    )
    if changed:
        app.settings.ui_scale = scale
        imgui.get_style().font_scale_main = scale
    changed, app.show_help = imgui.checkbox("Show shortcuts  (H)", app.show_help)
    imgui.spacing()


# ---------------------------------------------------------------------------
# Right panel: spectrum
# ---------------------------------------------------------------------------

def _spectrum_panel(app: App) -> None:
    _crop_section(app)
    _band_section(app)
    _band_settings_section(app)
    _band_curves_section(app)
    _shear_section(app)
    _capture_spectrum_section(app)
    _band_overlay_section(app)


def _crop_section(app: App) -> None:
    if not imgui.collapsing_header("Crop (region of interest)", _OPEN):
        return
    if app.frame is None:
        imgui.text_colored(GREY, "no frame yet")
        imgui.spacing()
        return

    frame_w, frame_h = app.frame.width, app.frame.height
    changed, app.show_full_frame = imgui.checkbox(
        "Show full frame  (C)", app.show_full_frame
    )
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "On: the whole frame, with the crop marked by draggable red lines.\n"
            "Off: only the crop is shown."
        )

    crop = app.crop
    labels = ("left x", "top y", "right x", "bottom y")
    limits = (frame_w, frame_h, frame_w, frame_h)
    for index, (label, limit) in enumerate(zip(labels, limits)):
        imgui.set_next_item_width(-1)
        edited, value = imgui.slider_int(
            f"##crop{index}", int(crop[index]), 0, int(limit), f"{label} %d"
        )
        if edited:
            crop[index] = value
            _normalise_crop(app, frame_w, frame_h, moved=index)

    width, height = app.crop_size
    imgui.text_colored(
        GREY,
        f"crop {width} x {height} px "
        f"({100.0 * width / max(frame_w, 1):.0f} % x {100.0 * height / max(frame_h, 1):.0f} %)",
    )
    if imgui.button("Reset to full frame", ImVec2(-1, 0)):
        app.reset_crop()
    imgui.push_text_wrap_pos(0.0)
    imgui.text_colored(GREY, "Every spectrum algorithm works inside the crop.")
    imgui.pop_text_wrap_pos()
    imgui.spacing()


def _normalise_crop(app: App, frame_w: int, frame_h: int, moved: int) -> None:
    """Keep the crop inside the frame and at least MIN_CROP px in each axis."""
    x0, y0, x1, y1 = (int(v) for v in app.crop)
    x0 = max(0, min(x0, frame_w - MIN_CROP))
    y0 = max(0, min(y0, frame_h - MIN_CROP))
    x1 = min(frame_w, max(x1, MIN_CROP))
    y1 = min(frame_h, max(y1, MIN_CROP))
    # The edge the user just moved wins; push the opposite one out of its way.
    if x1 - x0 < MIN_CROP:
        if moved == 0:
            x1 = min(frame_w, x0 + MIN_CROP)
        else:
            x0 = max(0, x1 - MIN_CROP)
    if y1 - y0 < MIN_CROP:
        if moved == 1:
            y1 = min(frame_h, y0 + MIN_CROP)
        else:
            y0 = max(0, y1 - MIN_CROP)
    app.crop = [x0, y0, x1, y1]


def _band_section(app: App) -> None:
    if not imgui.collapsing_header("Band angle (camera rotation)", _OPEN):
        return

    finder = app.band_finder
    if finder.running:
        imgui.progress_bar(finder.progress, ImVec2(-1, 0), f"scanning {finder.progress * 100:.0f} %")
        if imgui.button("Cancel", ImVec2(-1, 0)):
            app.cancel_band_search()
    else:
        imgui.begin_disabled(app.frame is None)
        if imgui.button("Find band   (pauses capture)", ImVec2(-1, 0)):
            app.start_band_search()
        imgui.end_disabled()

    result = app.band_result
    if result is None or not result.ok:
        imgui.text_colored(GREY, app.band_status or "not measured yet")
        imgui.spacing()
        return

    span = result.x_to - result.x_from
    imgui.text_colored(GREEN, f"angle  {result.angle_deg:+.4f} deg")
    imgui.text_colored(
        GREY,
        f"tan {result.tan_angle:+.5f}   rise over the {span:.0f} px crop: "
        f"{result.tan_angle * span:+.1f} px",
    )
    imgui.text(f"band centre y   {result.centre_y:8.2f}   (at x {result.reference_x:.0f})")
    imgui.text(f"band width      {result.fwhm_px:8.2f} px")
    imgui.text(f"edges y         {result.edge_lo_y:8.2f} .. {result.edge_hi_y:.2f}")
    if result.snr:
        imgui.text_colored(
            GREY,
            f"contrast/noise  {result.snr:.0f}   "
            f"{result.angles_evaluated} angles in {result.elapsed_s:.2f} s",
        )
    if result.warning:
        imgui.push_text_wrap_pos(0.0)
        imgui.text_colored(YELLOW, result.warning)
        imgui.pop_text_wrap_pos()

    if imgui.button("Copy numbers", ImVec2(-1, 0)):
        imgui.set_clipboard_text(
            f"angle_deg={result.angle_deg:.5f} tan={result.tan_angle:.6f} "
            f"centre_y={result.centre_y:.3f} width_px={result.fwhm_px:.3f} "
            f"edge_lo_y={result.edge_lo_y:.3f} edge_hi_y={result.edge_hi_y:.3f} "
            f"reference_x={result.reference_x:.1f} "
            f"x_from={result.x_from:.0f} x_to={result.x_to:.0f}"
        )
    imgui.spacing()


def _band_settings_section(app: App) -> None:
    if not imgui.collapsing_header("Search settings", 0):
        return
    params = app.band_params

    imgui.set_next_item_width(-1)
    _, params.angle_range_deg = imgui.slider_float(
        "##range", params.angle_range_deg, 0.5, 20.0, "scan range +-%.1f deg"
    )
    imgui.set_next_item_width(-1)
    _, params.angle_step_deg = imgui.slider_float(
        "##step", params.angle_step_deg, 0.001, 0.2, "angle step %.3f deg",
        imgui.SliderFlags_.logarithmic,
    )
    imgui.set_next_item_width(-1)
    _, params.lo_percentile = imgui.slider_float(
        "##lopct", params.lo_percentile, 0.0, 5.0, "low percentile %.2f %%"
    )
    imgui.set_next_item_width(-1)
    _, params.hi_percentile = imgui.slider_float(
        "##hipct", params.hi_percentile, 95.0, 100.0, "high percentile %.3f %%"
    )
    imgui.set_next_item_width(-1)
    _, params.smooth_window = imgui.slider_int(
        "##smooth", params.smooth_window, 1, 21, "width smoothing %d samples"
    )
    imgui.push_text_wrap_pos(0.0)
    imgui.text_colored(
        GREY,
        "Tilted scan lines, arithmetic mean along each line; the band width is "
        "taken at 25 % of the way from the low to the high percentile, crossings "
        "found from both ends of the projection; the width curve is "
        "Gaussian-filtered and its minimum taken.",
    )
    imgui.pop_text_wrap_pos()
    imgui.spacing()


def _band_curves_section(app: App) -> None:
    if not imgui.collapsing_header("Curves", _OPEN):
        return
    result = app.band_result
    if result is None or result.angles_deg.size < 2:
        imgui.text_colored(GREY, "run a search to see the curves")
        imgui.spacing()
        return

    imgui.plot_lines(
        "##fwhm_raw",
        result.fwhm_curve_raw,
        overlay_text="width vs angle (raw)",
        graph_size=ImVec2(-1, 60),
    )
    imgui.plot_lines(
        "##fwhm",
        result.fwhm_curve,
        overlay_text="width vs angle (filtered)",
        graph_size=ImVec2(-1, 60),
    )
    imgui.text_colored(
        GREY,
        f"{result.angles_deg[0]:+.2f} .. {result.angles_deg[-1]:+.2f} deg, "
        f"minimum at {result.angle_deg:+.4f}",
    )
    swing = result.fwhm_max - result.fwhm_min
    imgui.text_colored(
        GREY if swing > 0.5 else YELLOW,
        f"curve spans {result.fwhm_min:.2f} .. {result.fwhm_max:.2f} px "
        f"(swing {swing:.2f} px)",
    )
    if result.profile.size > 2:
        imgui.plot_lines(
            "##profile",
            result.profile,
            overlay_text="projection at that angle",
            graph_size=ImVec2(-1, 70),
        )
        imgui.text_colored(GREY, f"scan line 0 .. {result.profile.size - 1}")
    imgui.spacing()


def _shear_section(app: App) -> None:
    if not imgui.collapsing_header("Shear: spectrum basis", _OPEN):
        return

    finder = app.shear_finder
    if finder.running:
        imgui.progress_bar(
            finder.progress, ImVec2(-1, 0), f"scanning {finder.progress * 100:.0f} %"
        )
        if imgui.button("Cancel", ImVec2(-1, 0)):
            app.cancel_shear_search()
    else:
        ready = app.band_result is not None and app.band_result.ok and app.frame is not None
        imgui.begin_disabled(not ready)
        if imgui.button("Find shear   (pauses capture)", ImVec2(-1, 0)):
            app.start_shear_search()
        imgui.end_disabled()

    imgui.set_next_item_width(-1)
    _, app.shear_params.blur_scale = imgui.slider_float(
        "##blurscale", app.shear_params.blur_scale, 0.25, 6.0, "blur scale %.2f",
        imgui.SliderFlags_.logarithmic,
    )
    if imgui.is_item_hovered():
        imgui.set_tooltip("Sharpness = RMS of blur(1 x scale) - blur(4 x scale)")

    result = app.shear_result
    if result is None or not result.ok:
        imgui.text_colored(GREY, app.shear_status or "not measured yet")
        imgui.spacing()
        return

    imgui.text_colored(GREEN, f"lines tilt   {result.line_tilt_deg:+.3f} deg  (from Y)")
    imgui.text_colored(GREEN, f"shear        {result.shear_deg:+.3f} deg  (off perpendicular)")
    imgui.text_colored(GREY, f"du/dv {result.shear_tan:+.5f} px per px across the band")
    imgui.text("basis, frame coordinates (y down):")
    imgui.text(f"  X (wavelength) {result.axis_x[0]:+.6f} {result.axis_x[1]:+.6f}")
    imgui.text(f"  Y (lines)      {result.axis_y[0]:+.6f} {result.axis_y[1]:+.6f}")
    imgui.text_colored(
        GREY,
        f"summed {result.rows_used} rows x {result.columns_used} columns   "
        f"{result.angles_evaluated} angles in {result.elapsed_s:.2f} s",
    )

    if result.sharpness.size > 2:
        imgui.plot_lines(
            "##sharpness",
            result.sharpness,
            overlay_text="sharpness vs tilt",
            graph_size=ImVec2(-1, 70),
        )
        imgui.text_colored(
            GREY,
            f"{result.angles_deg[0]:+.1f} .. {result.angles_deg[-1]:+.1f} deg, "
            f"peak at {result.line_tilt_deg:+.3f}",
        )
        low, high = result.sharpness_min, result.sharpness_max
        ratio = high / low if low > 0 else float("inf")
        imgui.text_colored(GREY, f"curve spans {low:.3g} .. {high:.3g}  (x{ratio:.2f})")

    if imgui.button("Copy basis", ImVec2(-1, 0)):
        imgui.set_clipboard_text(
            f"band_angle_deg={result.band_angle_deg:.5f} "
            f"line_tilt_deg={result.line_tilt_deg:.5f} shear_deg={result.shear_deg:.5f} "
            f"shear_tan={result.shear_tan:.6f} "
            f"axis_x=({result.axis_x[0]:.6f},{result.axis_x[1]:.6f}) "
            f"axis_y=({result.axis_y[0]:.6f},{result.axis_y[1]:.6f})"
        )
    if imgui.button("Clear shear", ImVec2(-1, 0)):
        app.clear_shear()
    imgui.spacing()


def _capture_spectrum_section(app: App) -> None:
    if not imgui.collapsing_header("Spectrum", _OPEN):
        return
    ready = app.can_capture_spectrum
    imgui.begin_disabled(not ready)
    if imgui.button("Capture Spectra", ImVec2(-1, 0)):
        app.capture_spectrum()
    imgui.end_disabled()
    if not ready:
        imgui.push_text_wrap_pos(0.0)
        imgui.text_colored(GREY, "Needs the band angle and the shear measured first.")
        imgui.pop_text_wrap_pos()
    if app.show_spectrum:
        if imgui.button("Close spectrum window", ImVec2(-1, 0)):
            app.close_spectrum()
        imgui.text_colored(
            GREY, "Extracted from every new frame while the window is open."
        )
    if app.spectrum_status:
        imgui.text_colored(YELLOW, app.spectrum_status)
    imgui.spacing()


def _band_overlay_section(app: App) -> None:
    if not imgui.collapsing_header("Overlay", _OPEN):
        return
    _, app.show_band_overlay = imgui.checkbox("Show band lines  (B)", app.show_band_overlay)
    if imgui.button("Clear calibration", ImVec2(-1, 0)):
        app.clear_band()
    imgui.push_text_wrap_pos(0.0)
    imgui.text_colored(GREY, "Lines are drawn over the image; the frame data is untouched.")
    imgui.pop_text_wrap_pos()
    imgui.spacing()


# ---------------------------------------------------------------------------
# Spectrum window, under the preview
# ---------------------------------------------------------------------------

SPLITTER_H = 6.0


def _drag_handle(label: str, width: float, height: float = SPLITTER_H) -> float:
    """A thin draggable strip; returns how far the mouse moved vertically."""
    imgui.invisible_button(label, ImVec2(max(width, 1.0), height))
    hovered_or_active = imgui.is_item_hovered() or imgui.is_item_active()
    if hovered_or_active:
        imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ns)
        draw_list = imgui.get_window_draw_list()
        top_left, bottom_right = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        middle = 0.5 * (top_left.y + bottom_right.y)
        draw_list.add_line(
            ImVec2(top_left.x, middle), ImVec2(bottom_right.x, middle), GRIP_COLOUR, 2.0
        )
    return imgui.get_io().mouse_delta.y if imgui.is_item_active() else 0.0


def _spectrum_window(app: App) -> None:
    style = imgui.get_style()
    full_width = imgui.get_content_region_avail().x
    app.spectrum_height -= _drag_handle("##spectrum_splitter", full_width)

    spectrum = app.spectrum
    if spectrum is None or not spectrum.ok or not app.spectrum_texture.valid:
        imgui.text_colored(GREY, app.spectrum_status or "no spectrum yet")
        return

    avail = imgui.get_content_region_avail()
    strip_h = float(np.clip(avail.x / max(app.strip_ratio, 2.0), 16.0, max(16.0, avail.y - 90.0)))
    imgui.image(app.spectrum_texture.ref, ImVec2(avail.x, strip_h))
    moved = _drag_handle("##strip_splitter", avail.x)
    if moved:
        app.strip_ratio = float(np.clip(avail.x / max(strip_h + moved, 8.0), 3.0, 80.0))

    imgui.text_colored(
        GREY,
        f"{spectrum.length} samples   averaged over {spectrum.rows_averaged} rows   "
        f"band {spectrum.band_angle_deg:+.3f} deg, lines {spectrum.line_tilt_deg:+.3f} deg",
    )
    remaining = imgui.get_content_region_avail().y - style.item_spacing.y
    if remaining > 40.0:
        imgui.plot_lines("##spectrum_plot", spectrum.values, graph_size=ImVec2(-1, remaining))


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def _preview(app: App) -> None:
    style = imgui.get_style()
    avail = imgui.get_content_region_avail()
    status_h = imgui.get_frame_height() + style.item_spacing.y
    view = ImVec2(avail.x, max(64.0, avail.y - status_h))

    imgui.begin_child(
        "##view",
        view,
        0,
        imgui.WindowFlags_.horizontal_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse,
    )
    texture = app.texture
    if texture.valid:
        _image_area(app, view)
    else:
        message = "No image yet - pick a camera and press Connect"
        size = imgui.calc_text_size(message)
        imgui.set_cursor_pos(
            ImVec2(max(0.0, (view.x - size.x) * 0.5), max(0.0, (view.y - size.y) * 0.5))
        )
        imgui.text_colored(GREY, message)
    imgui.end_child()

    frame = app.frame
    parts = [f"{app.zoom * 100:.0f} %"]
    if frame is not None:
        parts.append(f"{frame.width}x{frame.height}")
        parts.append(f"frame {frame.index}")
        parts.append(f"{frame.exposure_us / 1000.0:g} ms  gain {frame.gain}")
    if app.cursor_text:
        parts.append(app.cursor_text)
    if app.camera is not None and app.camera.paused:
        parts.append("PAUSED")
    imgui.align_text_to_frame_padding()
    imgui.text_colored(GREY, "   |   ".join(parts))


CROP_GRAB_PX = 6.0  # how close to a crop line counts as grabbing it


def _image_area(app: App, view: ImVec2) -> None:
    io = imgui.get_io()
    texture = app.texture
    tex_w, tex_h = float(texture.width), float(texture.height)

    # The region of the frame on screen: the whole frame, or just the crop.
    if app.show_full_frame:
        vx0, vy0, vx1, vy1 = 0.0, 0.0, tex_w, tex_h
    else:
        cx0, cy0, cx1, cy1 = (float(v) for v in app.crop)
        vx0, vy0 = max(0.0, cx0), max(0.0, cy0)
        vx1, vy1 = min(tex_w, max(cx1, vx0 + 1.0)), min(tex_h, max(cy1, vy0 + 1.0))
    region_w, region_h = vx1 - vx0, vy1 - vy0

    if app.fit:
        app.zoom = float(
            np.clip(min(view.x / max(region_w, 1.0), view.y / max(region_h, 1.0)),
                    ZOOM_MIN, ZOOM_MAX)
        )
    zoom = app.zoom
    size = ImVec2(region_w * zoom, region_h * zoom)

    # Centre the image while it is smaller than the viewport.
    offset = ImVec2(max(0.0, (view.x - size.x) * 0.5), max(0.0, (view.y - size.y) * 0.5))
    if offset.x or offset.y:
        imgui.set_cursor_pos(ImVec2(imgui.get_cursor_pos_x() + offset.x,
                                    imgui.get_cursor_pos_y() + offset.y))

    origin = imgui.get_cursor_screen_pos()
    imgui.image(
        texture.ref,
        size,
        ImVec2(vx0 / tex_w, vy0 / tex_h),
        ImVec2(vx1 / tex_w, vy1 / tex_h),
    )

    def screen(x: float, y: float) -> ImVec2:
        """Image pixel -> screen position."""
        return ImVec2(origin.x + (x - vx0) * zoom, origin.y + (y - vy0) * zoom)

    def image_xy(pos: ImVec2):
        return (pos.x - origin.x) / zoom + vx0, (pos.y - origin.y) / zoom + vy0

    # An invisible button on top of the image gives us hover + drag handling.
    imgui.set_cursor_screen_pos(origin)
    imgui.invisible_button("##image", size)
    hovered = imgui.is_item_hovered()
    active = imgui.is_item_active()

    grab = _crop_edge_under_cursor(app, io.mouse_pos, screen) if app.show_full_frame else None
    if app.show_full_frame and (grab or app.crop_drag):
        imgui.set_mouse_cursor(
            imgui.MouseCursor_.resize_ew
            if (app.crop_drag or grab) in ("left", "right")
            else imgui.MouseCursor_.resize_ns
        )

    if hovered:
        x, y = image_xy(io.mouse_pos)
        value = app.pixel_value(int(x), int(y))
        app.cursor_text = f"x {int(x)}  y {int(y)}  =  {value} ADU" if value is not None else ""
        if io.mouse_wheel:
            _zoom_at(app, io.mouse_pos, origin, ZOOM_STEP ** io.mouse_wheel)
    else:
        app.cursor_text = ""

    if active and app.crop_drag is None and grab is not None:
        app.crop_drag = grab
    if not imgui.is_mouse_down(0):
        app.crop_drag = None

    if app.crop_drag is not None and app.frame is not None:
        x, y = image_xy(io.mouse_pos)
        index = {"left": 0, "top": 1, "right": 2, "bottom": 3}[app.crop_drag]
        app.crop[index] = int(round(x if index in (0, 2) else y))
        _normalise_crop(app, app.frame.width, app.frame.height, moved=index)
    elif active:  # left button held on the image: pan
        imgui.set_scroll_x(imgui.get_scroll_x() - io.mouse_delta.x)
        imgui.set_scroll_y(imgui.get_scroll_y() - io.mouse_delta.y)

    pan_x, pan_y = app.pan_request
    if pan_x or pan_y:
        imgui.set_scroll_x(imgui.get_scroll_x() + pan_x)
        imgui.set_scroll_y(imgui.get_scroll_y() + pan_y)

    if app.show_full_frame:
        _draw_crop_lines(app, screen, grab)
    if app.show_band_overlay:
        _draw_band_overlay(app, screen)


def _crop_edge_under_cursor(app: App, mouse: ImVec2, screen) -> Optional[str]:
    """Which crop edge the cursor is close enough to grab, if any."""
    if app.frame is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in app.crop)
    top_left = screen(x0, y0)
    bottom_right = screen(x1, y1)
    left, top = top_left.x, top_left.y
    right, bottom = bottom_right.x, bottom_right.y
    inside_y = min(top, bottom) - CROP_GRAB_PX <= mouse.y <= max(top, bottom) + CROP_GRAB_PX
    inside_x = min(left, right) - CROP_GRAB_PX <= mouse.x <= max(left, right) + CROP_GRAB_PX
    candidates = []
    if inside_y:
        candidates.append(("left", abs(mouse.x - left)))
        candidates.append(("right", abs(mouse.x - right)))
    if inside_x:
        candidates.append(("top", abs(mouse.y - top)))
        candidates.append(("bottom", abs(mouse.y - bottom)))
    candidates = [(name, distance) for name, distance in candidates if distance <= CROP_GRAB_PX]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1])[0]


def _draw_crop_lines(app: App, screen, grab: Optional[str]) -> None:
    """The four draggable red crop lines, plus dimming outside them."""
    if app.frame is None:
        return
    draw_list = imgui.get_window_draw_list()
    x0, y0, x1, y1 = (float(v) for v in app.crop)
    top_left = screen(x0, y0)
    bottom_right = screen(x1, y1)
    frame_tl = screen(0.0, 0.0)
    frame_br = screen(float(app.frame.width), float(app.frame.height))

    # Dim everything outside the crop so the region of interest is obvious.
    for rect in (
        (frame_tl.x, frame_tl.y, frame_br.x, top_left.y),
        (frame_tl.x, bottom_right.y, frame_br.x, frame_br.y),
        (frame_tl.x, top_left.y, top_left.x, bottom_right.y),
        (bottom_right.x, top_left.y, frame_br.x, bottom_right.y),
    ):
        if rect[2] > rect[0] and rect[3] > rect[1]:
            draw_list.add_rect_filled(
                ImVec2(rect[0], rect[1]), ImVec2(rect[2], rect[3]), CROP_DIM_COLOUR
            )

    active = app.crop_drag or grab
    for name, (a, b) in (
        ("left", ((top_left.x, top_left.y), (top_left.x, bottom_right.y))),
        ("right", ((bottom_right.x, top_left.y), (bottom_right.x, bottom_right.y))),
        ("top", ((top_left.x, top_left.y), (bottom_right.x, top_left.y))),
        ("bottom", ((top_left.x, bottom_right.y), (bottom_right.x, bottom_right.y))),
    ):
        highlighted = name == active
        draw_list.add_line(
            ImVec2(*a),
            ImVec2(*b),
            CROP_ACTIVE_COLOUR if highlighted else CROP_COLOUR,
            2.5 if highlighted else 1.5,
        )


def _dashed_line(
    draw_list, start: ImVec2, end: ImVec2, colour: int,
    thickness: float = 1.0, dash: float = 6.0, gap: float = 5.0,
) -> None:
    """A dashed line in screen space, so the pattern does not change with zoom."""
    dx, dy = end.x - start.x, end.y - start.y
    length = math.hypot(dx, dy)
    if length < 1.0:
        return
    ux, uy = dx / length, dy / length
    walked = 0.0
    while walked < length:
        stop = min(walked + dash, length)
        draw_list.add_line(
            ImVec2(start.x + ux * walked, start.y + uy * walked),
            ImVec2(start.x + ux * stop, start.y + uy * stop),
            colour,
            thickness,
        )
        walked = stop + gap


def _draw_shear_overlay(app: App, screen, draw_list) -> None:
    """Dashed lines along the spectral lines, spaced across the spectrum.

    Dashed on purpose: it shows what the shear search settled on without
    covering up the spectrum underneath.
    """
    band, shear = app.band_result, app.shear_result
    if band is None or not band.ok or shear is None or not shear.ok:
        return
    half = 0.5 * band.fwhm_px * 1.15  # a little past the band edges, to be visible
    lean = math.tan(math.radians(shear.line_tilt_deg))
    for index in range(1, SHEAR_LINE_COUNT + 1):
        fraction = index / (SHEAR_LINE_COUNT + 1)
        x = band.x_from + fraction * (band.x_to - band.x_from)
        centre = band.centre_y + (x - band.reference_x) * band.tan_angle
        _dashed_line(
            draw_list,
            screen(x - half * lean, centre - half),
            screen(x + half * lean, centre + half),
            SHEAR_LINE_COLOUR,
        )


def _draw_band_overlay(app: App, screen) -> None:
    """Band edges drawn on top of the image - never baked into the texture."""
    result = app.band_result
    if result is None or not result.ok:
        return
    draw_list = imgui.get_window_draw_list()

    for (x0, y0), (x1, y1) in result.edge_points():
        draw_list.add_line(screen(x0, y0), screen(x1, y1), BAND_EDGE_COLOUR, 1.5)
    (cx0, cy0), (cx1, cy1) = result.centre_points()
    draw_list.add_line(screen(cx0, cy0), screen(cx1, cy1), BAND_CENTRE_COLOUR, 1.0)
    _draw_shear_overlay(app, screen, draw_list)

    label_x = result.x_from + 0.02 * (result.x_to - result.x_from)
    label_y = result.edge_lo_y + (label_x - result.reference_x) * result.tan_angle
    anchor = screen(label_x, label_y)
    draw_list.add_text(
        ImVec2(anchor.x + 4.0, anchor.y - imgui.get_text_line_height() - 4.0),
        BAND_EDGE_COLOUR,
        f"{result.angle_deg:+.3f} deg   width {result.fwhm_px:.1f} px",
    )


def _zoom_at(app: App, mouse: ImVec2, origin: ImVec2, factor: float) -> None:
    """Zoom keeping the pixel under the cursor in place."""
    old_zoom = app.zoom
    new_zoom = float(np.clip(old_zoom * factor, ZOOM_MIN, ZOOM_MAX))
    if new_zoom == old_zoom:
        return
    app.zoom, app.fit = new_zoom, False
    # Image coordinate under the cursor, and where the child's content starts.
    image_x = (mouse.x - origin.x) / old_zoom
    image_y = (mouse.y - origin.y) / old_zoom
    content_x = origin.x + imgui.get_scroll_x()
    content_y = origin.y + imgui.get_scroll_y()
    imgui.set_scroll_x(max(0.0, image_x * new_zoom - (mouse.x - content_x)))
    imgui.set_scroll_y(max(0.0, image_y * new_zoom - (mouse.y - content_y)))


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

SHORTCUTS = (
    ("F", "fit image to window"),
    ("1", "zoom 100 %"),
    ("+ / -", "zoom in / out (repeats)"),
    ("arrows", "pan (repeats, Shift = faster)"),
    ("mouse wheel", "zoom at cursor"),
    ("left drag", "pan"),
    ("[ / ]", "exposure / 1.25 or x 1.25 (Shift = /2, x2)"),
    (", / .", "gain -5 / +5 (Shift = -25 / +25)"),
    ("Space", "pause / resume preview"),
    ("A", "toggle auto stretch"),
    ("B", "toggle the band overlay"),
    ("C", "show full frame / show crop only"),
    ("S", "save the current frame to captures/"),
    ("H", "show / hide this window"),
    ("Esc", "quit"),
)


def _help_window(app: App) -> None:
    imgui.set_next_window_size(ImVec2(430, 330), imgui.Cond_.first_use_ever)
    imgui.set_next_window_pos(
        ImVec2(imgui.get_io().display_size.x * 0.5, imgui.get_io().display_size.y * 0.5),
        imgui.Cond_.first_use_ever,
        ImVec2(0.5, 0.5),
    )
    expanded, still_open = imgui.begin("Shortcuts", True)
    if expanded:
        for keys, description in SHORTCUTS:
            imgui.text_colored(YELLOW, f"{keys:>12}")
            imgui.same_line()
            imgui.text(f"  {description}")
    imgui.end()
    app.show_help = bool(still_open)


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------


def handle_keys(app: App) -> None:
    io = imgui.get_io()
    app.pan_request = [0.0, 0.0]  # cleared every frame, even while typing
    if io.want_text_input:
        return
    keys = imgui.Key
    shift = io.key_shift

    if imgui.is_key_pressed(keys.escape, False):
        app.running = False
    if imgui.is_key_pressed(keys.h, False):
        app.show_help = not app.show_help
    if imgui.is_key_pressed(keys.f, False):
        app.fit = True
    if imgui.is_key_pressed(keys._1, False):
        app.fit, app.zoom = False, 1.0
    if imgui.is_key_pressed(keys.a, False):
        app.stretch.auto = not app.stretch.auto
        app.redraw_texture()
    if imgui.is_key_pressed(keys.space, False):
        app.toggle_pause()
    if imgui.is_key_pressed(keys.b, False):
        app.show_band_overlay = not app.show_band_overlay
    if imgui.is_key_pressed(keys.c, False):
        app.show_full_frame = not app.show_full_frame
    if imgui.is_key_pressed(keys.s, False):
        app.save_frame()

    if imgui.is_key_pressed(keys.equal, True) or imgui.is_key_pressed(keys.keypad_add, True):
        app.zoom = float(np.clip(app.zoom * ZOOM_STEP, ZOOM_MIN, ZOOM_MAX))
        app.fit = False
    if imgui.is_key_pressed(keys.minus, True) or imgui.is_key_pressed(keys.keypad_subtract, True):
        app.zoom = float(np.clip(app.zoom / ZOOM_STEP, ZOOM_MIN, ZOOM_MAX))
        app.fit = False

    step = EXPOSURE_STEP if not shift else 2.0
    if imgui.is_key_pressed(keys.right_bracket, True):
        app.scale_exposure(step)
    if imgui.is_key_pressed(keys.left_bracket, True):
        app.scale_exposure(1.0 / step)

    gain_step = GAIN_STEP * (5 if shift else 1)
    if imgui.is_key_pressed(keys.period, True):
        app.nudge_gain(gain_step)
    if imgui.is_key_pressed(keys.comma, True):
        app.nudge_gain(-gain_step)

    # Arrow keys pan the preview; ImGui's own scrolling is disabled there.
    pan = 40.0 * (5 if shift else 1)
    if imgui.is_key_pressed(keys.left_arrow, True):
        app.pan_request[0] -= pan
    if imgui.is_key_pressed(keys.right_arrow, True):
        app.pan_request[0] += pan
    if imgui.is_key_pressed(keys.up_arrow, True):
        app.pan_request[1] -= pan
    if imgui.is_key_pressed(keys.down_arrow, True):
        app.pan_request[1] += pan
