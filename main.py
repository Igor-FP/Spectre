#!/usr/bin/env python3
"""Spectre - live viewer for a ZWO ASI spectrograph camera.

Real-time 16-bit preview with exposure and gain control.

    python main.py                 # auto-connect to the first ASI camera
    python main.py --sim           # synthetic spectrum, no hardware needed
    python main.py --list          # list cameras and exit
    python main.py --raw8 --bin 2  # 8-bit, 2x2 binned (faster preview)
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys

import OpenGL.GL as gl
import pygame

from imgui_bundle import imgui

from spectre import asi_sdk, ui
from spectre.app import App, SIM_LABEL
from spectre.asi_sdk import ImgType
from spectre.imgui_backend import SpectreRenderer
from spectre.settings import Settings


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", action="store_true", help="use the built-in simulated camera")
    parser.add_argument(
        "--file", metavar="PATH", action="append", default=[],
        help="replay a saved frame (.fits/.npy); may be given more than once. "
             "Frames in captures/ are listed automatically.",
    )
    parser.add_argument("--list", action="store_true", help="list connected cameras and exit")
    parser.add_argument("--dll", metavar="PATH", help="path to ASICamera2.dll")
    parser.add_argument("--camera", type=int, metavar="N", help="camera index to connect to")
    parser.add_argument("--bin", type=int, default=1, choices=(1, 2, 3, 4), help="binning (default 1)")
    parser.add_argument("--raw8", action="store_true", help="capture 8-bit instead of 16-bit")
    parser.add_argument("--no-connect", action="store_true", help="start disconnected")
    parser.add_argument(
        "--keep-camera", action="store_true",
        help="adopt the exposure/gain already set in the camera instead of "
             "restoring the ones from settings.json",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="run the band-angle search as soon as the first frame arrives",
    )
    parser.add_argument(
        "--screenshot", metavar="PATH.png", help="save a PNG of the window and exit (debugging)"
    )
    parser.add_argument(
        "--screenshot-after", type=int, default=90, metavar="N",
        help="frames to render before the screenshot (default 90)",
    )
    return parser.parse_args(argv)


def list_cameras(dll_path) -> int:
    try:
        path = asi_sdk.load(dll_path)
    except OSError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"SDK {asi_sdk.get_sdk_version()}  ({path})")
    cameras = asi_sdk.list_cameras()
    if not cameras:
        print("no ASI cameras found")
    for info in cameras:
        print(
            f"[{info.camera_id}] {info.name}  {info.max_width}x{info.max_height}  "
            f"{info.bit_depth}-bit  pixel {info.pixel_size_um} um  "
            f"{'colour' if info.is_color else 'mono'}  "
            f"formats {[f.name for f in info.supported_formats]}  "
            f"bins {list(info.supported_bins)}"
        )
    return 0


def _enable_dpi_awareness() -> None:
    """Sharp text on scaled Windows desktops."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # system DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def create_window(settings: Settings):
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    # Two 400 px panels leave little room for the image, so use most of the
    # desktop unless the user has resized the window before.
    desktop = pygame.display.Info()
    size = (
        min(settings.window_width, max(1024, desktop.current_w - 100)),
        min(settings.window_height, max(700, desktop.current_h - 120)),
    )
    flags = pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE
    try:
        pygame.display.set_mode(size, flags, vsync=1)
    except pygame.error:
        pygame.display.set_mode(size, flags)
    pygame.display.set_caption("Spectre - ASI spectrograph viewer")
    return size


def setup_imgui(settings: Settings, size) -> SpectreRenderer:
    imgui.create_context()
    io = imgui.get_io()
    try:
        io.set_ini_filename(None)  # fixed layout; nothing worth persisting
    except TypeError:
        io.set_ini_filename("")
    io.key_repeat_delay = 0.35
    io.key_repeat_rate = 0.04
    imgui.get_style().font_scale_main = settings.ui_scale
    renderer = SpectreRenderer()
    io.display_size = size
    return renderer


def save_screenshot(path: str) -> None:
    """Grab the rendered back buffer (used by --screenshot for debugging)."""
    import numpy as np
    from PIL import Image

    width, height = pygame.display.get_surface().get_size()
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    raw = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)[::-1]
    Image.fromarray(pixels).save(path)
    print(f"screenshot saved to {path}")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list:
        return list_cameras(args.dll)

    settings = Settings.load()
    _enable_dpi_awareness()
    pygame.init()
    size = create_window(settings)
    renderer = setup_imgui(settings, size)
    # Key repeat for text fields; ImGui does its own repeat for shortcuts.
    pygame.key.set_repeat(400, 33)

    app = App(
        settings,
        dll_path=args.dll,
        img_type=ImgType.RAW8 if args.raw8 else ImgType.RAW16,
        binning=args.bin,
        keep_camera_settings=args.keep_camera,
        extra_files=[os.path.abspath(path) for path in args.file],
    )
    _select_startup_camera(app, args)
    if not args.no_connect and app.selected_entry is not None:
        app.connect()
        if app.connect_error:
            print(app.connect_error, file=sys.stderr)

    clock = pygame.time.Clock()
    rendered = 0
    calibration_requested = False
    try:
        while app.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    app.running = False
                elif event.type == pygame.VIDEORESIZE:
                    settings.window_width, settings.window_height = event.w, event.h
                renderer.process_event(event)
            renderer.process_inputs()

            app.poll_camera()  # GL upload happens outside the ImGui frame
            app.poll_band_search()
            app.poll_shear_search()

            if args.calibrate and not calibration_requested and app.frame is not None:
                app.start_band_search()
                calibration_requested = True

            imgui.new_frame()
            ui.handle_keys(app)
            ui.draw(app)

            gl.glClearColor(0.09, 0.09, 0.11, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            imgui.render()
            renderer.render(imgui.get_draw_data())

            rendered += 1
            if args.screenshot and rendered >= args.screenshot_after:
                save_screenshot(args.screenshot)
                app.running = False

            pygame.display.flip()
            clock.tick(120)
    finally:
        app.shutdown()
        try:
            renderer.shutdown()
        except Exception:
            pass
        pygame.quit()
    return 0


def _select_startup_camera(app: App, args: argparse.Namespace) -> None:
    if args.sim:
        app.selected = next(
            (i for i, entry in enumerate(app.entries) if entry.simulator), 0
        )
        return
    if args.file:
        wanted = os.path.abspath(args.file[0])
        for index, entry in enumerate(app.entries):
            if entry.path == wanted:
                app.selected = index
                return
        print(f"{wanted}: not loadable", file=sys.stderr)
        return
    if args.camera is not None:
        for index, entry in enumerate(app.entries):
            if entry.info is not None and entry.info.camera_id == args.camera:
                app.selected = index
                return
        print(f"camera #{args.camera} not found", file=sys.stderr)
        return
    # The simulator is only used when asked for, even if it was used last time.
    entry = app.selected_entry
    if entry is not None and entry.simulator:
        other = next((i for i, e in enumerate(app.entries) if not e.simulator), None)
        if other is not None:
            app.selected = other


if __name__ == "__main__":
    sys.exit(main())
