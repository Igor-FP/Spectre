#!/usr/bin/env python3
"""Command-line check of the camera path, with no GUI involved.

    python tools/probe_camera.py                 # info + a few frames
    python tools/probe_camera.py --exposure 50 --gain 200 --frames 20
    python tools/probe_camera.py --sim

Prints per-frame statistics so you can tell whether RAW16 data really comes
back 16-bit-scaled (an ASI290MM has a 12-bit ADC and the SDK shifts the value
left by 4, so raw values are multiples of 16).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectre import asi_sdk, camera as cam  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", action="store_true", help="use the simulated camera")
    parser.add_argument("--dll", help="path to ASICamera2.dll")
    parser.add_argument("--camera", type=int, default=0, help="camera id (default 0)")
    parser.add_argument("--exposure", type=float, default=20.0, help="exposure in ms")
    parser.add_argument("--gain", type=int, default=200)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--bin", type=int, default=1)
    parser.add_argument("--raw8", action="store_true")
    parser.add_argument("--save", metavar="PATH.npy", help="save the last frame as .npy")
    args = parser.parse_args(argv)

    img_type = asi_sdk.ImgType.RAW8 if args.raw8 else asi_sdk.ImgType.RAW16

    if args.sim:
        device = cam.SimulatedCamera()
    else:
        path = asi_sdk.load(args.dll)
        print(f"SDK {asi_sdk.get_sdk_version()}  ({path})")
        infos = asi_sdk.list_cameras()
        if not infos:
            print("no ASI cameras found", file=sys.stderr)
            return 1
        info = next((i for i in infos if i.camera_id == args.camera), infos[0])
        print(
            f"camera [{info.camera_id}] {info.name}  {info.max_width}x{info.max_height}  "
            f"{info.bit_depth}-bit  e/ADU {info.elec_per_adu:.4f}  "
            f"formats {[f.name for f in info.supported_formats]}"
        )
        device = cam.AsiCamera(info, img_type=img_type, binning=args.bin)
        print(f"serial {device.serial}")
        for name, rng in device.controls.items():
            print(
                f"  {name:<11} {rng.min_value} .. {rng.max_value} "
                f"(default {rng.default_value}, now {device.control_value(name)})"
            )

    device.set_control(cam.EXPOSURE, int(args.exposure * 1000))
    device.set_control(cam.GAIN, args.gain)
    device.start()
    print(f"\ngrabbing {args.frames} frame(s) at {args.exposure} ms, gain {args.gain} ...")

    last = None
    received = 0
    deadline = time.monotonic() + 30 + args.frames * args.exposure / 1000.0 * 3
    try:
        while received < args.frames and time.monotonic() < deadline:
            frame = device.latest_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            received += 1
            last = frame
            data = frame.data
            nonzero = data[data > 0]
            step = int(np.gcd.reduce(np.unique(nonzero[:4096]))) if nonzero.size else 0
            print(
                f"  #{frame.index:<4} {frame.mode:<5} {data.dtype} {data.shape}  "
                f"min {data.min():<6} max {data.max():<6} mean {data.mean():8.1f}  "
                f"value step {step:<3} exp {frame.exposure_us} us gain {frame.gain}"
            )
        stats = device.stats()
        print(
            f"\n{stats.frames} frames, {stats.fps:.2f} fps, dropped {stats.dropped}, "
            f"timeouts {stats.timeouts}, errors {stats.errors} {stats.last_error}"
        )
        if stats.temperature_c is not None:
            print(f"sensor temperature {stats.temperature_c:.1f} C")
        if last is not None and args.save:
            np.save(args.save, last.data)
            print(f"saved {args.save}")
    finally:
        device.close()
    return 0 if received else 1


if __name__ == "__main__":
    sys.exit(main())
