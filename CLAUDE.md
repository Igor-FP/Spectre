# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Working rules - read first

* **No changes to code or to a spec without asking.** Ideas for improvement are
  welcome, but as words: describe the idea, wait for explicit consent, only then
  implement. Do not "improve" a specified algorithm on the way past it, do not
  add guards, options or heuristics that were not asked for.
* **The spec is Igor's.** When he describes an algorithm, implement exactly that,
  including the details that look redundant - they are usually there for a reason
  that comes from the optics of the instrument. If a measurement contradicts the
  spec, report the measurement and the numbers, then wait.
* **Do not touch the real camera on your own initiative.** The spectrograph is
  physical hardware that is often half-assembled. Verify with `--sim`, or offline
  on the saved frames in `captures/`. Ask before any run that opens the camera,
  and say which command it would be.
* **Do not touch `settings.json`.** It is Igor's live state (exposure, gain,
  crop, calibration). If a test run needs different settings, copy the file
  aside and put it back afterwards.
* Language: discussion in Russian; code comments and CLI documentation in
  English.
* **ASCII-only in `.py` and `.bat`.** Not in output, not in comments, not in
  docstrings, not in argparse help. Under Windows the stream codec comes from the
  locale (cp1251) and one non-ASCII byte breaks `>` / `2>` redirection and
  `--help` with `UnicodeEncodeError`. Use `-`, `->`, `...`, `~`, `x`, `deg`,
  `"`, `'`. Markdown files may use Russian.
* Use the existing API, do not invent a parallel one, do not duplicate a formula
  in two places. The band geometry lives in `spectre/calib.py` and nowhere else.

## Project references

* `docs/TZ_BandAngle.md` - **spec** for the band angle search (camera rotation).
* `docs/TZ_Shear.md` - **spec** for the shear search (slit tilt, the Y axis of
  the spectrum basis).
* `docs/TZ_Spectrum.md` - **spec** for extracting the 1-D spectrum.
* `docs/KnowledgeBase.md` - **accumulated facts and traps**, append-only. Read
  before touching the geometry code: it lists the variants that were tried on
  real frames and why they failed.
* `README.md` - user-facing description, in Russian.

## What this is

A live viewer and geometry-calibration tool for a spectrograph built around a
**ZWO ASI290MM** camera (mono, 1936x1096, 12-bit ADC, USB3 camera). Windows 10,
Python 3.11, pygame + Dear ImGui (`imgui-bundle`) over OpenGL 3.3.

The point of the calibration chain is the **basis of the spectrum coordinate
system**: two vectors, X along the wavelength axis and Y along the spectral
lines. They are not perpendicular to each other (the slit and the grating are
not exactly aligned) and not perpendicular to the frame edges (the camera is not
screwed on perfectly). Next stage, not implemented: identifying lines in the
solar spectrum to get the zero point and the scale along X.

## File map

```
main.py                  window, event loop, CLI
spectre/asi_sdk.py       ctypes binding of ASICamera2.dll (from ASICamera2.h, SDK 1.21)
spectre/camera.py        grabber thread; AsiCamera (video/snap), FileCamera, SimulatedCamera
spectre/display.py       display stretch (LUT), frame statistics, GL texture
spectre/imgui_backend.py pygame backend for ImGui with the full keyboard mapped
spectre/calib.py         ALL geometry: band angle, shear, spectrum extraction
spectre/frameio.py       saving and loading frames (FITS + .npy)
spectre/app.py           application state, connection, frame pipeline
spectre/ui.py            ImGui panels, overlays, keyboard
spectre/settings.py      settings.json
tools/probe_camera.py    camera diagnostics from the console
captures/                frames saved by the S key / Save frame
```

## Run

```
main.bat                       # what Igor uses
python main.py                 # connect to the first ASI camera
python main.py --sim           # synthetic spectrum, no hardware
python main.py --file captures\name.fits   # replay a saved frame
python main.py --list
python tools/probe_camera.py   # no GUI
```

`--keep-camera` adopts the exposure/gain already set in the camera instead of
restoring them from `settings.json`. `--screenshot PATH.png` renders N frames and
saves the window - the only way to see the UI without a display.

`SPECTRE_SETTINGS=<path>` moves the settings file, so a test run does not touch
Igor's. Note: a Linux environment variable does **not** reach a Windows process
launched from WSL unless it is listed in `WSLENV`; backing up `settings.json` is
the reliable way.

## Hardware facts

* **RAW16 from an ASI290MM is 12-bit data shifted left by 4**: values are
  multiples of 16, full scale is 65535, so there are 4096 real levels.
  `tools/probe_camera.py` prints the step.
* The camera is on a **USB 2.0 port** on this machine (`IsUSB3Host = False`):
  full-frame RAW16 runs at ~8.2 fps (~35 MB/s), RAW8 at ~16 fps. Binning does
  not help - on a 290MM it is software binning, the full frame still crosses the
  bus.
* Control values (exposure, gain, offset, bandwidth) **live in the camera** and
  persist between sessions. The app must not overwrite them silently.
* `ASICamera2.dll` search order is in `spectre/asi_sdk.py::DEFAULT_DLL_PATHS`;
  on this machine it is `C:\src\ZWOCameraSDK\lib\x64\ASICamera2.dll`.
