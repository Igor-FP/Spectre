# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

**Read `CLAUDE.local.md` first.** It holds additional instructions that take
priority over this file. It is not tracked by git.

## Working rules - read first

* **No changes to code or to a spec without asking.** Ideas for improvement are
  welcome, but as words: describe the idea, wait for explicit consent, only then
  implement. Do not "improve" a specified algorithm on the way past it, do not
  add guards, options or heuristics that were not asked for.
* **The spec belongs to the project owner.** When an algorithm is described,
  implement exactly that, including the details that look redundant - they are
  usually there for a reason that comes from the optics of the instrument. If a
  measurement contradicts the spec, report the measurement and the numbers, then
  wait.
* **Do not touch the real camera on your own initiative.** The spectrograph is
  physical hardware that is often half-assembled. Verify with `--sim`, or offline
  on the saved frames in `captures/`. Ask before any run that opens the camera,
  and say which command it would be.
* **Do not touch `settings.json`.** It is the local live state (exposure, gain,
  crop, calibration). If a test run needs different settings, copy the file
  aside and put it back afterwards.
* **Never commit personal data, credentials or tokens** - in the files, in the
  commit message or in the commit identity - without explicit permission.
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
* `docs/TZ_Wavelength.md` - **spec** for the wavelength calibration: reference
  solar spectrum, anchor points, X -> wavelength.
* `docs/KnowledgeBase.md` - **accumulated facts and traps**, append-only. Read
  before touching the geometry code: it lists the variants that were tried on
  real frames and why they failed.
* `README.md` - user-facing description, in Russian.

## What this is

The control program for a simple home-made spectrograph: a diffraction grating,
whatever tube it is built into, and one of the ZWO monochrome cameras. The optical
layout does not matter and nothing in the code is model-specific.

**What the program is for:** shooting spectra and calibrating them, comparing
spectra against each other, plotting them, and saving them as 1-D FITS with the
calibration in the header.

Two of the calibration steps - the band angle and the shear - exist for exactly
one purpose: to produce the **basis of the spectrum coordinate system**, two
vectors, X along the wavelength axis and Y along the spectral lines, neither
perpendicular to the other (the slit and the grating are not exactly aligned) nor
to the frame edges (the camera is not screwed on perfectly). That basis is what
the spectrum is extracted along. The step after it, not implemented: identifying
known lines to get the zero point and the scale along X, i.e. the wavelength
calibration itself.

Windows 10, Python 3.11, pygame + Dear ImGui (`imgui-bundle`) over OpenGL 3.3.
Developed against an **ASI290MM** (mono, 1936x1096, 12-bit ADC).

## File map

```
main.py                  window, event loop, CLI
spectre/asi_sdk.py       ctypes binding of ASICamera2.dll (from ASICamera2.h, SDK 1.21)
spectre/camera.py        grabber thread; AsiCamera (video/snap), FileCamera, SimulatedCamera
spectre/display.py       display stretch (LUT), frame statistics, GL texture
spectre/imgui_backend.py pygame backend for ImGui with the full keyboard mapped
spectre/calib.py         ALL geometry: band angle, shear, spectrum extraction
spectre/reference.py     reference solar spectrum: blur, resample into our pixels
spectre/wavelength.py    anchor points and the X -> wavelength polynomial
spectre/frameio.py       saving and loading frames (FITS + .npy)
spectre/app.py           application state, connection, frame pipeline
spectre/ui.py            ImGui panels, overlays, keyboard
spectre/settings.py      settings.json
tools/probe_camera.py    camera diagnostics from the console
tools/fetch_reference.py downloads data/solar_reference.csv from BASS2000
data/solar_reference.csv reference solar spectrum, 300-1000 nm at 0.1 nm
captures/                frames saved by the S key / Save frame
```

## Run

```
main.bat                       # local launcher
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
the real one. Note: a Linux environment variable does **not** reach a Windows process
launched from WSL unless it is listed in `WSLENV`; backing up `settings.json` is
the reliable way.

## Hardware facts

* **RAW16 from an ASI290MM is 12-bit data shifted left by 4**: values are
  multiples of 16, full scale is 65535, so there are 4096 real levels.
  `tools/probe_camera.py` prints the step.
* Throughput is limited by the USB link, not the sensor: on a USB 2.0 port
  (`IsUSB3Host = False`) a full-frame RAW16 runs at ~8 fps (~35 MB/s), RAW8 at
  ~16 fps. Binning does not help - on a 290MM it is software binning, the full
  frame still crosses the bus.
* Control values (exposure, gain, offset, bandwidth) **live in the camera** and
  persist between sessions. The app must not overwrite them silently.
* `ASICamera2.dll` search order is in `spectre/asi_sdk.py::DEFAULT_DLL_PATHS`.
