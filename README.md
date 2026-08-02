# Spectre

Spectre is the control program for a simple home-made spectrograph: a diffraction
grating, whatever tube and mount you put it in, and one of the ZWO monochrome
cameras. The optical layout does not matter - nothing in the program assumes a
particular design, sensor size or camera model.

The goal is to shoot spectra and calibrate them, compare spectra against each
other, plot them, and save them as 1-D FITS with the calibration written into the
header.

Implemented so far:

* **live view** of the camera - the raw 16-bit linear frame, in real time, with
  exposure, gain, offset and USB bandwidth control, and a display stretch that
  only affects what you see, never the data;
* **crop** (region of interest) with draggable lines: a spectrograph throws glare
  and the zero order across the sensor, and every measurement is done inside the
  crop only;
* **band geometry calibration** - the tilt of the spectrum band against the frame
  axes, i.e. how the camera is rotated on the spectrograph, plus the band's
  centre line and edges;
* **shear calibration** - the direction of the spectral lines, i.e. the tilt of
  the slit against the grating. Together with the band angle this gives the basis
  of the spectrum coordinate system: two vectors, X along the wavelength axis and
  Y along the spectral lines, neither perpendicular to the other nor to the frame
  edges;
* **1-D spectrum extraction** along that basis, shown as a strip and as a plot;
* **saving frames** as 16-bit FITS (plus `.npy`) with exposure, gain, sensor
  temperature, crop position and the calibration in the header, and **replaying**
  a saved frame instead of a camera, so everything can be worked on without
  hardware.

Not there yet: the wavelength calibration itself (identifying known lines to get
the zero point and the scale along X), comparing several spectra, and the 1-D
FITS export.

Windows 10, Python 3.11+, pygame + Dear ImGui over OpenGL 3.3.

```
main.bat                                   # same as python main.py
python main.py                             # first ASI camera found
python main.py --file captures\name.fits   # replay a saved frame
python main.py --sim                       # synthetic spectrum, no hardware
python main.py --list                      # list cameras and exit
python tools/probe_camera.py               # check the camera without a GUI
```

Algorithm specs and accumulated traps live in [docs/](docs/):
[TZ_BandAngle.md](docs/TZ_BandAngle.md), [TZ_Shear.md](docs/TZ_Shear.md),
[TZ_Spectrum.md](docs/TZ_Spectrum.md), [KnowledgeBase.md](docs/KnowledgeBase.md).

## Install

```
pip install -r requirements.txt
```

`ASICamera2.dll` from the ZWO ASI Camera SDK is needed. Searched in this order
(`spectre/asi_sdk.py`, `DEFAULT_DLL_PATHS`):

1. the `SPECTRE_ASI_DLL` environment variable;
2. `<project>/lib/ASICamera2.dll`;
3. `C:\src\ZWOCameraSDK\lib\x64\ASICamera2.dll`;
4. `C:\Program Files\ASIStudio\ASICamera2.dll`;
5. plain `ASICamera2.dll` somewhere on `PATH`.

Or point at it directly: `python main.py --dll "C:\path\ASICamera2.dll"`.

## Which Python ImGui binding

**`imgui-bundle`** (`pip install imgui-bundle`), and here is why:

* `pyimgui` (the `imgui` package) is frozen on Dear ImGui 1.82 (2021) and barely
  moves;
* `imgui-bundle` is alive, carries Dear ImGui 1.92, ImPlot (useful for spectrum
  plots later), docking, and - the point here - a ready pygame backend:
  `imgui_bundle.python_backends.pygame_backend`;
* `dearpygui` is not an ImGui binding but a separate framework with its own
  render loop; it does not combine with pygame.

With pygame, ImGui renders through OpenGL, so the window is created with
`OPENGL | DOUBLEBUF | RESIZABLE` and the camera frame is not blitted through
`pygame.surfarray` but uploaded as a GL texture inside `imgui.image()`.

One catch: the stock `PygameRenderer` only forwards navigation keys to ImGui, so
`imgui.is_key_pressed(imgui.Key.f)` never fires and application shortcuts are
impossible. `spectre/imgui_backend.py` has `SpectreRenderer`, which maps the whole
keyboard and still feeds printable characters to text fields. Key repeat is left
to ImGui itself (`io.key_repeat_delay` / `io.key_repeat_rate`): one-shot actions
use `is_key_pressed(key, repeat=False)`, repeating ones `repeat=True`.

## Controls

| Key | Action |
|---|---|
| `F` | fit the frame to the window |
| `1` | 100 % zoom |
| `+` / `-` | zoom in / out (repeats) |
| arrows | pan (repeats, Shift = faster) |
| mouse wheel | zoom at the cursor |
| left drag | pan |
| `[` / `]` | exposure / 1.25 or x 1.25 (Shift = /2, x2) |
| `,` / `.` | gain -5 / +5 (Shift = +-25) |
| `Space` | pause / resume |
| `A` | auto stretch on / off |
| `B` | band overlay on / off |
| `C` | whole frame / crop only |
| `S` | save the current frame into `captures/` |
| `H` | shortcuts window |
| `Esc` | quit |

Hovering the image shows the pixel coordinates and its value in ADU in the status
line.

## How it is put together

```
main.py                  window, event loop, CLI
spectre/asi_sdk.py       ctypes binding of ASICamera2.dll (from ASICamera2.h, SDK 1.21)
spectre/camera.py        grabber thread; AsiCamera (video/snap), FileCamera, SimulatedCamera
spectre/display.py       display stretch (LUT), frame statistics, GL texture
spectre/imgui_backend.py pygame backend for ImGui with the full keyboard mapped
spectre/calib.py         all the geometry: band angle, shear, spectrum extraction
spectre/frameio.py       saving and loading frames (FITS / .npy)
spectre/app.py           application state, connection, frame pipeline
spectre/ui.py            ImGui panels, overlays, keyboard
spectre/settings.py      settings.json
tools/probe_camera.py    camera diagnostics from the console
```

Capture runs in its own thread; the UI thread never calls the SDK. It queues
control changes (`camera.set_control`) and picks up the newest finished frame
(`camera.latest_frame`) - if the UI cannot keep up, spare frames are dropped
rather than queued.

Two capture modes switch automatically on the exposure (`SNAP_THRESHOLD_US`, 1 s):

* **video** (`ASIStartVideoCapture` / `ASIGetVideoData`) for short exposures;
* **snap** (`ASIStartExposure` / `ASIGetDataAfterExp`) for long ones. In snap mode
  a running exposure is **aborted** when a control changes, otherwise a 30 s
  exposure would make the interface look frozen. Progress is shown in the panel.

The camera data is never modified: the frame stays `uint16` on a linear scale, and
what reaches the screen is the result of a lookup table (black point / white point
/ midtone transfer function), uploaded as a single-byte GL texture (`GL_R8` plus a
swizzle to grey, with mipmaps so that thin spectral lines do not vanish when the
view is zoomed out).

## Saving frames, and working without a camera

Left panel, **Save frame** (or the `S` key): the current frame is written into
`captures/` next to the app, exactly as the camera delivered it - linear, 16-bit,
no stretch. By default the crop is saved, not the whole frame; there is a checkbox
for the whole frame. Formats:

* **FITS** with a header: `EXPTIME` / `EXPUS`, `GAIN`, `OFFSET`, `XBINNING`,
  `INSTRUME`, `CCD-TEMP`, `BITDEPTH`, `XPIXSZ`, the crop position on the sensor
  (`CROPX0`...`CROPY1`) and, if the calibration has been done, its result
  (`BANDANG`, `BANDCY`, `BANDW`, `BANDLO`, `BANDHI`, `BANDREFX`);
* **.npy** - the raw numpy array, no header.

Saved frames appear in the camera list as `file: name.fits` and are played back
instead of a camera (`FileCamera`): the image, exposure and gain come from the
file, the exposure controls are hidden, and everything else - crop, stretch,
calibration, spectrum - works as on a live frame. To switch: `Disconnect` ->
`Refresh list` -> pick the file -> `Connect`. From the command line:
`python main.py --file captures\name.fits`.

The simulator (`--sim`) sits last in the camera list and is never selected on its
own. It does not reproduce what a real spectrograph looks like (uniform band, full
frame width, no glare) and is only good for checking that the code runs and the UI
draws.

## Crop (region of interest)

Right panel, **Crop**. Every spectrum algorithm works strictly inside the crop;
results are converted back to full-frame coordinates.

* **Show full frame** (key `C`) - the whole frame with four draggable **red
  lines**, everything outside the crop dimmed;
* unchecked - only the crop is shown;
* the bounds can also be typed with the sliders, and there is a reset button;
* the crop is saved in `settings.json`.

Statistics, the histogram and the auto stretch are measured inside the crop too:
letting the glare and the zero order set the display levels, or the numbers you
read off the panel, is worse than useless.

## Calibration: the band angle

Right panel, **Band angle**. Implemented as specified in
[docs/TZ_BandAngle.md](docs/TZ_BandAngle.md):

1. capture is paused and the crop is scanned top to bottom, one scan line per row
   (1 px steps), but the lines do not run along X - they are tilted: line number
   `Y` starts at (0, Y) on the left edge of the crop with slope `dy/dx = tan(L)`;
2. for every angle `L` from -10 to +10 degrees in 0.01 degree steps, the
   **arithmetic mean** of all pixels along the line is taken, giving
   `array[L][Y]` - a 1-D projection of the image. The mean, not a median: the
   spectrum covers only part of the width, and the mean still shows it as a bump;
3. in each projection the 0.1 % and 99.9 % percentiles are taken and a level
   **25 %** of the way between them; the two crossings of that level are found
   outwards from the peak of the projection, and the distance between them is the
   width of the bump (crossings are sub-pixel);
4. the width-versus-angle array is Gaussian-filtered (5 samples) and its minimum
   taken. That angle is the direction of the spectrum, and the two crossings at
   that angle are the edges of the band.

The edges and the centre line are drawn over the image (never into the texture),
key `B`. **Curves** shows the width curve before and after filtering, the
projection at the chosen angle, and the range the curve covers - the range tells
you at a glance how pronounced the minimum is on this particular frame.

Two things worth knowing in use:

* the crossings are searched **outwards from the peak**, not inwards from the ends
  of the array. Otherwise stray light (the zero order, a reflection arc) crosses
  the level too and what gets measured is the distance to it rather than the width
  of the band - which made the answer depend on where the crop line was, down to
  flipping the sign of the angle;
* the level is 25 %, not the midpoint: for a band with a flat top the
  half-maximum points barely move when the edges blur, so the width curve comes
  out flat and the minimum is picked by noise.

The crop has to be tall enough: a tilt of `L` costs the projection
`crop_width * tan(L)` rows, and the band plus some background has to fit in what
is left. Otherwise the large angles have no background left and the curve shoots
up at its ends.

## Calibration: shear and the basis

Right panel, **Shear: spectrum basis**, button `Find shear` (the band angle has to
be measured first). Implemented as specified in
[docs/TZ_Shear.md](docs/TZ_Shear.md): line directions from -10 to +10 degrees are
tried in 0.05 degree steps, and for each one a 1-D projection along the wavelength
axis is built - pixels are summed across the band width along the trial direction
and divided by how many were summed. The sharpness of a projection is the RMS of
the difference of two Gaussian blurs (scales 1 and 4, times the single
`blur scale` parameter); the sharpest projection gives the direction of the
spectral lines.

The result: the tilt of the lines against the frame Y axis, the shear as the
departure from perpendicularity, `du/dv` (how far the wavelength coordinate slides
per pixel across the band), and the basis itself - two unit vectors in frame
coordinates. The sharpness curve is shown next to it.

Nine **dashed** semi-transparent lines are drawn across the band along the found
direction: you can see what the search settled on, and the spectrum underneath
stays visible. They appear only after `Find shear`; `Find band` and
`Clear calibration` reset them, because the shear is measured relative to the
band.

Accuracy here is inherently lower than for the band angle: the lever arm is the
width of the band, not its length, so at a width of 125 px one pixel of smear is
already about 0.45 degrees.

## Extracting the spectrum

**Capture Spectra** (enabled once both the band angle and the shear are measured)
opens a window under the preview. Implemented as specified in
[docs/TZ_Spectrum.md](docs/TZ_Spectrum.md): the wavelength axis is walked and, at
every position, all pixels between the band edges are averaged along the
spectral-line vector; the ends, where a line does not fully cross the band, are
dropped.

The window shows a strip in which the brightness of a point is the value of the
spectrum (its height defaults to 1/20 of its width and can be dragged by the
handle underneath), and below it the same spectrum as a plot. The height of the
window itself is dragged by the handle on its top edge. While the window is open
the spectrum is re-extracted from every new frame.

**Average N** under the plot means the last N spectra are averaged - a plain
arithmetic mean - before anything is shown or written out. 1 is off. The history
is dropped whenever the spectrum stops covering the same columns, because
averaging across a moved crop would smear the lines rather than the noise.

## Wavelength calibration

The reference solar spectrum (Delbouille, Neven & Roland 1972, served by
BASS2000) lives in `data/solar_reference.csv`, 300-1130 nm at 0.1 nm; fetch or
refresh it with `python tools/fetch_reference.py`. It is drawn as a second strip
flush under ours, blurred to the instrument's resolution and **resampled into our
pixel columns** - the measured spectrum is never resampled.

**Calibrate Wavelength** turns on line identification; until it is on, the strips
ignore clicks. Click a line on the reference (a yellow line follows the cursor
there), then the same line on our spectrum (a green one follows there, joined to
the first by a sloped segment showing what is being tied to what). Three points
finish it. Undo takes back the last click of the session, Reset drops the lot.
Points and the fitted polynomial are kept in `settings.json`; a calibration is
either complete or absent, never half of one.

Once it stands: a wavelength scale under the plot, a colour strip showing the
visible range, and a cursor line running through the strips and the graph
together. Details and the open question of polynomial against a physical model
are in [docs/TZ_Wavelength.md](docs/TZ_Wavelength.md).

## Relative measurement, and getting the curve out

**Set baseline** keeps the current spectrum aside; everything is then read as a
per cent of it. Take one without the filter, put the filter in, and the graph is
its transmission curve. Zero and below in the baseline count as one so the
division always has something to divide by, and the graph is pinned to 0..100 %
because anything above the baseline is noise. Without a baseline the values are
per cent of the brightest sample.

**Export CSV** writes two columns, wavelength and per cent. **Export chart**
draws the same curve as a picture - PNG at 3840x2160 and SVG beside it, with the
grid, the colour strip under the axis, the edges of the visible range and every
identified line marked. Both land in `captures/`.

## Dark frames

**Make Dark** takes seven covered frames at the current gain and exposure,
medians them and files the result as `darks/dark_<gain>_<exposure key>.fit`,
where the key is the position on the exposure scale in whole per cent. Seven is
odd on purpose: the median of an even count averages the two middle samples and
would break the 16 ADU step of a 12-bit sensor read as RAW16.

**Use dark** follows what is on disk: it comes on by itself at startup and
whenever the gain or exposure lands on a combination that has one, and goes off
when it does not. **Use bias** stands in when there is no dark, taking a flat
level off every frame; it is off by default, and note that subtracting a pedestal
makes relative measurement meaningless wherever there is no light - the ratio
becomes noise divided by noise.

## Hardware notes (measured on this machine)

* **RAW16 from an ASI290MM is 12-bit data shifted left by 4.** Values are
  multiples of 16 and full scale is 65535 (`tools/probe_camera.py` prints "value
  step 16"), so normalising by 65535 is right and there are 4096 real levels.
* The camera here is plugged into a **USB 2.0 port** (`IsUSB3Host = False`):
  full-frame RAW16 runs at ~8 fps (~35 MB/s), RAW8 at ~16 fps. On USB 3.0 it will
  be several times faster. Binning does not help: on a 290MM it is done in
  software, the full frame still crosses the bus.
* `bandwidth` (ASI_BANDWIDTHOVERLOAD) and `high_speed` are under Advanced.
* Nothing in the program is model-specific: resolution, control ranges, pixel
  size, bit depth and the supported formats all come from the SDK. Colour cameras
  would need debayering, which is not implemented - hence "monochrome".

## Next

Which model the X -> wavelength mapping should be. A polynomial is what is
implemented; whether the physical grating formula describes this instrument
better is still open, and cannot be settled until there is a frame with enough
identified lines to hold points back from the fit and measure the residuals on
them. See [docs/TZ_Wavelength.md](docs/TZ_Wavelength.md).
