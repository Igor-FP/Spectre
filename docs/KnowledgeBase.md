# Knowledge base

Accumulated facts and traps. **Append-only**: add at the end of a section, do not
rewrite history - a wrong turn recorded here is worth more than a clean page.

## Camera and SDK

* **RAW16 from an ASI290MM is 12-bit shifted left by 4.** Values come back as
  multiples of 16, full scale 65535, 4096 real levels. Verified with
  `tools/probe_camera.py` ("value step 16"). Normalising the display by 65535 is
  correct.
* **The camera on this machine sits on a USB 2.0 port** (`IsUSB3Host = False`
  while `IsUSB3Camera = True`). Full-frame RAW16 = 8.2 fps, i.e. ~35 MB/s, which
  is exactly USB 2 throughput; RAW8 = 16 fps. **Binning does not help**: a 290MM
  has no hardware binning, the SDK bins on the host, the full frame still
  crosses the bus.
* **Control values live in the camera** and survive process restarts. Early
  versions of the app pushed the values from `settings.json` on connect and
  silently overwrote a carefully dialled-in exposure. Defaults for exposure/gain
  in `Settings` are `-1` = "keep what the camera has"; `--keep-camera` forces
  that regardless of what was saved.
* `ASIGetVideoData` is fine for short exposures; above ~1 s use
  `ASIStartExposure` / `ASIGetDataAfterExp` (`SNAP_THRESHOLD_US`). In snap mode
  the exposure is aborted when a control changes, otherwise a 30 s exposure makes
  the UI look frozen.
* Struct layout of `ASI_CAMERA_INFO` / `ASI_CONTROL_CAPS` matches plain ctypes
  natural alignment on MSVC x64 (240 and 248 bytes). `long` is 4 bytes there.

## UI stack

* `imgui-bundle` (Dear ImGui 1.92) is the live Python binding; `pyimgui` is
  frozen on ImGui 1.82. `imgui_bundle.python_backends.pygame_backend` exists and
  works with a pygame `OPENGL | DOUBLEBUF | RESIZABLE` window.
* **The stock `PygameRenderer` only forwards navigation keys.** Letters are
  commented out in its key map, so `imgui.is_key_pressed(imgui.Key.f)` never
  fires. `spectre/imgui_backend.py::SpectreRenderer` maps the whole keyboard and
  still feeds printable characters to text fields (the base class suppresses text
  input for any mapped key, which is why `process_event` is reimplemented rather
  than extended).
* ImGui 1.92 wants `imgui.ImTextureRef(tex_id)` in `imgui.image()`, not a raw int.
* The 16-bit frame is shown through a 65536-entry uint8 LUT and uploaded as a
  `GL_R8` texture with a swizzle to grey. **Mipmaps are needed**: without them
  thin spectral lines disappear when the view is zoomed out.
* Statistics, histogram and the auto stretch are measured **inside the crop**. On
  a real frame that changes the numbers a lot (mean 298 -> 1670 ADU, auto white
  point 11116 -> 17568 ADU) because a spectrograph throws glare and the zero
  order across the rest of the sensor.

## Band angle: what failed on real data

Every item here was measured on the frames in `captures/`.

* **Crossings searched inwards from the ends of the projection measure the
  outermost thing above the level, not the band.** With the parasitic arc inside
  the crop the "width" jumped to 570-620 px, and which angles got corrupted
  depended on where the left crop line was - the found angle flipped sign
  between crop positions. Fixed by searching outwards from the peak of the
  projection (`_crossing_outwards`), peak located on a copy smoothed over 9
  samples.
* **A width measured at the midpoint level is nearly invariant to the angle** for
  a flat-topped band: the half-maximum points do not move when the edges blur, so
  the curve is flat (147.9..161.4 px over +-10 deg) and the minimum is chosen by
  noise. Hence the 25 % level.
* **A scan line that runs off the crop** is averaged over fewer columns; at large
  angles no background rows are left at all, the level lands inside the smeared
  band, and the reported "width" becomes the whole height of the crop (the
  vertical jump at the end of the curve). Geometric requirement:
  `crop_height > band_width + crop_width * tan(range) + background margin`.
* Variants tried and **rejected**, with the numbers that killed them:
  * *exclude the rows whose scan line leaves the crop*: the row window then
    slides with the angle, the 0.1/99.9 percentiles are computed over a different
    set of rows for every angle, the whole curve tilts and the minimum slides to
    an end (-8.32 deg on a frame whose real angle was about -1.1);
  * *one fixed level taken at 0 deg*: as the band smears its peak drops towards
    the fixed level and the width at that level shrinks to zero, so the minimum
    lands at maximum smear (-10 deg on almost every frame);
  * *one fixed row window valid at every angle*: at the ends of the range the
    smeared band no longer fits in what is left of the window, same failure
    (-10 deg).
* The scan **range and step change only the run time**, not the answer: at the
  same crop, +-10 deg and +-2 deg give identical angles (1.2 s vs 0.1 s per
  frame); step 0.002 deg costs 5.8 s per frame and changes nothing. The Gaussian
  filter window is in samples, so a much finer step weakens it proportionally.

## Method and measurement discipline

* **The simulator is not representative.** Its band is uniform, spans the full
  width, has straight parallel edges and no glare. Conclusions about the method
  drawn on it were wrong twice. Use it only to check that code runs and that the
  UI draws.
* **Do not compare angles between frames as a measure of repeatability** unless
  it is known that nothing but the code changed: the camera and the source get
  moved deliberately, so that the solution does not only work at one angle.
* Real illumination along the slit is not constant, so the band is not uniform
  along its length, while its edges stay sharp. That is why the width of the
  projection tolerates the non-uniformity and metrics based on structure *along*
  the band (gradient energy, centroid fits) do not.

## Environment

* Windows Python can be run from WSL directly (the exact path is in
  `CLAUDE.local.md`); a pygame/OpenGL window opens on the Windows desktop.
* A Linux environment variable does **not** reach a Windows process started from
  WSL unless it is listed in `WSLENV`. `SPECTRE_SETTINGS` therefore did not work
  that way; copy `settings.json` aside and restore it instead.
* `--screenshot PATH.png` + `--screenshot-after N` is the only way to see the UI
  from here. Reading the PNG back shows geometry and panel text reliably; fine
  pixel detail and exact curve values are not readable.
* There is no numpy in the WSL python; anything that imports `spectre` has to run
  under the Windows interpreter (path in `CLAUDE.local.md`). It reaches WSL files
  through `\\wsl.localhost\<distro>\...`, so a test script can live in a scratch
  directory outside the repository.

## Dark frames

* A dark is only valid for the gain and exposure it was taken at, so that is what
  names it: `darks/dark_<gain>_<exposure key>.fit`. The exposure key is the
  position on the exposure scale in whole per cent, 1 ms = 0, 1 s = 50, 10 min =
  100. **The formula lives in `camera.py`, not in the UI**, exactly so that a
  file name cannot depend on a widget; the slider now calls the same functions.
  Bucketing to a per cent puts exposures within about 1.4 % in the same file.
* **Combine an odd number of frames.** The median of an even count averages the
  two middle samples, and a 12-bit sensor read as RAW16 comes back in steps of
  16 - averaging two of them lands on a multiple of 8 and quietly breaks the
  invariant the clipping test relies on. `DARK_FRAME_COUNT` is 7.
* `np.partition(stack, N//2, axis=0)[N//2]` is the median without numpy's promotion
  of the whole stack to float64 - which on a full-frame stack of 7 would be
  120 MB of temporaries for nothing.
* The master dark is a **whole frame**, never a crop: the crop moves, the dark
  cannot follow it.

## Plotting and pictures

* **`PlotLines` puts up its own tooltip** with the two ends of the segment under
  the cursor, in raw units, and there is no flag to turn it off. A `set_tooltip`
  called after it replaces it - ImGui's `SetTooltip` overrides a previous tooltip
  in the same frame.
* **`PlotLines` insets the curve by `frame_padding`**, 4 px each side by default,
  so a graph drawn under a strip of the same width does not line up with it.
  Pushing `StyleVar_.frame_padding` to zero fixes it. What is left is a one-pixel
  difference at the ends: the strips place sample i at `(i+0.5)/N` of the width,
  ImGui places it at `i/(N-1)`.
* **matplotlib must be forced onto the Agg backend** before `pyplot` is imported,
  and imported inside the function that needs it. Otherwise it can try to open a
  window of its own next to the live OpenGL one, and it costs about a second to
  import at startup for nothing.
* A drag widget (`drag_int`, `drag_float`) is indistinguishable from a button in
  this theme and needs a ctrl-click to type into. Numbers to be typed get an
  `input_int` with the label drawn beside it.

## Reference solar spectrum

* **BASS2000 caps one request at 1000 A**, so 300-1130 nm takes nine of them.
  Neighbouring chunks repeat their shared end point - deduplicate by wavelength.
  `resol` is the output step in angstroms and the server resamples to it.
* The service goes **well past 1000 nm** - at least to 1320 nm, where the visible
  atlas hands over to Delbouille, Roland, Brault & Testerman (1981). The join is
  clean: continuum 9860 just below 1000 nm against 9919 just above, 0.6 % apart.
* **The grid is not perfectly uniform.** The server inserts one extra sample at
  the exact catalogue wavelength of every labelled line, off the 0.1 nm grid -
  35 steps out of 8316 differ, the smallest being 0.004 nm. It stays strictly
  increasing, so interpolation is unaffected; the Gaussian blur treats samples as
  evenly spaced, so a labelled line is weighted about 2 % heavier than it should
  be at its own centre.  Cosmetic at these resolutions, but it is why the file is
  not the clean 0.1 nm ladder it looks like.
* **The response is UTF-8, not ASCII**: the hydrogen labels come as `Halpha`,
  `Hbeta`... spelled with real Greek letters, and G-prime with a real prime.
  Decoding the body as ASCII destroys them before they can be transliterated -
  decode UTF-8 first, spell out afterwards.  `tools/fetch_reference.py` does it.
* The atlas grid is 0.1 nm, roughly a quarter of one pixel of this spectrograph.
  Blur it in wavelength first (that is where the instrument resolution lives),
  and linear interpolation onto our columns costs nothing after that.
* Blurring the file and measuring the depth of H-alpha from the local continuum
  gives 68 % unblurred, 31.5 % at FWHM 1 nm, 19.5 % at 2 nm, 13.8 % at 3 nm -
  which matches the anchor table in `docs/TZ_Wavelength.md`, measured separately.
* **The reference strip locates a line only to +-0.5 px**: it is resampled onto
  our columns, so its minimum snaps to a column. At 0.57 nm/px that is 0.28 nm.
  Nothing to fix - it is the click precision, and the reason the fit wants
  several points rather than two good ones.
