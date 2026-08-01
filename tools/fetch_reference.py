"""Download the reference solar spectrum used by the wavelength calibration.

Source: the Solar Spectrum service of the Paris-Meudon observatory (BASS2000),
serving the atlas of Delbouille, Neven & Roland (1972), Jungfraujoch, disc
centre.  The service caps one request at 1000 A, so the range is fetched in
chunks and stitched together.

    python tools/fetch_reference.py                  # 300-1000 nm, 0.1 nm step
    python tools/fetch_reference.py --step-nm 0.05
    python tools/fetch_reference.py --out other.csv

The output is the CSV `spectre/reference.py` expects: wavelength in nm,
intensity normalised to the continuum, and the line label the service gives
where it has one.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "https://bass2000.obspm.fr/php/getSolarSpectrumDB.php"

#: Largest window the service accepts in one request, in angstroms.
CHUNK_A = 1000.0

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "solar_reference.csv"
)

#: The service labels the hydrogen lines with Greek letters and writes G-prime
#: with a real prime.  Everything downstream - the CSV, the panel, the console -
#: stays ASCII, so they are spelled out on the way in.  The keys are escapes for
#: the same reason: one non-ASCII byte in a .py breaks redirection and --help
#: under a cp1251 console.
TRANSLITERATE = {
    "\u03b1": "alpha",
    "\u03b2": "beta",
    "\u03b3": "gamma",
    "\u03b4": "delta",
    "\u03b5": "epsilon",
    "\u2032": "'",
    "\u2033": '"',
}


def to_ascii(text: str) -> str:
    for source, replacement in TRANSLITERATE.items():
        text = text.replace(source, replacement)
    return text.encode("ascii", errors="replace").decode("ascii")


def fetch_chunk(start_a: float, width_a: float, step_a: float, timeout: float) -> str:
    url = (
        f"{ENDPOINT}?WL={start_a:.1f}&DW={width_a:.1f}"
        f"&resol={step_a:g}&fmt=txt"
    )
    with urllib.request.urlopen(url, timeout=timeout) as response:
        # The service sends UTF-8; the Greek letters in the labels have to
        # survive this far, `to_ascii` spells them out afterwards.
        return response.read().decode("utf-8", errors="replace")


def parse(text: str):
    """(wavelength_A, intensity, label) for every data line of one response."""
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            wavelength = float(parts[0])
            intensity = float(parts[1])
        except ValueError:
            continue  # the header line, or whatever the server prepends
        label = to_ascii(" ".join(parts[2].split())) if len(parts) > 2 else ""
        rows.append((wavelength, intensity, label))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-nm", type=float, default=300.0)
    parser.add_argument("--to-nm", type=float, default=1000.0)
    parser.add_argument(
        "--step-nm", type=float, default=0.1, help="output sampling step, nm (default 0.1)"
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    step_a = args.step_nm * 10.0
    start_a, end_a = args.from_nm * 10.0, args.to_nm * 10.0

    collected = {}
    labels = {}
    position = start_a
    while position < end_a:
        width = min(CHUNK_A, end_a - position)
        print(f"{position / 10:.0f}-{(position + width) / 10:.0f} nm ...", end=" ", flush=True)
        for attempt in range(3):
            try:
                text = fetch_chunk(position, width, step_a, args.timeout)
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"retry ({exc})", end=" ", flush=True)
                time.sleep(2.0 * (attempt + 1))
        else:
            print("failed")
            return 1
        rows = parse(text)
        print(f"{len(rows)} points")
        for wavelength, intensity, label in rows:
            key = round(wavelength, 4)
            collected[key] = intensity  # chunks share their end points
            if label:
                labels[key] = label
        position += width

    if not collected:
        print("nothing downloaded")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="ascii", newline="\n") as handle:
        handle.write("# Reference solar spectrum, disc centre.\n")
        handle.write("# Delbouille, Neven & Roland (1972), Jungfraujoch; served by BASS2000\n")
        handle.write("# (bass2000.obspm.fr).  Intensity is linear, continuum = 10000.\n")
        handle.write("wavelength_nm,intensity,label\n")
        for key in sorted(collected):
            handle.write(f"{key / 10:.4f},{collected[key]:.0f},{labels.get(key, '')}\n")

    print(
        f"wrote {len(collected)} points to {args.out} "
        f"({os.path.getsize(args.out) / 1024:.0f} KB), {len(labels)} labelled"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
