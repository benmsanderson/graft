"""Fetch LUH3's wood-harvest variables without downloading the transitions file.

mrdownscale's nonland path reads wood harvest out of LUH3's historic
transitions: twelve variables (`<source>_harv` and `<source>_bioh`) for the
harmonization's history, out of a 17.4 GB file holding every transition from
850. THREDDS can subset it server-side, and harvest fields are mostly zeros,
so the years that matter come to about 75 MB - worth having on a metered
connection, and quick enough to redo.

One request for all twelve variables times the server out (it scans the whole
file), so each variable is fetched separately and they are merged here. The
result is a netCDF holding only those variables and years; the fork's
readLUH3 selects layers by the file's own time axis, so it reads either this
or the published file.

Needs xarray and netCDF4:

    python scripts/fetch_luh3_harvest.py ~/madrat/sources/LUH3
    python scripts/fetch_luh3_harvest.py DIR --years 1994 2023 --jobs 3
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NCSS = ("https://esgf-node.ornl.gov/thredds/ncss/grid/user_pub_work/input4MIPs/CMIP7/CMIP/"
        "UofMD/UofMD-landState-3-1-1/land/yr/multiple-transitions/gn/v20250325/"
        "multiple-transitions_input4MIPs_landState_CMIP_UofMD-landState-3-1-1_gn_0850-2023.nc")
SOURCES = ("primf", "primn", "secmf", "secyf", "secnf", "pltns")
# _harv is harvested area as a fraction of the cell, _bioh the carbon taken
VARIABLES = tuple(f"{s}_{kind}" for s in SOURCES for kind in ("harv", "bioh"))


def fetch(variable: str, first: int, last: int, into: Path, tries: int = 4) -> Path:
    """One variable, retried: the server 504s intermittently, more so when
    asked for several variables at once, and each request makes it scan the
    17.4 GB file again."""
    path = into / f"{variable}_{first}-{last}.nc"
    if path.exists():
        return path
    url = (f"{NCSS}?var={variable}"
           f"&time_start={first}-01-01T00:00:00Z&time_end={last}-01-01T00:00:00Z&accept=netcdf4")
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=1200) as r:
                body = r.read()
            if not body.startswith(b"\x89HDF"):  # an error page, not netCDF4
                raise RuntimeError(f"server returned {len(body)} bytes, not netCDF: {body[:120]!r}")
            path.write_bytes(body)
            print(f"  {variable}: {len(body) / 1e6:.1f} MB", flush=True)
            return path
        except Exception as error:
            if attempt == tries:
                raise RuntimeError(f"{variable}: {error}") from error
            wait = 30 * attempt
            print(f"  {variable}: {type(error).__name__}, retrying in {wait}s "
                  f"({attempt}/{tries - 1})", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dest", help="LUH3 source folder, e.g. ~/madrat/sources/LUH3")
    ap.add_argument("--years", nargs=2, type=int, default=(1994, 2023), metavar=("FIRST", "LAST"),
                    help="LUH transitions are from-year, so 1994-2023 covers 1995-2024 (default)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel requests; more than one makes the server time out "
                         "(default 1)")
    args = ap.parse_args(argv)

    import xarray as xr

    first, last = args.years
    dest = Path(args.dest).expanduser()
    parts = dest / ".harvest_parts"
    parts.mkdir(parents=True, exist_ok=True)
    out = dest / f"multiple-transitions_woodharvest_{first}-{last}.nc"
    if out.exists():
        print(f"{out} exists already")
        return 0

    print(f"fetching {len(VARIABLES)} wood harvest variables, {first}-{last}, "
          f"{args.jobs} at a time; about a minute each")
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            paths = list(pool.map(lambda v: fetch(v, first, last, parts), VARIABLES))
    except Exception as error:
        print(f"\n{error}", file=sys.stderr)
        print("partial files are kept; rerun to continue", file=sys.stderr)
        return 1

    merged = xr.merge([xr.open_dataset(p, decode_times=False) for p in paths])
    # keep the published file's layout: merging reorders the dimensions, and
    # terra then reads the grid through GDAL's geolocation arrays instead of a
    # geotransform, losing the time axis when the whole file is opened at once
    merged = merged.transpose("time", "lat", "lon", ...)
    for name in merged.data_vars:
        merged[name].encoding.update(zlib=True, complevel=4)
    merged.to_netcdf(out)
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.0f} MB, "
          f"{len(merged.data_vars)} variables, {merged.sizes.get('time')} years)")
    print("the per-variable parts are kept in", parts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
